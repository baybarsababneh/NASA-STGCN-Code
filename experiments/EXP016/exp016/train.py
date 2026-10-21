import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import io, hashlib, platform, random, time
from pathlib import Path
import numpy as np
import torch
from .model import STGCN
from .data import Windows,splits
from .io import atomic,save_json,read_json,commit,committed,sha

class BudgetReached(Exception):pass

def configure(device):
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Scientific run requires a CUDA GPU. In Colab select Runtime > Change runtime type > A100 GPU. CPU is supported only for engineering fixtures.')

def environment(device):
    import pandas,h5py,more_itertools
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pandas.__version__,
        h5py=h5py.__version__,more_itertools=more_itertools.__version__,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
        deterministic=True,tf32=False,amp=False)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    if s['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])
def save_state(path,state):
    b=io.BytesIO();torch.save(state,b);payload=b.getvalue()
    envelope=io.BytesIO();torch.save({'sha256':hashlib.sha256(payload).hexdigest(),'payload':payload},envelope)
    p=Path(path)
    if p.exists():atomic(p.with_suffix('.previous.pt'),p.read_bytes())
    atomic(p,envelope.getvalue())
def load_state(path,device):
    # Only load checkpoints produced by this trusted package, never arbitrary uploads.
    p=Path(path);fail=[]
    for q in (p,p.with_suffix('.previous.pt')):
        if not q.exists():continue
        try:
            e=torch.load(q,map_location='cpu',weights_only=False)
            if hashlib.sha256(e['payload']).hexdigest()!=e['sha256']:raise ValueError('Checkpoint checksum mismatch')
            return torch.load(io.BytesIO(e['payload']),map_location=device,weights_only=False)
        except Exception as ex:fail.append(str(q)+': '+str(ex))
    raise RuntimeError('No valid checkpoint generation: '+'; '.join(fail))

def train_one(array,c,seed,folder,identity,device='cuda',deadline=float('inf'),mirror=None,stop_after_steps=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if committed(folder,identity):return read_json(folder/'history.json')
    fit,val=splits(len(array),c);seed_all(seed)
    model=STGCN(array.shape[1],c['hidden'],c['history'],c['horizon'],c['dropout']).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],betas=(.9,.999),eps=c['adam_epsilon'],foreach=False)
    windows=Windows(array,c,device)
    state=dict(identity=identity,epoch=0,cursor=0,steps=0,permutation=None,best=float('inf'),wait=0,
               history=[],epoch_sum=0.,epoch_count=0,elapsed_seconds=0.)
    last=folder/'last.pt'
    if last.exists():
        state=load_state(last,device)
        if state['identity']!=identity:raise RuntimeError('Training state identity mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    session_start=time.perf_counter();base_time=state['elapsed_seconds']
    def checkpoint():
        state.update(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),
                     elapsed_seconds=base_time+time.perf_counter()-session_start)
        save_state(last,state)
        if mirror:mirror.sync()
    try:
        while state['epoch']<c['epochs'] and state['wait']<c['patience']:
            model.train()
            if state['permutation'] is None:state['permutation']=np.random.permutation(fit)
            order=state['permutation']
            while state['cursor']<len(order):
                if time.monotonic()>=deadline:checkpoint();raise BudgetReached('Session time budget reached; saved exact next training step.')
                starts=order[state['cursor']:state['cursor']+c['batch_size']]
                x,y=windows.batch(starts);opt.zero_grad(set_to_none=True)
                pred=model(x);loss=torch.mean((pred-y)**2)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
                opt.step();state['cursor']+=len(starts);state['steps']+=1
                state['epoch_sum']+=float(loss.detach())*len(starts);state['epoch_count']+=len(starts)
                if state['steps']%c['checkpoint_steps']==0:checkpoint()
                if stop_after_steps is not None and state['steps']>=stop_after_steps:
                    checkpoint();return {'interrupted':True,'steps':state['steps']}
            model.eval();vl=0.
            with torch.no_grad():
                for j in range(0,len(val),c['batch_size']):
                    x,y=windows.batch(val[j:j+c['batch_size']]);v=torch.mean((model(x)-y)**2)
                    if not torch.isfinite(v):raise FloatingPointError('Nonfinite validation loss')
                    vl+=float(v)*len(x)
            vl/=len(val)
            improved=vl < state['best']-c['min_delta']
            if improved:
                state['best']=vl;state['wait']=0
                save_state(folder/'best_validation.pt',{'model':model.state_dict(),'epoch':state['epoch']+1,'val_mse':vl,'identity':identity})
            else:state['wait']+=1
            state['history'].append(dict(epoch=state['epoch']+1,train_mse=state['epoch_sum']/state['epoch_count'],val_mse=vl,
                early_stop_wait=state['wait'],steps=state['steps']))
            state['epoch']+=1;state['cursor']=0;state['permutation']=None;state['epoch_sum']=0.;state['epoch_count']=0
            checkpoint()
            print(f'{folder.name}: epoch {state["epoch"]}/{c["epochs"]}, val MSE={vl:.7g}',flush=True)
        # Upstream saves termination weights, NOT best-validation weights.
        save_state(folder/'final.pt',{'model':model.state_dict(),'identity':identity,'epoch':state['epoch'],
                     'selected':'termination_weights_matching_author_code'})
        hist=dict(epochs=state['epoch'],steps=state['steps'],fit_windows=len(fit),validation_windows=len(val),
            parameters=sum(p.numel() for p in model.parameters()),seconds=state['elapsed_seconds'],history=state['history'])
        save_json(folder/'history.json',hist);commit(folder,['final.pt','best_validation.pt','last.pt','history.json'],identity)
        if mirror:mirror.sync(force=True)
        return hist
    except (KeyboardInterrupt,BudgetReached):
        # Budget saves at a complete optimizer-step boundary; KeyboardInterrupt rolls
        # back to the latest atomic checkpoint, never writes a half-executed step.
        raise

@torch.no_grad()
def predict(array,c,checkpoint,device):
    s=load_state(checkpoint,device);model=STGCN(array.shape[1],c['hidden'],c['history'],c['horizon'],c['dropout']).to(device)
    model.load_state_dict(s['model']);model.eval();w=Windows(array,c,device)
    n=len(array)-c['history']-c['horizon'];res=[]
    for j in range(0,n,c['prediction_batch_size']):
        x,_=w.batch(np.arange(j,min(n,j+c['prediction_batch_size'])));res.append(model(x).cpu().numpy())
    a=np.concatenate(res)
    if a.shape!=(n,c['horizon']) or not np.isfinite(a).all():raise ValueError('Invalid forecast output')
    return a
