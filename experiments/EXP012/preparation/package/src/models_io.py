"""Frozen-weight loading and causal windows; no graph rebuilding."""
import io,json,zipfile,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from models_exp007 import MessageGatedGCN
from v2_models_exp005 import CommandAwareTCN
from lstm_baseline import ControlledLSTM

class Windows(Dataset):
    def __init__(self,raw,features,start,end,lookback,offset=0):
        self.raw=raw;self.features=features;self.start=start;self.end=end;self.lookback=lookback;self.offset=offset
        if not offset+lookback<=start<end<=offset+len(raw): raise ValueError('Invalid causal interval')
    def __len__(self):return self.end-self.start
    def __getitem__(self,i):
        t=self.start+i-self.offset
        x=np.ascontiguousarray(self.raw[t-self.lookback:t,self.features].T,dtype=np.float32)
        return torch.from_numpy(x),torch.tensor(self.raw[t,0],dtype=torch.float32)

def loader(raw,features,start,end,lookback,batch=256,shuffle=False,seed=17,offset=0):
    g=torch.Generator().manual_seed(seed)
    return DataLoader(Windows(raw,features,start,end,lookback,offset),batch_size=batch,shuffle=shuffle,
                      generator=g,num_workers=0,pin_memory=False)
def standardize(raw,mean,scale):
    z=raw.astype(np.float32,copy=True);z[:,0]=(z[:,0]-mean)/scale;return z
def read_parent(root,meta):
    from integrity import sha,digest
    if sha(root/meta['archive'])!=meta['archive_sha256']:raise RuntimeError('Parent archive hash differs')
    if digest(json.loads((root/'provenance/RUN_FREEZE.json').read_text()))!=meta['parent_run_hash']:
        raise RuntimeError('Parent run freeze hash differs')
    with zipfile.ZipFile(root/meta['archive']) as z:
        ck=torch.load(io.BytesIO(z.read('checkpoint.pt')),map_location='cpu',weights_only=True)
        r=json.loads(z.read('result.json'));j=json.loads(z.read('job.json'))
    if r!=meta['reference_result'] or r!=ck['result'] or j!=meta['job'] or j!=ck['job'] or ck['run_hash']!=meta['parent_run_hash']:
        raise RuntimeError('Parent checkpoint/result identity mismatch')
    if j['gate_mode']!='fixed' or j['fold']!='canonical': raise RuntimeError('Only frozen canonical fixed/removal parents allowed')
    config=json.loads((root/'provenance/effective_config.json').read_text())
    if ck['config']!=config or ck['spec']!=config['baseline_by_mission'][j['mission']]:
        raise RuntimeError('Frozen parent specification changed')
    graph=json.loads((root/'provenance/graph_manifest.json').read_text())
    g=next(g for g in graph if g['channel']==j['channel'] and g['fit_fraction']==.6)
    if r['graph_features']!=g['features'] or digest(ck['state_dict']['adjacency'].tolist())!=g['adjacency_sha256']:
        raise RuntimeError('Frozen graph identity changed')
    s=ck['spec'];model=MessageGatedGCN(r['graph_num_nodes'],r['hidden_size'],ck['state_dict']['adjacency'].numpy(),s['kernel_size'],s['dilations'],'fixed',j['remove_messages'])
    model.load_state_dict(ck['state_dict'],strict=True)
    return model,ck,r
def read_tcn(path,row,features):
    ck=torch.load(path,map_location='cpu',weights_only=True);r=ck['result'];c=ck['candidate']
    actual=json.loads(r['selected_feature_columns'])
    if r['channel']!=row['channel'] or r['mission']!=row['mission'] or actual!=features or r['num_values']!=row['rows']:
        raise RuntimeError('EXP005 identity/support disagrees with frozen GCN input contract')
    expected='c03_rf7_large_lr1e3' if row['mission']=='MSL' else 'c11_rf31_large_lr1e3'
    if c['candidate_id']!=expected or r['candidate_id']!=expected or r['seed'] not in [17,29,43]:raise RuntimeError('Wrong selected EXP005 checkpoint')
    original=json.loads((Path(__file__).resolve().parents[1]/'provenance/EXP005_original_config.json').read_text())
    if ck['experiment_config']!=original or c!=next(v for v in original['candidates'] if v['candidate_id']==expected):
        raise RuntimeError('Original EXP005 specification differs')
    spec=ck['experiment_config']['dataset']
    if spec['fit_fraction']!=.6 or spec['validation_end_fraction']!=.8:raise RuntimeError('Wrong EXP005 split')
    if (r['fit_samples'],r['validation_samples'],r['calibration_samples'])!=(row['fit_end']-r['lookback'],row['stop_end']-row['fit_end'],row['rows']-row['stop_end']):
        raise RuntimeError('EXP005 sample boundaries differ')
    model=CommandAwareTCN(len(features),r['hidden_size'],r['kernel_size'],json.loads(r['dilations']))
    model.load_state_dict(ck['state_dict'],strict=True)
    return model,ck,r

@torch.no_grad()
def predict(model,raw,row,features,start,end,lookback,device='cuda',offset=0):
    pred=[];total=0.;n=0;model.eval();criterion=torch.nn.MSELoss()
    for xb,yb in loader(raw,features,start,end,lookback,offset=offset):
        xb=xb.to(device);yb=yb.to(device);p=model(xb);loss=criterion(p,yb)
        if not torch.isfinite(p).all() or not torch.isfinite(loss):raise RuntimeError('Nonfinite forecast')
        total+=loss.item()*len(p);n+=len(p);pred.append(p.cpu().numpy())
    return np.concatenate(pred),total/n

def tensor_hash(state):
    import hashlib
    h=hashlib.sha256()
    for key in sorted(state):
        t=state[key].detach().cpu().contiguous();h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()

def seeds(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def train_lstm(raw,row,features,seed,settings,progress,identity,device='cuda'):
    """Resume at epoch boundaries with optimiser and all relevant RNG states."""
    from integrity import atomic,sha,write_json
    import time
    seeds(seed);model=ControlledLSTM(len(features)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'],foreach=False,fused=False)
    lookback=settings['lookback']
    fit=loader(raw,features,lookback,row['fit_end'],lookback,batch=settings['batch_size'],shuffle=True,seed=seed)
    history=[];best=float('inf');best_state=None;wait=0;next_epoch=1;elapsed=0.
    if progress.exists():
        pointer=json.loads(progress.read_text());epoch_path=progress.parent/pointer['file']
        if Path(pointer['file']).name!=pointer['file'] or sha(epoch_path)!=pointer['sha256']:
            raise RuntimeError('LSTM epoch progress corrupted')
        ck=torch.load(epoch_path,map_location='cpu',weights_only=True)
        if ck['identity']!=identity:raise RuntimeError('LSTM progress identity mismatch')
        model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to(device)
        history=ck['history'];best=ck['best'];best_state=ck['best_state'];wait=ck['wait'];next_epoch=ck['next_epoch'];elapsed=ck['elapsed']
        torch.set_rng_state(ck['torch_rng']);fit.generator.set_state(ck['loader_rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(ck['cuda_rng'])
        # Training uses Torch randomness only. Python/NumPy are seeded and not sampled afterward.
    clock=time.perf_counter()
    for epoch in range(next_epoch,settings['max_epochs']+1):
        if wait>=settings['patience']:break
        model.train();total=0.;n=0
        for xb,yb in fit:
            optimizer.zero_grad(set_to_none=True);xb=xb.to(device);yb=yb.to(device)
            loss=torch.nn.functional.mse_loss(model(xb),yb)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite LSTM training loss')
            loss.backward();optimizer.step();total+=loss.item()*len(yb);n+=len(yb)
        _,val=predict(model,raw,row,features,row['fit_end'],row['stop_end'],lookback,device)
        improved=val<best-settings['min_delta']
        if improved:best=val;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};wait=0
        else:wait+=1
        history.append({'epoch':epoch,'fit_mse':total/n,'validation_mse':val,'selected':improved})
        ck={'identity':identity,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'optimizer':optimizer.state_dict(),'best_state':best_state,'best':best,'wait':wait,'next_epoch':epoch+1,
            'history':history,'torch_rng':torch.get_rng_state(),'loader_rng':fit.generator.get_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],'elapsed':elapsed+time.perf_counter()-clock}
        stream=io.BytesIO();torch.save(ck,stream)
        epoch_path=progress.parent/f'epoch_{epoch:03d}.pt';atomic(epoch_path,stream.getvalue())
        write_json(progress,{'file':epoch_path.name,'sha256':sha(epoch_path),'identity':identity,'next_epoch':epoch+1})
        for older in progress.parent.glob('epoch_*.pt'):
            if older!=epoch_path:older.unlink()
        print(f"LSTM {row['channel']} seed{seed} epoch{epoch}: validation={val:.8g}",flush=True)
    if best_state is None:raise RuntimeError('No eligible LSTM checkpoint')
    model.load_state_dict(best_state);pred,val=predict(model,raw,row,features,row['fit_end'],row['stop_end'],lookback,device)
    if val!=best:raise RuntimeError('Selected LSTM replay mismatch')
    result={'channel':row['channel'],'mission':row['mission'],'seed':seed,'validation_mse':val,
            'best_epoch':next(h['epoch'] for h in reversed(history) if h['selected']), 'epochs_ran':len(history),
            'hit_ceiling':len(history)==settings['max_epochs'],'parameters':sum(p.numel() for p in model.parameters()),
            'elapsed_seconds':elapsed+time.perf_counter()-clock,'features':features,'lookback':lookback}
    return model,{'state_dict':best_state,'result':result,'settings':settings,'identity':identity},history,pred
