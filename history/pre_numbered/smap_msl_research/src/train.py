"""Train/tune exclusively on supplied TRAIN arrays. No test arrays or labels used.

Chronological 60/20/20 fit/early-stop/calibration; previous observations can be
context at boundaries but no target belongs to two splits. No cross-file windows.
"""
import argparse, copy, time, platform, sys
import numpy as np
import pandas as pd
import torch
from common import ROOT,PROTOCOL,load,dump,sha
from models import adjacency,make

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

def prepare(ch,L):
    raw=load(ch,'train');n=len(raw);a=int(n*.6);b=int(n*.8)
    active=np.r_[0,np.flatnonzero(raw[:a,1:].std(0)>0)+1]
    x=raw[:,active].copy();mu=float(x[:a,0].mean());sd=float(x[:a,0].std())
    constant=sd<1e-7
    if constant: sd=1.0
    x[:,0]=(x[:,0]-mu)/sd
    windows=np.lib.stride_tricks.sliding_window_view(x,L,axis=0)[:-1].copy()
    targets=x[L:,0].copy();times=np.arange(L,n)
    return x,torch.tensor(windows),torch.tensor(targets),times,dict(active=active.tolist(),mean=mu,scale=sd,constant_target=constant,fit_end=a,validation_end=b,n_train=n)

@torch.no_grad()
def predict(model,x,batch=1024):
    model.eval();return torch.cat([model(z) for z in x.split(batch)]).numpy()

def run(ch,cfg,seed=17):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    x,X,y,t,meta=prepare(ch,cfg['lookback'])
    a=adjacency(x[:meta['fit_end']],cfg['graph']);model=make(cfg,x.shape[1],a)
    tr=np.flatnonzero(t<meta['fit_end']);va=np.flatnonzero((t>=meta['fit_end'])&(t<meta['validation_end']))
    assert len(tr)>0 and len(va)>0
    optimizer=torch.optim.AdamW(model.parameters(),lr=PROTOCOL['learning_rate'],weight_decay=PROTOCOL['weight_decay'])
    best=float('inf');state=None;bad=0;history=[];start=time.time()
    for epoch in range(PROTOCOL['epochs']):
        model.train();loss_sum=0
        for idx in np.array_split(rng.permutation(tr),max(1,int(np.ceil(len(tr)/PROTOCOL['batch_size'])))):
            optimizer.zero_grad();loss=(model(X[idx])-y[idx]).square().mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();loss_sum+=float(loss.detach())*len(idx)
        val=float(np.mean((predict(model,X[va])-y[va].numpy())**2));history.append(dict(epoch=epoch+1,train_mse=loss_sum/len(tr),val_mse=val))
        if val < best-1e-7:
            best=val;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;bad=0
        else: bad+=1
        if bad>=PROTOCOL['patience']:break
    model.load_state_dict(state)
    out=ROOT/'checkpoints'/cfg['name']/f'{ch}_s{seed}.pt';out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=state,config=cfg,metadata=meta,seed=seed,adjacency=a),out)
    pred=predict(model,X)
    np.savez_compressed(out.with_suffix('.npz'),times=t,pred=pred*meta['scale']+meta['mean'],truth=y.numpy()*meta['scale']+meta['mean'])
    dump(out.with_suffix('.json'),dict(history=history,metadata=meta,configuration=cfg,seed=seed))
    return dict(channel=ch,configuration=cfg['name'],seed=seed,val_mse=best,best_epoch=best_epoch,epochs=len(history),parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-start,nodes=x.shape[1],edges=int((a>0).sum()),checkpoint=str(out.relative_to(ROOT)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--configs',default='all');ap.add_argument('--channels',default='all');ap.add_argument('--seed',type=int,default=17);args=ap.parse_args()
    # Mission metadata only. Label intervals never read by this training program.
    inv=pd.read_csv(ROOT/'audit/data_inventory.csv').drop_duplicates('channel').sort_values('channel')
    configs=PROTOCOL['graph_candidates']+PROTOCOL['controls']
    if args.configs!='all':configs=[c for c in configs if c['name'] in args.configs.split(',')]
    if args.channels!='all':inv=inv[inv.channel.isin(args.channels.split(','))]
    path=ROOT/'results/training.csv';rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    done={(r['channel'],r['configuration'],r['seed']) for r in rows}
    for cfg in configs:
        for r in inv.itertuples():
            if (r.channel,cfg['name'],args.seed) in done: continue
            result=run(r.channel,cfg,args.seed);result['spacecraft']=r.spacecraft;rows.append(result)
            pd.DataFrame(rows).to_csv(path,index=False)
            print(f"{cfg['name']} {r.channel} seed={args.seed} val={result['val_mse']:.5f} epoch={result['best_epoch']} {result['seconds']:.1f}s",flush=True)
    dump(ROOT/'results/environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),cuda=torch.cuda.is_available(),protocol_sha256=sha(ROOT/'configs/protocol.json')))

if __name__=='__main__':main()
