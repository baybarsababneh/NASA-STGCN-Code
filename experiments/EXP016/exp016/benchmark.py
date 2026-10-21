import time, tempfile
import numpy as np
import torch
from .model import STGCN
from .data import Windows,splits
from .train import seed_all,environment,save_state
from .io import save_json

def benchmark(root,rows,c,out,device,steps=12):
    """Time real TRAIN windows and full-sized optimizer steps; discard fitted weights.
    Count every job/window/batch. Never turn this pilot into scientific fitting.
    """
    seed_all(9071);timings=[]
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    for nodes in sorted({r['nodes'] for r in rows}):
        r=max((r for r in rows if r['nodes']==nodes),key=lambda r:r['train_rows'])
        a=np.load(root/'train'/(r['channel']+'.npy'));fit,val=splits(len(a),c)
        w=Windows(a,c,device);m=STGCN(nodes,c['hidden'],250,10,.3).to(device)
        opt=torch.optim.Adam(m.parameters(),lr=.001,eps=1e-7,foreach=False)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        times=[]
        for k in range(steps+3):
            starts=fit[(np.arange(64)+64*k)%len(fit)];sync();t=time.perf_counter()
            x,y=w.batch(starts);opt.zero_grad(set_to_none=True);loss=((m(x)-y)**2).mean();loss.backward()
            if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()):raise FloatingPointError('Benchmark nonfinite batch')
            opt.step();float(loss.detach());sync()
            if k>=3:times.append(time.perf_counter()-t)
        m.eval();eval_times=[]
        with torch.no_grad():
            for k in range(steps):
                sync();t=time.perf_counter();x,y=w.batch(fit[np.arange(64)%len(fit)]);z=m(x);float(((z-y)**2).mean());sync();eval_times.append(time.perf_counter()-t)
        with torch.no_grad():
            sync();t=time.perf_counter();xb,_=w.batch(fit[np.arange(c['prediction_batch_size'])%len(fit)]);pb=m(xb);sync();inference_time=time.perf_counter()-t
            if not torch.isfinite(pb).all():raise FloatingPointError('Nonfinite full prediction batch')
        with tempfile.TemporaryDirectory() as td:
            t=time.perf_counter();save_state(td+'/bench.pt',{'model':m.state_dict(),'optimizer':opt.state_dict()});cp=time.perf_counter()-t
        subset=[v for v in rows if v['nodes']==nodes]
        train_batches=sum((v['fit_windows']+63)//64 for v in subset)
        val_batches=sum((v['val_windows']+63)//64 for v in subset)
        infer_batches=sum((v['test_rows']-260+63)//64 for v in subset)
        full_epoch=train_batches*float(np.median(times))+val_batches*float(np.median(eval_times))+len(subset)*cp
        timings.append(dict(nodes=nodes,representative_channel=r['channel'],train_batch_seconds_median=float(np.median(times)),
            train_batch_seconds_max=max(times),eval_batch_seconds_median=float(np.median(eval_times)),checkpoint_seconds=cp,inference_batch256_seconds=inference_time,
            train_batches_per_seed_epoch=train_batches,validation_batches_per_seed_epoch=val_batches,
            inference_batches64_per_seed=infer_batches,seconds_per_seed_epoch=full_epoch,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,measured_train_steps=steps))
        del m,opt,w
        if device=='cuda':torch.cuda.empty_cache()
    sec=sum(r['seconds_per_seed_epoch'] for r in timings);s=len(c['seeds'])
    estimates={str(e):sec*e*s/3600 for e in (11,20,35)}
    inference_hours=sum(r['inference_batches64_per_seed']/4*r['inference_batch256_seconds'] for r in timings)*s/3600
    result=dict(environment=environment(device),timings=timings,training_jobs=len(rows)*s,
        training_only_hours_by_epochs=estimates,estimated_inference_hours=inference_hours,
        conservative_training_hours_35_epochs=1.5*estimates['35'],
        estimated_training_under_six_hours=1.5*estimates['35']<6,
        note='Measured on this machine, extrapolated by exact batch counts; not an observed full-run runtime. Excludes setup, detector processing, export and Drive I/O. Six hours is not guaranteed. CPU timings are NOT A100 timings.')
    save_json(out/'RUNTIME_ESTIMATE.json',result);print(result,flush=True)
    return result
