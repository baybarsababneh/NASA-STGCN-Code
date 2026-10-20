"""Read-only input, alignment and released-weight verification. No fitting/scoring."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import zipfile, json, io, hashlib, csv, sys
import numpy as np
import pandas as pd
import h5py
from threadpoolctl import threadpool_limits
threadpool_limits(1)
ROOT=Path(__file__).resolve().parent
NASA=ROOT.parent/'project_sources/03-archive.zip'
SCORES=ROOT/'materialized/NASA Research Project/EXP012_LABEL_BLIND_SCORE_EXPORT.zip'
SOURCE=ROOT/'exp012_source'
OUT=ROOT/'outputs'
def sha(b):return hashlib.sha256(b).hexdigest()
def array(z,p):return np.load(io.BytesIO(z.read(p)),allow_pickle=False)
def save(k,v):(OUT/k).write_text(json.dumps(v,indent=2,allow_nan=False))
def replay(f,x):
    cfg=json.loads(f.attrs['model_config'])['config']
    a=x.astype(np.float32)
    for layer in cfg:
        c=layer['config'];typ=layer['class_name'];name=c['name']
        if typ in ('Dropout','Activation'):continue
        g=f['model_weights'][name][name]
        if typ=='LSTM':
            assert c['activation']=='tanh' and c['recurrent_activation']=='hard_sigmoid'
            w=g['kernel:0'][:];u=g['recurrent_kernel:0'][:];b=g['bias:0'][:]
            h=np.zeros((len(x),80),np.float32);cell=h.copy();seq=[]
            projected=a@w+b
            for t in range(a.shape[1]):
                v=projected[:,t]+h@u
                i=np.clip(v[:,:80]*.2+.5,0,1);fg=np.clip(v[:,80:160]*.2+.5,0,1)
                cc=np.tanh(v[:,160:240]);o=np.clip(v[:,240:]*.2+.5,0,1)
                cell=fg*cell+i*cc;h=o*np.tanh(cell);seq.append(h.copy())
            a=np.stack(seq,axis=1) if c['return_sequences'] else h
        elif typ=='Dense':a=a@g['kernel:0'][:]+g['bias:0'][:]
        else:raise ValueError(typ)
    return a
def main():
    OUT.mkdir(exist_ok=True);inv=[];replays=[];errors=[]
    z=zipfile.ZipFile(NASA);s=zipfile.ZipFile(SCORES)
    manifest=json.loads((SOURCE/'data_manifest.json').read_text())
    testfreeze=json.loads(s.read('TEST_INPUT_FREEZE.json'))
    assert sha(NASA.read_bytes())==testfreeze['archive_sha256']
    commit_count=0;file_count=0
    for n in s.namelist():
        if n.endswith('/COMMIT.json'):
            co=json.loads(s.read(n));base=n.rsplit('/',1)[0]+'/'
            for name,digest in co['files'].items():assert sha(s.read(base+name))==digest;file_count+=1
            commit_count+=1
    for row in manifest:
        c=row['channel'];tr=z.read('data/data/train/'+c+'.npy');te=z.read('data/data/test/'+c+'.npy')
        assert sha(tr)==row['original_TRAIN_sha256']
        assert sha(te)==testfreeze['selected_array_sha256'][c]
    for n in z.namelist():
        if not n.endswith('.h5'):continue
        c=Path(n).stem;tr=array(z,'data/data/train/'+c+'.npy');te=array(z,'data/data/test/'+c+'.npy')
        pred=array(z,'data/data/2018-05-19_15.00.10/y_hat/'+c+'.npy')
        sm=array(z,'data/data/2018-05-19_15.00.10/smoothed_errors/'+c+'.npy')
        assert len(pred)==len(te)-260 and sm.shape==pred.shape
        assert np.isfinite(pred).all() and np.isfinite(te).all() and np.isfinite(tr).all()
        ix=np.arange(250,len(te)-10);e=np.abs(te[ix,0]-pred)
        calc=pd.Series(e).ewm(span=105,adjust=True).mean().to_numpy()
        if c!='C-2':calc[:250]=calc[:500].mean()
        delta=float(np.max(np.abs(sm-calc)))
        error_pass=bool(np.allclose(sm,calc,atol=1e-12,rtol=1e-12))
        if c in testfreeze['selected_array_sha256']:assert error_pass,(c,delta)
        errors.append({'channel':c,'max_absolute_error':delta,'cached_smoothing_replay_pass':error_pass,
                       'in_frozen_evaluation_roster':c in testfreeze['selected_array_sha256']})
        with h5py.File(io.BytesIO(z.read(n)),'r') as f:
            config=json.loads(f.attrs['model_config']);layers=config['config']
            assert [v['config']['units'] for v in layers if v['class_name']=='LSTM']==[80,80]
            assert next(v['config']['units'] for v in layers if v['class_name']=='Dense')==10
            # Fixed, label-independent first/middle/last available forecast positions.
            pos=np.unique(np.array([0,len(pred)//2,len(pred)-1]));
            x=np.stack([te[j:j+250] for j in pos])
            got=replay(f,x)[:,0];expected=pred[pos]
            difference=float(np.max(np.abs(got-expected)))
            passed=bool(np.allclose(got,expected,atol=1e-5,rtol=1e-4))
            if c in testfreeze['selected_array_sha256']:assert passed,(c,difference,got,expected)
            replays.append({'channel':c,'positions':pos.tolist(),'target_indices':(pos+250).tolist(),
                'maximum_absolute_difference':difference,'pass_atol1e-5_rtol1e-4':passed,
                'prediction_sha256':sha(z.read('data/data/2018-05-19_15.00.10/y_hat/'+c+'.npy')),
                'model_sha256':sha(z.read(n))})
        inv.append({'channel':c,'train_rows':len(tr),'test_rows':len(te),'width':te.shape[1],
                    'prediction_rows':len(pred),'target_start':250,'target_end_exclusive':len(te)-10})
    (ROOT/'sources/released_params.log').write_bytes(z.read('data/data/2018-05-19_15.00.10/params.log'))
    save('array_and_prediction_inventory.json',inv);save('sampled_model_replays.json',replays);save('cached_error_replays.json',errors)
    save('SOURCE_VERIFICATION.json',{'NASA_archive_sha256':sha(NASA.read_bytes()),
        'score_export_sha256':sha(SCORES.read_bytes()),'score_commits_verified':commit_count,
        'score_files_hash_verified':file_count,'TRAIN_TEST_array_bindings_verified':len(manifest)*2,
        'released_models':len(inv),'model_replays':sum(len(x['positions']) for x in replays),
        'maximum_sampled_prediction_difference':max(x['maximum_absolute_difference'] for x in replays),
        'maximum_evaluation_roster_cached_smoothed_error_difference':max(x['max_absolute_error'] for x in errors if x['in_frozen_evaluation_roster']),
        'excluded_array_error_failures':[x for x in errors if not x['cached_smoothing_replay_pass']],
        'alignment':'prediction[k] = forecast for raw TEST[k+250,0]; last ten raw rows omitted',
        'release_horizon':10,'saved_scalar_is':'first output, not ten-output mean',
        'no_training':True,'no_NASA_metric_calculation':True})
    print(json.dumps(json.loads((OUT/'SOURCE_VERIFICATION.json').read_text()),indent=2))
if __name__=='__main__':main()
