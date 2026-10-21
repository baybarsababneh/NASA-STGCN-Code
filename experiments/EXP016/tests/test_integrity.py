import json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from exp016.io import ROOT,commit,committed,save_json,freeze,export_run,sha
from exp016.metrics import labels

def test_committed_payload_tampering_rejected(tmp_path):
    save_json(tmp_path/'x.json',{'x':1});commit(tmp_path,['x.json'],'id');assert committed(tmp_path,'id')
    save_json(tmp_path/'x.json',{'x':2})
    with pytest.raises(RuntimeError):committed(tmp_path,'id')

def test_immutable_freeze_rejects_change(tmp_path):
    p=tmp_path/'freeze.json';freeze(p,{'v':1});freeze(p,{'v':1})
    with pytest.raises(RuntimeError):freeze(p,{'v':2})

def test_archive_manifest_recomputes(tmp_path):
    import zipfile,hashlib
    out=tmp_path/'run';save_json(out/'STATUS.json',{'status':'PAUSED_RESUMABLE'})
    zpath=export_run(out)
    with zipfile.ZipFile(zpath) as z:
        m=json.loads(z.read('run/RESULT_MANIFEST.json'))
        for p,h in m.items():assert hashlib.sha256(z.read('run/'+p)).hexdigest()==h

def test_label_duplicate_policy_on_synthetic_annotations(tmp_path):
    import csv
    p=tmp_path/'synthetic_labels.csv'
    with p.open('w') as f:
        w=csv.writer(f);w.writerow(['chan_id','spacecraft','anomaly_sequences','class','num_values'])
        for i in range(80):w.writerow([f'fixture-{i}','SMAP','[[1,2]]','[point]',10])
        w.writerow(['P-2','SMAP','[[3,4]]','[point]',10]);w.writerow(['P-2','SMAP','[[4,5]]','[point]',10])
    primary,alternative=labels(p)
    assert len(primary)==81 and 'T-10' not in primary
    assert primary['P-2']['events']==[[3,4]] and alternative['P-2']==[[4,5]]


def test_scientific_cpu_rejected(tmp_path):
    r=subprocess.run([sys.executable,'-m','exp016.run','run','--cpu','--out',str(tmp_path)],cwd=ROOT,text=True,capture_output=True)
    assert r.returncode!=0 and 'CPU is engineering-only' in r.stderr

def test_bootstrap_refuses_reduced_native_roster():
    from exp016.evaluate import bootstrap
    from exp016.metrics import counts
    from exp016.io import read_json
    c=read_json(ROOT/'config.json');c.update(seeds=[17],bootstrap_repetitions=10)
    count,_=counts([[251,252]],np.array([False,True,True,False]),np.arange(250,254))
    rows=[]
    for ch in ('fixture-a','fixture-b'):
        for mode in ('uniform','native_C2'):
            for role,seed in [('released_lstm',0),('stgcn',17)]:
                if ch=='fixture-b' and role=='released_lstm' and mode=='native_C2':continue
                rows.append(dict(channel=ch,mission='SMAP',detector=mode,role=role,seed=seed,**count))
    results=bootstrap(rows,c)
    assert results and all(r['detector']=='uniform' for r in results)
