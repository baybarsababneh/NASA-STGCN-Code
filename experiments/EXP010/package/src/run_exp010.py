"""GPU-native inference-only EXP010. Frozen checkpoints, finite perturbation bank, no selector."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse,hashlib,io,json,sys,time,zipfile,shutil
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from common import sha,digest,atomic_bytes,freeze_json,write_json
from execution import configure_cuda,Lease
from json_e01 import dumps_e01
from trace_io import commit,read
from mechanisms import cases,perturb,response,nominal_context
import run_exp008 as parent
ROOT=Path(__file__).resolve().parents[1]

def model_for(meta):
    with zipfile.ZipFile(ROOT/meta['archive'])as z:ck=torch.load(io.BytesIO(z.read('checkpoint.pt')),map_location='cpu',weights_only=True)
    r=meta['reference_result'];j=meta['job'];assert ck['result']==r and ck['job']==j and ck['run_hash']==meta['parent_run_hash']
    spec=ck['spec'];model=parent.MessageGatedGCN(r['graph_num_nodes'],r['hidden_size'],ck['state_dict']['adjacency'].numpy(),spec['kernel_size'],spec['dilations'],j['gate_mode'],j['remove_messages'])
    model.load_state_dict(ck['state_dict'],strict=True);model.to('cuda');model.eval();return model,ck

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert out!=ROOT and ROOT not in out.parents
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for f,h in manifest.items():assert sha(ROOT/f)==h,f
    cfg=json.loads((ROOT/'config.json').read_text());plan=json.loads((ROOT/'plan.json').read_text());assert len(plan)==114
    assert {p['job']['channel']for p in plan}==set(cfg['mechanical_channels']+cfg['posthoc_mechanism_channels'])
    parent.guard();env=configure_cuda();original=json.loads((ROOT/'provenance/RUN_FREEZE.json').read_text())
    if env!=original['environment']:
        write_json(out/'ENVIRONMENT_MISMATCH.json',dict(expected=original['environment'],actual=env));raise RuntimeError('Frozen GPU environment mismatch; no automatic fallback')
    with Lease(out)as lease:
        freeze=dict(experiment=cfg['experiment'],environment=env,config=cfg,manifest_sha256=sha(ROOT/'MANIFEST.json'));freeze_json(out/'RUN_FREEZE.json',freeze);runhash=digest(freeze)
        for f in list(manifest)+['MANIFEST.json']:
            if f.startswith(('checkpoints/','train_prefix/')):continue
            p=out/'reproducibility'/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,p)
        data={r['channel']:r for r in json.loads((ROOT/'data_prefix_manifest.json').read_text())};graphs={(g['channel'],g['fit_fraction']):g for g in json.loads((ROOT/'provenance/graph_manifest.json').read_text())}
        expected=set();records=[];policy_rows=[];context_rows=[];completed=0
        for meta in plan:
            j=meta['job'];r=meta['reference_result'];raw=parent.load_arrays(meta,data[j['channel']]);probe_list=cases(r['num_values'])
            cohort='mechanical'if j['channel']in cfg['mechanical_channels']else'posthoc_mechanism'
            identity=dict(runhash=runhash,parent_job=meta['job_id'],case='nominal');uid=digest(identity)[:24];expected.add(uid)
            saved=read(out,uid,identity)
            if saved is None:
                rec,base=parent.replay(meta,data[j['channel']],graphs)
                commit(out,uid,identity,dict(kind='nominal',parent_replay=rec),base,lease)
            else:rec,base=saved
            context_rows.extend(dict(**v,cohort=cohort)for v in nominal_context(base,raw,meta))
            completed+=1;model=None;ck=None
            for case in probe_list:
                identity=dict(runhash=runhash,parent_job=meta['job_id'],case=case['id']);uid=digest(identity)[:24];expected.add(uid);saved=read(out,uid,identity)
                if saved is None:
                    if model is None:model,ck=model_for(meta)
                    changed_raw=perturb(raw,case)
                    changed,mse=parent.predict(model,changed_raw,r['graph_features'],r['fit_end'],r['stop_end'])
                    common=dict(mission=j['mission'],channel=j['channel'],seed=j['seed'],role=j['role'],cohort=cohort,case_id=case['id'],family=case['family'],amplitude=case['amplitude'],sign=case['sign'],onset=case['onset'],end=case['end'],fit_scale_used_fallback=bool(np.std(raw[:r['fit_end'],0])<1e-8))
                    models=['gcn'if j['role']=='fixed'else'message_removed']
                    if j['role']=='fixed'and j['seed']==17:models.append('persistence')
                    rows=[];pr=[]
                    for name in models:
                        response_row,diag=response(base,changed,case,r['num_values'],name);rows.append(dict(**common,**response_row));pr.extend(dict(**common,**v)for v in diag)
                    record=dict(kind='perturbed',case=case,responses=rows,policy_diagnostics=pr,forecast_mse_diagnostic_only=mse,parent_archive_sha256=meta['archive_sha256'])
                    assert all(torch.equal(v.cpu(),ck['state_dict'][k])for k,v in model.state_dict().items()),'Frozen weights changed'
                    commit(out,uid,identity,record,changed,lease)
                else:record,changed=saved
                records.extend(record['responses']);policy_rows.extend(record['policy_diagnostics']);completed+=1
                if completed%50==0:print(f'{completed}/3078 units committed/verified',flush=True)
                write_json(out/'STATUS.json',dict(status='RUNNING',completed=completed,total=3078))
            if model is not None:del model,ck;torch.cuda.empty_cache()
        actual={p.name.removesuffix('.done.json')for p in (out/'units').glob('*.done.json')};assert actual==expected and len(actual)==3078
        pd.DataFrame(context_rows).to_csv(out/'nominal_context.csv',index=False)
        pd.DataFrame(records).to_csv(out/'response_streams.csv',index=False);pd.DataFrame(policy_rows).to_csv(out/'policy_diagnostics.csv',index=False)
        # No candidate ranking or model/detector promotion is implemented.
        df=pd.DataFrame(records);keys=['mission','cohort','model','family','amplitude','sign']
        channels=df.groupby(keys+['channel'],dropna=False).agg(probe_rows=('status','size'),null_cases=('status',lambda v:int((v=='NULL_PERTURBATION').sum())),mean_response_ratio=('response_ratio','mean')).reset_index()
        channels.to_csv(out/'channel_response_summary.csv',index=False)
        channels.groupby(keys,dropna=False).agg(channels_total=('channel','size'),channels_nonnull=('mean_response_ratio','count'),mean_channel_response_ratio=('mean_response_ratio','mean'),median_channel_response_ratio=('mean_response_ratio','median'),probe_rows=('probe_rows','sum'),null_cases=('null_cases','sum')).to_csv(out/'descriptive_response_summary.csv')
        freeze_json(out/'STUDY_CLOSE.json',dict(status='COMPLETE_DIAGNOSTIC_NO_SELECTION',units=3078,nominal_checkpoint_replays=114,perturbed_model_streams=2964,unique_persistence_probe_records=494,new_fits=0,weights_changed=False,detector_selected=False,NASA_TEST_accessed=False,anomaly_labels_accessed=False,final_calibration_accessed=False))
        write_json(out/'STATUS.json',dict(status='COMPLETE_DIAGNOSTIC_NO_SELECTION',completed=3078,total=3078))
        write_json(out/'OUTPUT_MANIFEST.json',{str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name not in ('OUTPUT_MANIFEST.json','ACTIVE_RUN.json')and 'logs'not in p.relative_to(out).parts})
        print((out/'STUDY_CLOSE.json').read_text(),flush=True)
if __name__=='__main__':main()
