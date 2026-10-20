"""Read-only post-run integrity, inventory, ranking and gate verification."""
import argparse,json,zipfile
from pathlib import Path
import pandas as pd
from run_exp009 import ROOT,sha,dumps,candidates,BASELINES,choose,table_summary

def verify(out):
    out=Path(out);manifest=json.loads((out/'OUTPUT_MANIFEST.json').read_text())
    for f,h in manifest.items():assert sha((out/f).read_bytes())==h,f
    identity=json.loads((out/'RUN_FREEZE.json').read_text());cfg=identity['config'];runhash=sha(dumps(identity))
    selection=json.loads((out/'DETECTOR_CANDIDATE_FREEZE.json').read_text());assert selection['runhash']==runhash
    sn=pd.read_csv(out/'selection_nominal.csv').to_dict('records');sp=pd.read_csv(out/'selection_stress.csv').to_dict('records')
    recomputed=choose(table_summary(sn,sp,cfg),candidates(cfg));assert recomputed==selection['selected']
    with zipfile.ZipFile(ROOT/'data/EXP008_results.zip') as z:plan=json.loads(z.read('PLAN.json'))
    expected={}
    def add(p,model,c,stage):
        key=dict(job_id=p['job_id'],model=model,candidate=c,stage=stage,runhash=runhash);uid=sha(dumps(key))[:24];assert uid not in expected;expected[uid]=key
    for p in plan:
        if p['job']['fold']=='canonical' and p['job']['role']=='fixed':
            for c in candidates(cfg):add(p,'gcn',c,'selection')
        stage='canonical_audit' if p['job']['fold']=='canonical' else 'earlier_origin_audit';mission=p['job']['mission']
        models=['gcn' if p['job']['role']=='fixed' else 'message_removed']
        if p['job']['role']=='fixed' and p['job']['seed']==17:models.append('persistence')
        for model in models:
            for c in BASELINES+([recomputed[mission]]if recomputed[mission] else []):add(p,model,c,stage)
    actual={p.name.removesuffix('.done.json')for p in (out/'units').glob('*.done.json')};assert actual==set(expected)
    all_files={p.name for p in (out/'units').iterdir()};assert all_files=={s+ext for s in expected for ext in ('.json','.done.json')}
    for uid,key in expected.items():
        p=out/'units'/f'{uid}.json';assert json.loads(p.with_suffix('.done.json').read_text())==dict(key=key,sha256=sha(p.read_bytes()))
    close=json.loads((out/'STUDY_CLOSE.json').read_text());assert close['units']==len(expected)
    an=pd.read_csv(out/'audit_nominal.csv').to_dict('records');ap=pd.read_csv(out/'audit_stress.csv').to_dict('records')
    gate_rows=[]
    for mission,sel in recomputed.items():
        if sel is None:continue
        for stage,fold in [('canonical_audit',None),('earlier_origin_audit',None),('earlier_origin_audit','R1'),('earlier_origin_audit','R2')]:
            nn=[r for r in an if r['mission']==mission and r['stage']==stage and r['model']=='gcn' and r['candidate']==sel['id'] and (fold is None or r['fold']==fold)]
            pp=[r for r in ap if r['mission']==mission and r['stage']==stage and r['model']=='gcn' and r['candidate']==sel['id'] and r['family'] in ('spike','step8','step32','ramp32') and (fold is None or r['fold']==fold)]
            gate_rows.extend(table_summary(nn,pp,cfg))
    gates=pd.DataFrame(gate_rows)
    for mission,sel in recomputed.items():
        if sel is None:assert close['mission_decisions'][mission]=='NO_FEASIBLE_CANDIDATE'
        else:
            g=gates[gates.mission==mission];assert len(g)==4
            passed=bool(g.nominal_ok.all()and g.sensitivity_ok.all());assert close['mission_decisions'][mission]==('PROVISIONAL_FREEZE_REQUIRES_REVIEW' if passed else 'AUDIT_FAILED_NO_FALLBACK')
    print(json.dumps(dict(status='PASS',unique_units=len(expected),ranking_recomputed=True,commit_hashes_verified=True,scope='integrity and decision logic; scientific review still required'),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
