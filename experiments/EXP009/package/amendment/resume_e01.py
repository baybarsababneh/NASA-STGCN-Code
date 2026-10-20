"""Versioned EXP009 E01 launcher. Never edits parent source, scientific freeze, or units."""
import argparse,datetime,fcntl,hashlib,importlib,json,os,platform,shutil,sys
from pathlib import Path
from json_e01 import dumps_e01
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def write_new_or_same(p,value):
    b=dumps_e01(value)
    if p.exists():assert p.read_bytes()==b,f'Existing evidence changed: {p}'
    else:
        p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(b);os.replace(tmp,p)

def inventory(out,runhash,planned):
    records={}
    for mark in sorted((out/'units').glob('*.done.json')):
        uid=mark.name.removesuffix('.done.json');p=mark.with_name(uid+'.json');m=json.loads(mark.read_text());key=m['key']
        assert key['runhash']==runhash and digest(dumps_e01(key))[:24]==uid,'Wrong unit identity'
        assert uid in planned and key==planned[uid],'Unplanned detector unit'
        assert p.is_file() and m==dict(key=key,sha256=sha(p)),'Committed unit hash mismatch'
        json.loads(p.read_text())
        records[uid]=dict(result_sha256=sha(p),marker_sha256=sha(mark))
    return records

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',required=True);ap.add_argument('--output',required=True);ap.add_argument('--evidence',required=True);ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
    parent=Path(args.parent).resolve();out=Path(args.output).resolve();ev=Path(args.evidence).resolve()
    assert out.is_dir() and (out/'RUN_FREEZE.json').is_file(),'Resume requires the existing EXP009 results, not a new empty run'
    assert ev!=out and out not in ev.parents,'Engineering evidence must be a sibling of results'
    em=json.loads((HERE/'E01_MANIFEST.json').read_text())
    for f,h in em['files'].items():assert sha(HERE/f)==h,f'E01 source mismatch: {f}'
    manifest=json.loads((parent/'MANIFEST.json').read_text())
    assert sha(parent/'MANIFEST.json')==em['parent_manifest_file_sha256']
    for f,h in manifest.items():assert sha(parent/f)==h,f'Original source/input changed: {f}'
    cfg=json.loads((parent/'config.json').read_text());old=json.loads((out/'RUN_FREEZE.json').read_text())
    sys.path.insert(0,str(parent/'src'));r=importlib.import_module('run_exp009')
    assert r.ROOT.resolve()==parent
    env=dict(python=platform.python_version(),numpy=r.np.__version__,pandas=r.pd.__version__,platform=platform.platform())
    expected=dict(config=cfg,manifest_sha256=digest(r.dumps(manifest)),environment=env)
    assert old==expected,'Original scientific/source/environment identity differs; do not bypass this check'
    assert r.dumps(old)==dumps_e01(old),'Serialization changed parent identity bytes'
    runhash=digest(r.dumps(old))
    # Enumerate original unit identities without calculating detector scores.
    import zipfile
    with zipfile.ZipFile(parent/'data/EXP008_results.zip') as z:plan=json.loads(z.read('PLAN.json'))
    planned={}
    def add(p,model,c,stage):
        key=dict(job_id=p['job_id'],model=model,candidate=c,stage=stage,runhash=runhash);uid=digest(r.dumps(key))[:24];assert uid not in planned;planned[uid]=key
    for p in plan:
        if p['job']['fold']=='canonical' and p['job']['role']=='fixed':
            for c in r.candidates(cfg):add(p,'gcn',c,'selection')
    selection=out/'DETECTOR_CANDIDATE_FREEZE.json'
    if selection.exists():
        selected=json.loads(selection.read_text());assert selected['runhash']==runhash
        for p in plan:
            stage='canonical_audit' if p['job']['fold']=='canonical' else 'earlier_origin_audit';mission=p['job']['mission']
            models=['gcn'if p['job']['role']=='fixed'else'message_removed']
            if p['job']['role']=='fixed'and p['job']['seed']==17:models.append('persistence')
            for model in models:
                for c in r.BASELINES+([selected['selected'][mission]]if selected['selected'][mission]else[]):add(p,model,c,stage)
    with open(out/'RUN.lock','a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=inventory(out,runhash,planned)
        assert current,'No existing committed units found; this is a resume amendment'
        ev.mkdir(parents=True,exist_ok=True)
        snap=ev/'PRE_E01_COMMIT_INVENTORY.json'
        if not snap.exists():write_new_or_same(snap,current)
        initial=json.loads(snap.read_text())
        for uid,record in initial.items():assert current.get(uid)==record,'Pre-E01 committed bytes changed'
        snapshot_dir=ev/'parent_pre_e01'
        if not snapshot_dir.exists():
            snapshot_dir.mkdir()
            for name in ('RUN_FREEZE.json','requirements-resolved.txt'):
                if (out/name).exists():shutil.copy2(out/name,snapshot_dir/name)
            if (out/'reproducibility').exists():shutil.copytree(out/'reproducibility',snapshot_dir/'reproducibility')
        # Original snapshots and amended execution source remain separately attributable.
        source=ev/'amendment_source';source.mkdir(exist_ok=True)
        for f in list(em['files'])+['E01_MANIFEST.json']:
            dest=source/f;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(dest)==sha(HERE/f)
            else:shutil.copy2(HERE/f,dest)
        link=dict(amendment='EXP009_DETECTOR_V1_E01',scientific_experiment='EXP009_DETECTOR_V1',parent_runhash=runhash,parent_run_freeze_sha256=sha(out/'RUN_FREEZE.json'),parent_manifest_sha256=sha(parent/'MANIFEST.json'),amendment_manifest_sha256=sha(HERE/'E01_MANIFEST.json'),pre_amendment_committed_units=len(initial),change='JSON fallback converts supported NumPy scalar values to native Python scalar values; all original scientific functions and commit identities retained')
        write_new_or_same(ev/'ENGINEERING_AMENDMENT.json',link)
        print(f'E01 preflight: {len(current)} current committed units verified; {len(initial)} pre-amendment units protected.',flush=True)
    if args.preflight_only:return
    r.dumps=dumps_e01  # Sole change to parent runtime behaviour; parent files remain byte-identical.
    try:
        r.run(argparse.Namespace(input=parent/'data/EXP008_results.zip',output=out))
        verifier=importlib.import_module('verify_results');verifier.verify(out)
    finally:
        for uid,record in initial.items():
            assert sha(out/'units'/f'{uid}.json')==record['result_sha256']
            assert sha(out/'units'/f'{uid}.done.json')==record['marker_sha256']
        assert sha(out/'RUN_FREEZE.json')==link['parent_run_freeze_sha256']
    write_new_or_same(ev/'POST_E01_PRESERVATION.json',dict(pre_amendment_units=len(initial),all_pre_amendment_results_and_markers_byte_identical=True,parent_freeze_byte_identical=True,runhash_unchanged=True,status='COMPLETE'))
    hashes={str(p.relative_to(ev)):sha(p)for p in ev.rglob('*')if p.is_file()and p.name!='EVIDENCE_MANIFEST.json'}
    (ev/'EVIDENCE_MANIFEST.json').write_bytes(dumps_e01(hashes))
    print('E01 complete: original committed units and scientific freeze preserved.',flush=True)

if __name__=='__main__':main()
