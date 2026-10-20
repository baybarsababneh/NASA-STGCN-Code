"""Select and verify only the immutable inputs needed for this comparison."""
import io, json, zipfile, hashlib
from pathlib import Path
import numpy as np
from integrity import ROOT, read, sha, checked, atomic, safe_rel, arrays, write

def verify_package():
    from integrity import manifest
    manifest(ROOT,'MANIFEST.json')

def verify_static(root):
    root=Path(root);inventory=read(root/'INPUT_INVENTORY.json')
    for n,h in inventory['present'].items(): checked(root/safe_rel(n),h)
    subset=read(root/'inputs/exp012_scores/SUBSET.json')
    seal=read(root/'inputs/exp012_scores/SCORE_EXPORT_FREEZE.json')
    checked(root/'inputs/exp012_scores/SCORE_EXPORT_FREEZE.json',subset['original_seal_sha256'])
    checked(root/'inputs/exp012_scores/EVALUATION_RUN_FREEZE.json',seal['run_freeze_sha256'])
    checked(root/'inputs/exp012_scores/TEST_INPUT_FREEZE.json',seal['TEST_input_freeze_sha256'])
    rows=read(root/'inputs/TEST_ROSTER.json')
    expected={f'{arm}__{ch}__{s}' for ch in rows for arm in ['gcn','lstm'] for s in [17,29,43]}
    if set(subset['roster'])!=expected or len(subset['roster'])!=len(expected): raise RuntimeError('EXP012 subset roster differs')
    for n,h in subset['files'].items():
        if seal['files'].get(n)!=h: raise RuntimeError('EXP012 subset lost its original score-seal binding')
        checked(root/'inputs/exp012_scores'/safe_rel(n),h)
    for j in subset['roster']:
        base=root/'inputs/exp012_scores/units'/j
        unit=read(base/'COMMIT.json');record=read(base/'record.json')
        if record['job_id']!=j or unit['identity']['job_id']!=j: raise RuntimeError('EXP012 unit identity mismatch')
        for n,h in unit['files'].items(): checked(base/safe_rel(n),h)
        ch=record['channel']; row=rows[ch]
        if record['TEST_array_sha256']!=row['sha256'] or record['original_TEST_rows']!=row['rows'] or record['mission']!=row['mission'] or record['target_start']!=250: raise RuntimeError('EXP012 support/source differs')
    from integrity import manifest
    for n,m in [('exp012_reports','LABEL_EVALUATION_COMPLETE.json'),('exp013_reports','COMPLETE.json')]: manifest(root/'inputs'/n,m)
    cfg=read(root/'config.json');detector=read(root/'inputs/EXP012_CONFIG.json')['detector']
    if cfg['detector']!=detector or cfg['target_start']!=250 or cfg['gdn_lookback']!=5 or cfg['gdn_batch_size']!=32 or cfg['new_training']: raise RuntimeError('Frozen scientific configuration differs')
    contract=read(root/'INPUT_CONTRACT.json')
    checked(ROOT/'vendor/frozen/hundman_detector.py',contract['frozen_detector_sha256'])
    checked(ROOT/'vendor/frozen/metrics.py',contract['frozen_evaluator_sha256'])
    return inventory,rows

class Source:
    def __init__(self,path):
        self.path=Path(path);self.zip=None
        if not self.path.exists(): raise FileNotFoundError('Required EXP013 source is missing: '+str(self.path))
        if self.path.is_file():
            self.zip=zipfile.ZipFile(self.path)
            names=self.zip.namelist()
            if len(names)!=len(set(names)): raise RuntimeError('Duplicate source archive entries')
            for n in names: safe_rel(n)
            self.names=set(names)
        else: self.names=None
    def close(self):
        if self.zip: self.zip.close()
    def payload(self,n):
        n=safe_rel(n).as_posix()
        if self.zip:
            if n not in self.names: raise FileNotFoundError('Missing EXP013 constituent: '+n)
            info=self.zip.getinfo(n)
            if info.file_size>64*2**20 or (info.external_attr>>16)&0o170000==0o120000: raise RuntimeError('Oversized or unsafe EXP013 constituent: '+n)
            return self.zip.read(n)
        p=self.path/n
        if p.is_symlink() or self.path.resolve() not in p.resolve().parents: raise RuntimeError('Unsafe constituent path')
        if p.stat().st_size>64*2**20: raise RuntimeError('Oversized constituent')
        return p.read_bytes()

def recover(root,source,dst):
    root=Path(root);dst=Path(dst);contract=read(root/'INPUT_CONTRACT.json')
    copied={}; origin=Source(source)
    try:
        def take(n,h):
            p=dst/safe_rel(n)
            # Re-read the original input on every invocation. A changed original
            # is rejected even if a previously verified local cache exists.
            b=origin.payload(n)
            if hashlib.sha256(b).hexdigest()!=h: raise RuntimeError('Frozen EXP013 source differs: '+n)
            if p.exists(): checked(p,h)
            else: atomic(p,b)
            copied[n]=h
        for n,h in contract['known_required_files'].items(): take(n,h)
        seal=read(dst/'scores/SCORE_SEAL.json')
        expected_study=sha(root/'inputs/freeze/STUDY_FREEZE.json')
        if seal['study_freeze_sha256']!=expected_study or seal['NASA_labels_opened']: raise RuntimeError('EXP013 score seal has wrong study identity')
        rows=read(root/'inputs/TEST_ROSTER.json')
        for ch,r in rows.items():
            j=f"{r['mission']}__gdn__17__{ch}"
            for leaf in ['COMMIT.json','record.json','scores.npz']:
                n=f'units/{j}/{leaf}'
                if n not in seal['files']: raise RuntimeError('Frozen GDN reference absent from original score seal: '+n)
                take('scores/'+n,seal['files'][n])
            unit=read(dst/'scores/units'/j/'COMMIT.json');record=read(dst/'scores/units'/j/'record.json')
            if record['job_id']!=j or record['identity']!=unit['identity'] or record['status']!='FINITE_SCORES': raise RuntimeError('GDN reference unit is not complete')
            th=read(dst/'calibration'/f"{r['mission']}__gdn__17/thresholds.json")
            t=th['channels'][ch]
            if record['identity']['TEST_array_sha256']!=r['sha256'] or record['identity']['study']!=expected_study or record['identity']['model_state_sha256']!=th['model_state_sha256'] or record['threshold']!=t['threshold'] or record['detector_span']!=t['span']: raise RuntimeError('GDN reference provenance differs')
            for n,h in unit['files'].items(): checked(dst/'scores/units'/j/safe_rel(n),h)
        return {'verified_constituents':copied,'source_scope':'two selected GDN models and their 81 sealed native reference streams; other EXP013 bytes are not independently replayed','score_seal_sha256':sha(dst/'scores/SCORE_SEAL.json'),'study_freeze_sha256':expected_study}
    finally: origin.close()

def load_test(root,ch):
    root=Path(root);r=read(root/'inputs/TEST_ROSTER.json')[ch]
    with zipfile.ZipFile(root/'inputs/NASA_arrays_only.zip') as z:
        n='data/data/test/'+ch+'.npy'
        b=z.read(n)
    if hashlib.sha256(b).hexdigest()!=r['sha256']: raise RuntimeError('NASA TEST array hash differs: '+ch)
    a=np.load(io.BytesIO(b),allow_pickle=False)
    if a.shape!=(r['rows'],r['columns']) or a.dtype.kind!='f' or not np.isfinite(a).all(): raise RuntimeError('Invalid NASA TEST shape/type/values: '+ch)
    return a

def required_paths(root):
    c=read(Path(root)/'INPUT_CONTRACT.json');r=read(Path(root)/'inputs/TEST_ROSTER.json')
    return {**c['known_required_files'],**{f"scores/units/{v['mission']}__gdn__17__{ch}/{n}":'hash supplied by the pinned SCORE_SEAL.json' for ch,v in r.items() for n in ['COMMIT.json','record.json','scores.npz']}}
