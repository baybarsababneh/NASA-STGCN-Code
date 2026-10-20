"""Select and verify only the immutable inputs needed for this comparison."""
import io, json, zipfile, hashlib
from pathlib import Path
import numpy as np
from integrity import ROOT, read, sha, checked, atomic, safe_rel, arrays, write

def verify_package():
    from integrity import manifest
    manifest(ROOT,'MANIFEST.json')

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

def load_test(root,ch):
    root=Path(root);r=read(root/'inputs/TEST_ROSTER.json')[ch]
    with zipfile.ZipFile(root/'inputs/NASA_arrays_only.zip') as z:
        n='data/data/test/'+ch+'.npy'
        b=z.read(n)
    if hashlib.sha256(b).hexdigest()!=r['sha256']: raise RuntimeError('NASA TEST array hash differs: '+ch)
    a=np.load(io.BytesIO(b),allow_pickle=False)
    if a.shape!=(r['rows'],r['columns']) or a.dtype.kind!='f' or not np.isfinite(a).all(): raise RuntimeError('Invalid NASA TEST shape/type/values: '+ch)
    return a

def verify_static(root):
    from integrity import manifest, digest
    root=Path(root);inventory=read(root/'INPUT_INVENTORY.json')
    for n,h in inventory['present'].items():checked(root/safe_rel(n),h)
    cfg=read(root/'config.json');rows=read(root/'inputs/TEST_ROSTER.json');contract=read(root/'INPUT_CONTRACT.json')
    if cfg['detector']!=read(root/'inputs/EXP012_CONFIG.json')['detector'] or cfg['new_training'] or cfg['new_calibration'] or cfg['discriminator_hybrid']:raise RuntimeError('Frozen detector/scope differs')
    if cfg['lookback']!=64 or cfg['native_batch_size']!=256 or cfg['target_start']!=250 or cfg['roles']!=['graph','removed','fusion'] or cfg['seeds']!=[17,29,43]:raise RuntimeError('Frozen architecture/support settings differ')
    for rel,key in [('vendor/frozen/hundman_detector.py','frozen_detector_sha256'),('vendor/frozen/metrics.py','frozen_evaluator_sha256'),('vendor/exp013/architectures.py','frozen_architecture_sha256')]:checked(ROOT/rel,contract[key])
    checked(root/'inputs/freeze/STUDY_FREEZE.json',contract['study_freeze_sha256'])
    study=read(root/'inputs/freeze/STUDY_FREEZE.json');final=read(root/'inputs/freeze/FINAL_AUDIT.json')
    if final['study_freeze_sha256']!=contract['study_freeze_sha256'] or final['score_seal_sha256']!=contract['known_required_files']['scores/SCORE_SEAL.json']:raise RuntimeError('Original final freeze links differ')
    for n,h in contract['known_required_files'].items():
        if n!='scores/SCORE_SEAL.json' and study['files'].get(n)!=h:raise RuntimeError('Source contract has lost STUDY binding')
    for name in ['DATA_FREEZE.json','RUN_FREEZE.json','ENGINEERING_GATE.json']:
        if name in study['files']:checked(root/'inputs/freeze'/name,study['files'][name])
    for name,seal in [('exp012_reports','LABEL_EVALUATION_COMPLETE.json'),('exp013_reports','COMPLETE.json'),('exp014_reports','COMPLETE.json')]:manifest(root/'inputs'/name,seal)
    if not cfg.get('fixture',False):
        if len(rows)!=81 or sum(r['mission']=='MSL' for r in rows.values())!=27:raise RuntimeError('Common NASA roster differs')
        if read(root/'evidence/EXP014_FINAL_AUDIT.json')['new_score_streams']!=81:raise RuntimeError('Completed EXP014 receipt differs')
    return inventory,rows

def required_paths(root):
    root=Path(root);cfg=read(root/'config.json');contract=read(root/'INPUT_CONTRACT.json');rows=read(root/'inputs/TEST_ROSTER.json')
    return {**contract['known_required_files'],**{f"scores/units/{row['mission']}__{role}__{seed}__{ch}/{leaf}":'hash from the pinned original SCORE_SEAL.json' for ch,row in rows.items() for role in cfg['roles'] for seed in cfg['seeds'] for leaf in ['COMMIT.json','record.json','scores.npz']}}

def recover(root,source,dst):
    root=Path(root);dst=Path(dst);cfg=read(root/'config.json');contract=read(root/'INPUT_CONTRACT.json')
    copied={};origin=Source(source)
    try:
        def take(n,h):
            b=origin.payload(n)
            if hashlib.sha256(b).hexdigest()!=h:raise RuntimeError('Frozen EXP013 constituent differs: '+n)
            p=dst/safe_rel(n)
            if p.exists():checked(p,h)
            else:atomic(p,b)
            copied[n]=h
        for n,h in contract['known_required_files'].items():take(n,h)
        seal=read(dst/'scores/SCORE_SEAL.json');study=contract['study_freeze_sha256']
        if seal['study_freeze_sha256']!=study or seal['NASA_labels_opened']:raise RuntimeError('Wrong original score-seal identity')
        rows=read(root/'inputs/TEST_ROSTER.json')
        for role in cfg['roles']:
            for seed in cfg['seeds']:
                for ch,row in rows.items():
                    job=f"{row['mission']}__{role}__{seed}";jid=job+'__'+ch
                    for leaf in ['COMMIT.json','record.json','scores.npz']:
                        n=f'units/{jid}/{leaf}'
                        if n not in seal['files']:raise RuntimeError('Native reference absent: '+n)
                        take('scores/'+n,seal['files'][n])
                    base=dst/'scores/units'/jid;unit=read(base/'COMMIT.json');record=read(base/'record.json')
                    th=read(dst/'calibration'/job/'thresholds.json');t=th['channels'][ch]
                    if record['job_id']!=jid or record['identity']!=unit['identity'] or record['status']!='FINITE_SCORES':raise RuntimeError('Incomplete native reference: '+jid)
                    identity=record['identity']
                    if identity['TEST_array_sha256']!=row['sha256'] or identity['study']!=study or identity['role']!=role or identity['seed']!=seed or identity['channel']!=ch or identity['mission']!=row['mission']:raise RuntimeError('Native provenance differs: '+jid)
                    if identity['model_state_sha256']!=th['model_state_sha256'] or record['threshold']!=t['threshold'] or record['detector_span']!=t['span']:raise RuntimeError('Native threshold/state binding differs: '+jid)
                    if 'threshold_record_sha256' in identity and identity['threshold_record_sha256']!=__import__('integrity').digest(t):raise RuntimeError('Native channel threshold-record identity differs')
                    for n,h in unit['files'].items():checked(base/safe_rel(n),h)
                    a=arrays(base/'scores.npz');ix=np.arange(250,row['rows'],dtype=np.int64)
                    if set(a)!=set(['target_index','native_score','detector_score','flags','issue_index']) or not np.array_equal(a['target_index'],ix):raise RuntimeError('Native score schema/support differs')
                    if any(v.shape!=ix.shape for v in a.values()) or a['flags'].dtype!=np.bool_ or not np.isfinite(a['native_score']).all() or not np.isfinite(a['detector_score']).all():raise RuntimeError('Invalid native reference vectors')
        expected=len(rows)*len(cfg['roles'])*len(cfg['seeds'])
        return {'verified_constituents':copied,'selected_checkpoints':2*len(cfg['roles'])*len(cfg['seeds']),
                'native_reference_streams':expected,'score_seal_sha256':sha(dst/'scores/SCORE_SEAL.json'),
                'study_freeze_sha256':study,'reference_predictions_saved':False,'source_scope':'18 frozen graph/removal/fusion fits and their 729 native TEST references; no previous scientific results altered'}
    finally:origin.close()

