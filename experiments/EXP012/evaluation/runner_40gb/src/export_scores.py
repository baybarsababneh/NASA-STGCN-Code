"""Phase A. Frozen weights/thresholds only. No label input or training calls."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');sys.dont_write_bytecode=True
import argparse,json,zipfile,time,hashlib
from pathlib import Path
import numpy as np
from common import *
ROOT=Path(__file__).resolve().parents[1]
class LabelBlindGuard:
 def __init__(self,cache,out):self.cache=Path(cache).resolve();self.out=Path(out)
 def hook(self,event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve();name=p.name.lower()
  if 'labeled_anomal' in name or 'labelled_anomal' in name:raise PermissionError('Labels prohibited in phase A')
  if p.suffix=='.npy':
   if p.parent!=self.cache:raise PermissionError('Array outside selected TEST cache')
   with open(self.out/'LABEL_BLIND_ACCESS_EVENTS.jsonl','a') as f:f.write(json.dumps({'path':p.name,'phase':'FROZEN_TEST_SCORE_EXPORT'})+'\n');f.flush();os.fsync(f.fileno())
 def install(self):sys.addaudithook(self.hook)
def stage_test(path,cache,rows):
 cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);hashes={}
 with zipfile.ZipFile(path) as z:
  if len(z.namelist())!=len(set(z.namelist())):raise RuntimeError('Duplicate NASA ZIP paths')
  for row in rows:
   name='data/data/test/'+row['channel']+'.npy'
   if name not in z.namelist():raise RuntimeError('Missing TEST array '+name)
   # No TRAIN array, model or label ZIP entry is opened.
   blob=z.read(name);h=hashlib.sha256(blob).hexdigest();dst=cache/(row['channel']+'.npy')
   if dst.exists():
    if sha(dst)!=h:raise RuntimeError('Changed cached TEST input')
   else:atomic(dst,blob)
   hashes[row['channel']]=h
 return hashes
def verify_scores(out):
 out=Path(out);freeze_record=json.loads((out/'SCORE_EXPORT_FREEZE.json').read_text());expected=freeze_record['files']
 actual={p.relative_to(out).as_posix() for p in (out/'units').rglob('*') if p.is_file() and '.tmp-' not in p.name}
 if actual!=set(expected):raise RuntimeError('Score export roster differs')
 for n,h in expected.items():
  if sha(out/n)!=h:raise RuntimeError('Score export changed '+n)
 if sha(out/'EVALUATION_RUN_FREEZE.json')!=freeze_record['run_freeze_sha256']:raise RuntimeError('Score-run identity changed')
 if sha(out/'TEST_INPUT_FREEZE.json')!=freeze_record['TEST_input_freeze_sha256']:raise RuntimeError('TEST input freeze changed')
 if freeze_record['status']!='COMPLETE_LABEL_BLIND_EXPORT_LOCKED' or freeze_record['NASA_labels_opened'] or freeze_record['unique_streams']!=1053 or len(freeze_record['roster'])!=len(set(freeze_record['roster'])) or len(freeze_record['roster'])!=1053:raise RuntimeError('Invalid sealed export')
 return freeze_record
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--package',type=Path,required=True);ap.add_argument('--nasa-archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--execute-test',action='store_true');ap.add_argument('--compatibility-directory',type=Path,required=True);a=ap.parse_args()
 if not a.execute_test:raise RuntimeError('Phase A requires explicit --execute-test; this program never trains')
 verify_manifest(ROOT,'RUNNER_MANIFEST.json');lock=json.loads((ROOT/'EVALUATION_BINDING.json').read_text());r=a.results.resolve();p=a.package.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 if out==r or r in out.parents or out==p or p in out.parents:raise RuntimeError('Keep evaluation outputs separate')
 verify_manifest(r,'OUTPUT_MANIFEST.json');verify_manifest(p,'PACKAGE_MANIFEST.json')
 for key,file in [('results_manifest_sha256',r/'OUTPUT_MANIFEST.json'),('model_freeze_sha256',r/'MODEL_FREEZE.json'),('calibration_freeze_sha256',r/'CALIBRATION_FREEZE.json'),('benchmark_freeze_sha256',r/'BENCHMARK_FREEZE.json'),('package_manifest_sha256',p/'PACKAGE_MANIFEST.json')]:
  if sha(file)!=lock[key]:raise RuntimeError('Preparation binding differs '+key)
 contract=json.loads((p/'evaluation_contract.json').read_text())
 if contract!=lock['evaluation_contract']:raise RuntimeError('Evaluation contract differs')
 if sha(a.nasa_archive)!=lock['NASA_archive_sha256']:raise RuntimeError('NASA archive bytes differ from the audited input archive')
 sys.path.insert(0,str(r/'reproducibility/src'))
 import torch
 from integrity import configure_cuda
 from models_io import read_parent,read_tcn,standardize,predict,tensor_hash
 from lstm_baseline import ControlledLSTM
 from hundman_detector import detect
 env=configure_cuda();run=json.loads((r/'RUN_FREEZE.json').read_text())
 from runtime_compatibility import check_environment,require_pass
 check_environment(env,run['environment'],json.loads((ROOT/'COMPATIBILITY_AMENDMENT.json').read_text()))
 receipt=require_pass(a.compatibility_directory,env,r,p,ROOT)
 cfg=json.loads((p/'config.json').read_text());rows=json.loads((p/'data_manifest.json').read_text());plan=json.loads((p/'parent_plan.json').read_text());parent={(m['job']['channel'],m['job']['seed'],m['job']['remove_messages']):m for m in plan};mf=json.loads((r/'MODEL_FREEZE.json').read_text());cf=json.loads((r/'CALIBRATION_FREEZE.json').read_text());weights={v['job_id']:v for v in mf['weights']};thresholds={v['job_id']:v['threshold']['threshold'] for v in cf['thresholds']}
 if len(weights)!=972 or len(thresholds)!=1053:raise RuntimeError('Preparation roster incomplete')
 identity={'version':lock['version'],'runner_manifest_sha256':sha(ROOT/'RUNNER_MANIFEST.json'),'preparation_binding_sha256':sha(ROOT/'EVALUATION_BINDING.json'),'environment':env,'NASA_archive_sha256':sha(a.nasa_archive),'evaluation_contract':contract,'common_target_start':250,'label_access':False,'compatibility_pass_sha256':receipt}
 freeze(out/'EVALUATION_RUN_FREEZE.json',identity)
 if (out/'SCORE_EXPORT_FREEZE.json').exists():verify_scores(out);print('EXISTING LOCKED EXPORT VERIFIED; no additional inference');return
 with Lease(out):
  cache=out/'test_cache';LabelBlindGuard(cache,out).install();input_hashes=stage_test(a.nasa_archive,cache,rows);freeze(out/'TEST_INPUT_FREEZE.json',{'archive_sha256':identity['NASA_archive_sha256'],'selected_array_sha256':input_hashes,'channels':81,'labels_opened':False});roster=[]
  for row in rows:
   original=np.load(cache/(row['channel']+'.npy'),allow_pickle=False)
   if original.ndim!=2 or original.shape[1]!=row['columns'] or not np.isfinite(original).all():raise RuntimeError('Invalid TEST shape/values')
   n=len(original);raw=standardize(original,row['target_mean'],row['target_scale']);ix=np.arange(250,n,dtype=np.int64)
   for arm,seed in [(arm,seed) for arm in ['gcn','removed','tcn','lstm'] for seed in [17,29,43]]+[('persistence',None)]:
    stem=f"{arm}__{row['channel']}"+(f'__{seed}' if seed is not None else '');uid={'run_hash':digest(identity),'job_id':stem};b=reuse(out,stem,uid)
    if b is None:
     model=None
     if len(ix)==0:pred=np.empty(0,np.float32)
     elif arm=='persistence':pred=raw[249:n-1,0].copy()
     else:
      if arm in ['gcn','removed']:
       model,ck,ref=read_parent(p,parent[(row['channel'],seed,arm=='removed')]);features=ref['graph_features']
      elif arm=='tcn':model,ck,ref=read_tcn(r/'parent_weights/exp005'/f"{row['channel']}_seed{seed}.pt",row,row['incoming_features']);features=row['incoming_features']
      else:
       ck=torch.load(r/'units'/stem/'checkpoint.pt',map_location='cpu',weights_only=True);model=ControlledLSTM(len(row['incoming_features']));model.load_state_dict(ck['state_dict'],strict=True);features=row['incoming_features']
      model=model.to('cuda');before=tensor_hash(model.state_dict())
      if before!=weights[stem]['state_sha256']:raise RuntimeError('Selected state differs before TEST inference')
      pred,_=predict(model,raw,row,features,250,n,64);again,_=predict(model,raw,row,features,250,n,64)
      if not np.array_equal(pred,again) or tensor_hash(model.state_dict())!=before:raise RuntimeError('TEST repeatability/state gate failed')
     y=original[250:,0].astype(np.float64);yhat=pred.astype(np.float64)*row['target_scale']+row['target_mean'];score=np.abs(raw[250:,0].astype(np.float64)-pred.astype(np.float64));q=thresholds[stem];rawflags=score>q
     if len(ix):d=detect(y,yhat,ix,cfg['detector'])
     else:d={'flags':np.empty(0,bool),'issue_index':np.empty(0,np.int64),'smoothed':np.empty(0,float),'windows':[],'status':'NO_EVALUATION_SUPPORT','startup_averaging_lookahead':0}
     arrays={'target_index':ix,'target_standardized':raw[250:,0],'prediction_standardized':pred,'target':y,'prediction':yhat,'scores_raw':score,'scores_ewma':d['smoothed'],'raw_flags':rawflags,'raw_issue_index':np.where(rawflags,ix,-1),'adaptive_flags':d['flags'],'adaptive_issue_index':d['issue_index']}
     record={'job_id':stem,'arm':arm,'channel':row['channel'],'mission':row['mission'],'seed':seed,'original_TEST_rows':n,'target_start':250,'target_end_exclusive':n,'threshold':q,'adaptive_status':d['status'],'startup_averaging_lookahead':d['startup_averaging_lookahead'],'TEST_array_sha256':input_hashes[row['channel']],'selected_state_sha256':weights[stem]['state_sha256'] if seed is not None else None,'labels_opened':False}
     b=commit(out,stem,uid,{'scores.npz':npz_bytes(arrays),'record.json':json.dumps(record,sort_keys=True).encode(),'adaptive_windows.json':json.dumps(normal(d['windows']),sort_keys=True).encode()})
     if model is not None:del model;torch.cuda.empty_cache()
    roster.append(stem);print(f'[score export {len(roster)}/1053] {stem}',flush=True)
  expected=set(thresholds)
  if set(roster)!=expected or len(roster)!=1053 or {b.name for b in (out/'units').iterdir() if b.is_dir()}!=expected:raise RuntimeError('Unexpected score job roster')
  files={f.relative_to(out).as_posix():sha(f) for f in sorted((out/'units').rglob('*')) if f.is_file() and '.tmp-' not in f.name}
  freeze(out/'SCORE_EXPORT_FREEZE.json',{'status':'COMPLETE_LABEL_BLIND_EXPORT_LOCKED','run_freeze_sha256':sha(out/'EVALUATION_RUN_FREEZE.json'),'TEST_input_freeze_sha256':sha(out/'TEST_INPUT_FREEZE.json'),'preparation_binding_sha256':sha(ROOT/'EVALUATION_BINDING.json'),'runner_manifest_sha256':sha(ROOT/'RUNNER_MANIFEST.json'),'unique_streams':1053,'roster':roster,'files':files,'NASA_labels_opened':False,'contract':contract})
  verify_scores(out)
  print('LABEL-BLIND EXPORT LOCKED. Labels have not been opened. Preserve SCORE_EXPORT_FREEZE.json before phase B.')
if __name__=='__main__':main()
