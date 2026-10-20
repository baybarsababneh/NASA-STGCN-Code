"""Synthetic-only tests. Never opens NASA arrays or NASA labels."""
import unittest,tempfile,sys,json,csv,subprocess,shutil,platform
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import sha,write,commit,npz_bytes,verify_manifest,reuse,safe_extract
from metrics import channel,pooled,fbeta,ap,bootstrap
from export_scores import verify_scores,LabelBlindGuard,stage_test

class MetricTests(unittest.TestCase):
 def test_undefined_before_zero_recall(self):
  self.assertTrue(np.isnan(fbeta(float('nan'),0.)))
  self.assertTrue(np.isnan(pooled([[0,1,0,0,0]])['corrected_event_f0_5']))
  self.assertEqual(pooled([[0,1,0,0,5]])['corrected_event_f0_5'],0.)
 def test_tied_ap(self):
  self.assertEqual(ap([1,0,1,0],[.8,.8,.3,.3]),.5)
  self.assertEqual(ap([0,1],[0.,1.]),1.)
  self.assertTrue(np.isnan(ap([0,0],[4.,1.])))
 def test_events_counts_original_delay_and_outside(self):
  ix=np.arange(250,260);f=np.array([1,1,0,0,0,1,0,0,0,0],bool);issue=np.where(f,259,-1)
  r,er,out=channel([(240,252),(1,3)],ix,f,issue,np.arange(10.),np.arange(10.))
  self.assertEqual([r[k] for k in ['H','M','U','FPt','N0']],[1,0,1,1,8])
  self.assertEqual(er[0]['delay'],19);self.assertTrue(er[0]['left_truncated']);self.assertTrue(er[0]['late_post_event_notification'])
  self.assertEqual(len(out),1);self.assertEqual(r['nominal_alarm_starts'],1)
 def test_empty_support_and_all_anomaly(self):
  r,er,out=channel([(3,9)],[],[],[],[],[])
  self.assertEqual(r['samples'],0);self.assertEqual(len(out),1);self.assertTrue(np.isnan(r['AP_raw']));self.assertTrue(np.isnan(r['corrected_event_f0_5']))
  r,_,_=channel([(250,253)],np.arange(250,253),[0,0,0],[-1,-1,-1],[0,0,0],[0,0,0])
  self.assertEqual(r['M'],1);self.assertTrue(np.isnan(r['corrected_event_f0_5']))
 def test_bootstrap_seed_group_and_single_persistence(self):
  a=np.tile([1,0,0,0,5],(4,3,1));b=np.tile([0,1,0,0,5],(4,1,1))
  v=bootstrap(a,b,10000,20261005);self.assertEqual(v['difference'],1.);self.assertEqual(v['percentile_CI95'],[1.,1.]);self.assertEqual(v['undefined_replicates'],0)
  self.assertEqual(bootstrap(a,a,100,20261005)['percentile_CI95'],[0.,0.])
 def test_phase_a_label_guard(self):
  with tempfile.TemporaryDirectory() as t:
   g=LabelBlindGuard(Path(t)/'cache',t)
   with self.assertRaises(PermissionError):g.hook('open',(str(Path(t)/'labeled_anomalies.csv'),'r',0))
   with self.assertRaises(PermissionError):g.hook('open',(str(Path(t)/'train/X.npy'),'r',0))
 def test_commit_reuse_corruption(self):
  with tempfile.TemporaryDirectory() as t:
   b=commit(t,'x',{'run':'synthetic'},{'a.txt':b'a'});self.assertEqual(reuse(t,'x',{'run':'synthetic'}),b)
   (b/'a.txt').write_bytes(b'b')
   with self.assertRaises(RuntimeError):reuse(t,'x',{'run':'synthetic'})
 def test_selected_zip_members_only(self):
  import zipfile
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);archive=t/'synthetic.zip'
   with zipfile.ZipFile(archive,'w') as z:
    z.writestr('data/data/test/SYNTHETIC.npy',b'synthetic only');z.writestr('data/data/train/SYNTHETIC.npy',b'NEVER READ');z.writestr('labeled_anomalies.csv',b'NEVER READ');z.writestr('data/data/test/T-10.npy',b'NEVER READ')
   calls=[];read=zipfile.ZipFile.read
   def logged(z,name,*a,**k):calls.append(name);return read(z,name,*a,**k)
   with patch.object(zipfile.ZipFile,'read',logged):stage_test(archive,t/'cache',[{'channel':'SYNTHETIC'}])
   self.assertEqual(calls,['data/data/test/SYNTHETIC.npy'])
 def test_separate_complete_phase_b_with_synthetic_81_channel_fixture(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);runner=t/'runner';shutil.copytree(ROOT/'src',runner/'src');scores=t/'scores';scores.mkdir();out=t/'reports'
   channels=['P-2']+[f'SYNTHETIC-{i}' for i in range(80)];metas={};label_path=t/'synthetic_labels.csv'
   with label_path.open('w') as f:
    w=csv.DictWriter(f,fieldnames=['chan_id','spacecraft','num_values','anomaly_sequences']);w.writeheader()
    for i,ch in enumerate(channels):
     n=200 if i==0 else 262;mission='MSL' if i<27 else 'SMAP';events=[[0,1]] if i==0 or i==1 else [[251,252]]
     if i<23:events+=[[3,4]]
     w.writerow(dict(chan_id=ch,spacecraft=mission,num_values=n,anomaly_sequences=str(events)));metas[ch]=(n,mission)
    w.writerow(dict(chan_id='P-2',spacecraft='MSL',num_values=200,anomaly_sequences='[[10,11]]'))
   contract={'historical_labels_sha256':sha(label_path),'fixture':'SYNTHETIC_ONLY_NOT_NASA'};write(runner/'EVALUATION_BINDING.json',{'version':'SYNTHETIC_TEST','evaluation_contract':contract,'channels':[{'channel':ch,'mission':v[1]} for ch,v in metas.items()],'label_metrics_environment':{'python':platform.python_version(),'numpy':np.__version__}})
   write(runner/'RUNNER_MANIFEST.json',{'files':{p.relative_to(runner).as_posix():sha(p) for p in runner.rglob('*') if p.is_file() and '__pycache__' not in p.parts}})
   roster=[]
   for ch,(n,mission) in metas.items():
    ix=np.arange(250,n);f=(ix==251)|(ix==255);s=ix.astype(float)
    for arm,seed in [(a,k) for a in ['gcn','removed','tcn','lstm'] for k in [17,29,43]]+[('persistence',None)]:
     jid=f'{arm}__{ch}'+(f'__{seed}' if seed is not None else '');roster.append(jid)
     record={'job_id':jid,'arm':arm,'channel':ch,'mission':mission,'seed':seed,'original_TEST_rows':n,'TEST_array_sha256':'synthetic'}
     commit(scores,jid,{'synthetic':True},{'record.json':json.dumps(record).encode(),'scores.npz':npz_bytes({'target_index':ix,'raw_flags':f,'adaptive_flags':f,'raw_issue_index':np.where(f,ix,-1),'adaptive_issue_index':np.where(f,261,-1),'scores_raw':s,'scores_ewma':s})})
   write(scores/'EVALUATION_RUN_FREEZE.json',{'synthetic':True});write(scores/'TEST_INPUT_FREEZE.json',{'synthetic':True})
   freeze={'status':'COMPLETE_LABEL_BLIND_EXPORT_LOCKED','NASA_labels_opened':False,'unique_streams':1053,'roster':roster,'contract':contract,'files':{p.relative_to(scores).as_posix():sha(p) for p in (scores/'units').rglob('*') if p.is_file()},'run_freeze_sha256':sha(scores/'EVALUATION_RUN_FREEZE.json'),'TEST_input_freeze_sha256':sha(scores/'TEST_INPUT_FREEZE.json'),'preparation_binding_sha256':sha(runner/'EVALUATION_BINDING.json'),'runner_manifest_sha256':sha(runner/'RUNNER_MANIFEST.json')}
   write(scores/'SCORE_EXPORT_FREEZE.json',freeze);before=sha(scores/'SCORE_EXPORT_FREEZE.json');verify_scores(scores)
   cmd=[sys.executable,str(runner/'src/evaluate_labels.py'),'--scores',str(scores),'--labels',str(label_path),'--output',str(out),'--evaluate-locked-exports']
   p=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
   complete=verify_manifest(out,'LABEL_EVALUATION_COMPLETE.json');self.assertEqual(complete['channel_detector_rows'],2106)
   audit=json.loads((out/'LABEL_ROSTER_AUDIT.json').read_text());self.assertEqual(audit['original_events'],104);self.assertEqual(len(audit['duplicates']),1)
   with (out/'channel_metrics.csv').open() as f:rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),2106)
   with (out/'mission_metrics.csv').open() as f:mission=list(csv.DictReader(f))
   self.assertEqual(len(mission),20);self.assertTrue(all(v['seed_records']=='1' for v in mission if v['arm']=='persistence'))
   contrasts=json.loads((out/'paired_channel_bootstrap.json').read_text());self.assertEqual(len(contrasts),16);self.assertTrue(all(v['repetitions']==10000 for v in contrasts))
   again=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(again.returncode,0,again.stderr);self.assertIn('no recomputation',again.stdout);self.assertEqual(before,sha(scores/'SCORE_EXPORT_FREEZE.json'))
   stream=scores/'units'/roster[0]/'record.json';stream.write_text('{}')
   bad=subprocess.run(cmd,capture_output=True,text=True);self.assertNotEqual(bad.returncode,0);self.assertIn('Score export changed',bad.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
