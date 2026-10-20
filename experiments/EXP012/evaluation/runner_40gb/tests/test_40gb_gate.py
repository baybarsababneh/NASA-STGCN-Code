"""Synthetic/static checks only. These do not authorise real 40GB use."""
import unittest,sys,json,tempfile,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import write,sha,commit
from runtime_compatibility import check_environment,verify_receipt
from validate_40gb import exact_array,NominalOnlyGuard

class GateTests(unittest.TestCase):
 def setUp(self):
  self.ref={'gpu':'NVIDIA A100-SXM4-80GB','gpu_bytes':85094825984,'capability':[8,0],'driver':'580.82.07','numpy':'2.1.3','tf32':False,'threads':1,'packages':{'torch':'fixed'},'torch_build':'fixed'}
  self.policy={'changed_runtime_fields':['gpu','gpu_bytes'],'target_runtime_fields':{'gpu':'NVIDIA A100-SXM4-40GB','gpu_bytes':42405855232}}
  self.actual={**copy.deepcopy(self.ref),**self.policy['target_runtime_fields']}
 def test_only_gpu_name_and_capacity_are_accepted(self):
  r=check_environment(self.actual,self.ref,self.policy);self.assertTrue(r['all_other_fields_exact'])
 def test_every_other_runtime_field_is_still_exact(self):
  for k in set(self.ref)-{'gpu','gpu_bytes'}:
   with self.subTest(k=k):
    a=copy.deepcopy(self.actual);a[k]=None
    with self.assertRaises(RuntimeError):check_environment(a,self.ref,self.policy)
 def test_wrong_device_capacity_schema_and_scope_rejected(self):
  for k,v in [('gpu','NVIDIA A100-SXM4-80GB'),('gpu','NVIDIA H100'),('gpu_bytes',85094825984)]:
   a=copy.deepcopy(self.actual);a[k]=v
   with self.assertRaises(RuntimeError):check_environment(a,self.ref,self.policy)
  a=copy.deepcopy(self.actual);a['unexpected']=1
  with self.assertRaises(RuntimeError):check_environment(a,self.ref,self.policy)
  policy=copy.deepcopy(self.policy);policy['changed_runtime_fields'].append('driver')
  with self.assertRaises(RuntimeError):check_environment(self.actual,self.ref,policy)
 def test_bitwise_comparison_has_no_tolerance(self):
  a=np.array([1.,0.],np.float32);self.assertTrue(exact_array(a,a.copy(),'synthetic'))
  for b in [np.array([1.,-0.],np.float32),np.array([1.+1e-6,0.],np.float32),a.astype(np.float64),a.reshape(1,2)]:
   with self.assertRaises(RuntimeError):exact_array(a,b,'synthetic')
 def test_missing_receipt_blocks_authorization(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaisesRegex(RuntimeError,'NOT AUTHORISED'):verify_receipt(t,{},[],{})
 def test_no_label_or_test_array_access_in_validator(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);g=NominalOnlyGuard(t/'package',t/'results',t)
   for path in ['labeled_anomalies.csv','renamed.csv','archive.zip','test/X.npy','elsewhere/X.npz']:
    with self.assertRaises(PermissionError):g.hook('open',(str(t/path),'r',0))
   g.hook('open',(str(t/'package/development/SYNTHETIC.npy'),'r',0))
 def test_complete_synthetic_receipt_verified_and_tampering_blocked(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);ident={'fixture':'SYNTHETIC_NO_GPU'};write(t/'COMPATIBILITY_RUN_FREEZE.json',ident)
   roster=[f'SYNTHETIC-{i}' for i in range(1053)];weights={jid:{'state_sha256':'synthetic-state'} for jid in roster[:972]}
   for jid in roster:commit(t,jid,{'synthetic':True},{'REPLAY_RECORD.json':json.dumps({'job_id':jid,'exact_pass':True,'selected_state_sha256':'synthetic-state' if jid in weights else None}).encode()})
   receipt={'status':'PASS_40GB_NOMINAL_BITWISE_REPLAY','identity':ident,'run_freeze_sha256':sha(t/'COMPATIBILITY_RUN_FREEZE.json'),'roster':roster,'files':{p.relative_to(t).as_posix():sha(p) for p in (t/'units').rglob('*') if p.is_file()},'counts':{'streams':1053,'checkpoint_states':972,'nominal_split_streams':2106,'detector_split_streams':4212,'adaptive_windows':2080,'mismatches':0,'scientific_fits':0},'NASA_TEST_opened':False,'NASA_labels_opened':False,'scientific_settings_changed':False}
   write(t/'COMPATIBILITY_PASS.json',receipt);self.assertTrue(verify_receipt(t,ident,roster,weights))
   wrong=copy.deepcopy(receipt);wrong['counts']['mismatches']=1;write(t/'COMPATIBILITY_PASS.json',wrong)
   with self.assertRaises(RuntimeError):verify_receipt(t,ident,roster,weights)
   write(t/'COMPATIBILITY_PASS.json',receipt)
   with self.assertRaises(RuntimeError):verify_receipt(t,{'fixture':'other'},roster,weights)
   (t/'units'/roster[0]/'REPLAY_RECORD.json').write_text('{}')
   with self.assertRaises(RuntimeError):verify_receipt(t,ident,roster,weights)

if __name__=='__main__':unittest.main(verbosity=2)
