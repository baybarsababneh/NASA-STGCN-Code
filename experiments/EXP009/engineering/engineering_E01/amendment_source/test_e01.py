"""Synthetic regression tests. Never performs EXP009 selection on project residuals."""
import contextlib,hashlib,io,json,sys,tempfile,types,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from json_e01 import dumps_e01

PARENT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent.parent/'EXP009_DETECTOR_V1'
sys.argv=sys.argv[:1];sys.path.insert(0,str(PARENT/'src'));import run_exp009 as r
legacy=r.dumps

class ScalarTests(unittest.TestCase):
    def test_nested_scalars(self):
        x=dict(a=np.int64(22),b=[np.uint64(2**63+5),np.float32(1.25),np.bool_(True)])
        self.assertEqual(json.loads(dumps_e01(x)),dict(a=22,b=[2**63+5,1.25,True]))
    def test_legacy_bytes(self):
        for x in (None,True,1,1.5,[1,'a'],dict(a=1,b=[False,None,1.23])):self.assertEqual(legacy(x),dumps_e01(x))
        manifest=json.loads((PARENT/'MANIFEST.json').read_text());self.assertEqual(legacy(manifest),dumps_e01(manifest))
    def test_reject_nonfinite_and_unknown(self):
        for x in (float('nan'),float('inf'),np.float32('nan'),np.float64('inf')):
            with self.assertRaises(ValueError):dumps_e01(dict(x=x))
        for x in (np.array([1,2]),complex(1,2),object()):
            with self.assertRaises(TypeError):dumps_e01(dict(x=x))
    def test_actual_short_segment_failure(self):
        for n in (439,312):
            lo=int(.6*n);hi=int(.8*n)
            p=dict(job=dict(mission='MSL',channel='synthetic_short',seed=17,fold='canonical'),reference_result=dict(num_values=n,fit_end=lo,stop_end=hi))
            a=dict(target_index=np.arange(lo,hi),target=np.zeros(hi-lo),prediction=np.zeros(hi-lo),persistence=np.zeros(hi-lo))
            result=r.evaluate(p,a,'gcn',dict(id='absolute_h2',family='absolute',h=2),'selection',1e-6)
            with self.assertRaises(TypeError):legacy(result)
            saved=json.loads(dumps_e01(result));self.assertEqual(len(saved['probes']),48)
            for x,y in zip(result['probes'],saved['probes']):
                self.assertEqual(x,y)
    def test_900_commit_resume(self):
        with zipfile.ZipFile(PARENT/'data/EXP008_results.zip')as z:plan=json.loads(z.read('PLAN.json'))
        streams=[]
        for p in plan:
            rr=p['reference_result'];t=np.arange(rr['fit_end'],rr['stop_end']if p['job']['fold']=='canonical'else rr['assess_end'])
            streams.append((p,dict(target_index=t,target=np.zeros(len(t)),prediction=np.zeros(len(t)),persistence=np.zeros(len(t)))))
        calls=0
        def fake(p,a,model,c,stage,scale):
            nonlocal calls
            calls+=1
            meta=dict(mission=p['job']['mission'],channel=p['job']['channel'],fold=p['job']['fold'],seed=0 if model=='persistence'else p['job']['seed'],model=model,candidate=c['id'],stage=stage)
            nominal=dict(**meta,occupancy=0.,starts_per_1000=0.)
            # First 900 units are valid under V1. Unit 901 recreates the reported serializer failure.
            duration=np.int64(22)if calls>=901 else 32
            probes=[dict(**meta,family=f,amplitude=amp,incremental_hit=True,duration=duration)for f in ('spike','step8','step32','ramp32')for amp in (2,4,8)]
            return dict(nominal=nominal,probes=probes)
        with tempfile.TemporaryDirectory(prefix='EXP009_E01_SYNTHETIC_')as tmp:
            out=Path(tmp)/'results';args=types.SimpleNamespace(input=PARENT/'data/EXP008_results.zip',output=out)
            with patch.object(r,'load_parent',return_value=streams),patch.object(r,'evaluate',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(TypeError,'int64'):r.run(args)
            markers=list((out/'units').glob('*.done.json'));self.assertEqual(len(markers),900)
            protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in (out/'units').glob('*.json')}
            frozen=(out/'RUN_FREEZE.json').read_bytes()
            import resume_e01
            argv=['resume_e01.py','--parent',str(PARENT),'--output',str(out),'--evidence',str(Path(tmp)/'engineering')]
            with patch.object(sys,'argv',argv+['--preflight-only']),contextlib.redirect_stdout(io.StringIO()):resume_e01.main()
            # Run the real amendment launcher with only scientific numerical outputs mocked.
            before=calls
            with patch.object(sys,'argv',argv),patch.object(r,'load_parent',return_value=streams),patch.object(r,'evaluate',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):resume_e01.main()
            self.assertEqual(calls-before,4833-900)
            self.assertEqual((out/'RUN_FREEZE.json').read_bytes(),frozen)
            for p,h in protected.items():self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(),h)
            self.assertEqual(len(list((out/'units').glob('*.done.json'))),4833)
            # Restarting the E01 launcher must not recompute any committed detector unit.
            with patch.object(sys,'argv',argv),patch.object(r,'load_parent',return_value=streams),patch.object(r,'evaluate',side_effect=AssertionError('Unexpected recomputation')),contextlib.redirect_stdout(io.StringIO()):resume_e01.main()
            # Refuse a corrupted original result before changing any records.
            victim=Path(next(iter(protected)));victim.write_text('{}')
            with patch.object(sys,'argv',argv+['--preflight-only']),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(AssertionError):resume_e01.main()
            r.dumps=legacy

if __name__=='__main__':unittest.main(verbosity=2)
