import sys, unittest, tempfile, shutil, zipfile, json, os
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(CODE/'src'),str(CODE),str(CODE/'tests')]
import numpy as np
from integrity import *
from inputs import Source,recover,verify_static
from backup import Backup,EnvironmentMismatch
from fixture import create
from run import execute,snapshot

class EngineeringTests(unittest.TestCase):
    def test_atomic_units_stale_identity_corruption_and_staging(self):
        with tempfile.TemporaryDirectory() as td:
            work=Path(td)/'work';p=commit(work,'u',{'run':1},{'value':b'old'})
            self.assertEqual(p,commit(work,'u',{'run':1},{'value':b'new'}));self.assertEqual((p/'value').read_bytes(),b'old')
            with self.assertRaises(RuntimeError):committed(work,'u',{'run':2})
            (work/'units/.tmp-interrupted').mkdir();(work/'units/.tmp-interrupted/value').write_bytes(b'partial')
            discard_uncommitted(work);self.assertFalse((work/'units/.tmp-interrupted').exists())
            (p/'value').write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError):committed(work,'u',{'run':1})

    def test_backup_verified_fallback_and_environment_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);w=base/'work';d=base/'drive';c=base/'cache';w.mkdir()
            env={'device':'CPU_TEST_ONLY'};write(w/'RUN_FREEZE.json',{'environment':env});(w/'value').write_bytes(b'one')
            backup=Backup(w,d,c,{'input':1});first=backup.save('first')
            (w/'value').write_bytes(b'two');second=backup.save('second')
            (d/'objects'/second['objects']['state_archive']).write_bytes(b'corrupt-newest')
            shutil.rmtree(w);self.assertTrue(backup.restore(env));self.assertEqual((w/'value').read_bytes(),b'one')
            shutil.rmtree(w)
            with self.assertRaises(EnvironmentMismatch):backup.restore({'device':'DIFFERENT'})
            self.assertTrue((d/'objects'/first['objects']['state_archive']).exists())

    def test_archive_paths_and_duplicate_entries(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../outside',b'bad')
            with self.assertRaises(RuntimeError):Source(p)
            with self.assertRaises(RuntimeError):safe_extract(p,Path(td)/'out')
            with zipfile.ZipFile(p,'w') as z:z.writestr('same',b'a');z.writestr('same',b'b')
            with self.assertRaises(RuntimeError):Source(p)

    def test_label_parse_and_open_guards(self):
        from report import labels
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'labeled_anomalies.csv';p.write_text('not parsed')
            guard=LabelGuard(p);guard.install()
            try:
                with self.assertRaises(PermissionError):p.read_text()
                with self.assertRaises(RuntimeError):labels(td,False)
                guard.allowed=True;self.assertEqual(p.read_text(),'not parsed')
            finally:guard.active=False

    def test_complete_fixture_interruption_recovery_and_export(self):
        import torch
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root,source=create(base/'fixture')
            archive=base/'source.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for p in source.rglob('*'):
                    if p.is_file():z.write(p,p.relative_to(source).as_posix())
            w=base/'work';d=base/'drive';c=base/'cache'
            # First channel committed and backed up, then the local runtime is lost.
            with self.assertRaises(InterruptedError):execute(root,archive,w,d,c,'cpu',True,1)
            unit=next(p for p in (w/'units').iterdir() if p.is_dir());jid=unit.name;unit_hash=sha(unit/'scores.npz')
            self.assertFalse((w/'SCORE_SEAL.json').exists());self.assertFalse((w/'reports').exists())
            shutil.rmtree(w);shutil.rmtree(c)
            result=execute(root,archive,w,d,c,'cpu',True)
            self.assertEqual(result['status'],'CPU_SYNTHETIC_FIXTURE_COMPLETE');self.assertFalse(result['real_NASA_comparison'])
            self.assertEqual(sha(w/'units'/jid/'scores.npz'),unit_hash)
            audit=read(w/'FINAL_AUDIT.json');self.assertFalse(audit['A100_validation']);self.assertEqual(audit['new_scientific_fits'],0)
            manifest(w/'reports','COMPLETE.json')
            from export import export
            with self.assertRaises(RuntimeError):export(w,base/'report_export')
            export(w,base/'report_export',allow_fixture=True)
            self.assertEqual(sha(base/'report_export/mission_metrics.csv'),sha(w/'reports/mission_metrics.csv'))
            self.assertTrue((d/'FINAL_RESULTS.zip').is_file());latest=read(d/'LATEST_EXPORT.json');checked(d/latest['archive'],latest['sha256'])
            with zipfile.ZipFile(d/latest['archive']) as z:
                m=json.loads(z.read('SNAPSHOT_MANIFEST.json'))
                for n,h in m['files'].items():self.assertEqual(__import__('hashlib').sha256(z.read(n)).hexdigest(),h)
            events=[json.loads(l) for l in (w/'PROGRESS.jsonl').read_text().splitlines()]
            first_label=next(i for i,r in enumerate(events) if r['action']=='ANNOTATION_CSV_PARSED')
            self.assertTrue(any(r['action']=='SCORE_SEAL_VERIFIED_BEFORE_ANNOTATION_PARSE' for r in events[:first_label]))
            # A completed run skips every scientific unit and retains the same reports.
            hashes={p.name:sha(p/'scores.npz') for p in (w/'units').iterdir()}
            again=execute(root,archive,w,d,c,'cpu',True);self.assertEqual(result,again)
            self.assertEqual(hashes,{p.name:sha(p/'scores.npz') for p in (w/'units').iterdir()})
            # A different local/durable workspace runs the same numerical workflow.
            clean=execute(root,archive,base/'clean_work',base/'clean_drive',base/'clean_cache','cpu',True)
            self.assertEqual(clean['new_comparisons'],result['new_comparisons'])
            for p in (w/'units').iterdir():
                a=arrays(p/'scores.npz');b=arrays(base/'clean_work/units'/p.name/'scores.npz')
                for k in a:self.assertTrue(np.array_equal(a[k],b[k]))
            # An original source changed after a valid cache was made must fail.
            target=source/'models/MSL__gdn__17/selected.pt';payload=target.read_bytes();target.write_bytes(payload[:-1]+b'x')
            with self.assertRaises(RuntimeError):recover(root,source,c/'verified_source')
            target.write_bytes(payload)
            # Corrupted committed output cannot be promoted on a later resume.
            p=next((w/'units').iterdir())/'scores.npz';p.write_bytes(b'corrupted')
            with self.assertRaises(RuntimeError):execute(root,archive,w,d,c,'cpu',True)

if __name__=='__main__':unittest.main()
