"""Release integrity and final report durability fixtures; no NASA values or labels."""
import ast,copy,hashlib,json,sys,tempfile,types,unittest,zipfile
from pathlib import Path
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import core,train,execution_reports
from recovery import LocalTransport
from core import ROOT,sha,read,write,seal

class ReleaseTests(unittest.TestCase):
    def test_original_scientific_members_and_amended_job_factory(self):
        m=read(ROOT/'E01_AMENDMENT_MANIFEST.json')
        self.assertEqual(m['counts']['unchanged_original_members'],31)
        for name,r in m['original_members'].items():
            if r['status']=='byte_identical':
                if name=='src/architectures.py':
                    self.assertEqual(sha(ROOT/'frozen_originals/architectures.py'),r['parent_sha256'])
                    self.assertEqual(sha(ROOT/name),read(ROOT/'E03_SOURCE_AMENDMENT.json')['changed_parent_members'][name]['corrected_sha256'])
                else:self.assertEqual(sha(ROOT/name),r['parent_sha256'],name)
        original=ast.parse((ROOT/'frozen_originals/train.py').read_text());amended=ast.parse((ROOT/'src/train.py').read_text())
        for name in ['jobs','job_name','model_for']:
            a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in amended.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False),name)
        self.assertEqual(len(train.jobs()),22)

    def test_original_input_archives_and_pinned_official_sources(self):
        for name in ['inputs/NASA_arrays_only.zip','inputs/EXP012_RETROSPECTIVE_EVALUATION_REPORTS.zip','inputs/pinned_gdn.zip','inputs/pinned_catch.zip']:
            with zipfile.ZipFile(ROOT/name) as z:
                self.assertIsNone(z.testzip());self.assertEqual(len(z.namelist()),len(set(z.namelist())))
        # CRC/member checks concern opaque bytes, not TEST arrays or labels.
        pins=read(ROOT/'INPUT_PINS.json')
        self.assertEqual(sha(ROOT/'inputs/NASA_arrays_only.zip'),'ccfd48b4dce0aece170a77cfb2054de3c7df4b08a753f5a3a5ee33e729098ac9')
        self.assertEqual(sha(ROOT/'inputs/EXP012_RETROSPECTIVE_EVALUATION_REPORTS.zip'),'f1e12b2f8714a0a41c1b76b9fcfdb8db79e8734ccc58f2da77d01f06fa2018af')

    def test_report_and_receipt_are_verified_and_persisted_without_labels(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);w=p/'work';cache=p/'cache';cache.mkdir();w.mkdir()
            for name in ['FINAL_AUDIT.json','ACCESS.jsonl','STUDY_FREEZE.json','CONFIRMATION_COMPLETE.json','RUN_FREEZE.json','DATA_FREEZE.json','ENGINEERING_GATE.json','E01_ENGINEERING_GATE.json','PERFORMANCE_PREFLIGHT.json','scores/SCORE_SEAL.json']:
                write(w/name,{'status':'PASS','fictional_only':True})
            (w/'ACCESS.jsonl').write_text('{}\n');(w/'EXECUTION_PROGRESS.jsonl').write_text('{}\n')
            write(w/'reports/table.json',{'fictional_only':True});seal(w/'reports','COMPLETE.json')
            transport=LocalTransport(p/'drive');recovery=types.SimpleNamespace(transport=transport,latest=lambda:({'name':'fictional_verified_recovery'},{}))
            runner=types.SimpleNamespace(work=w,cache=cache,recovery=recovery,device='cpu',interval=600,
                completed=lambda:22,backup=Mock(),verify_score_seal=Mock(),log_record=Mock(),transfer_progress=lambda *a:None,measured_checkpoint_seconds=[.01])
            report,receipt=execution_reports.persist_reports(runner)
            self.assertTrue(report.is_file());self.assertEqual(sha(report),read(receipt)['sha256'])
            self.assertEqual(len([r for r in transport.list('reports') if r['properties']['status']=='complete']),1)
            self.assertEqual(len([r for r in transport.list('report_receipt') if r['properties']['status']=='complete']),1)
            with zipfile.ZipFile(report) as z:
                m=json.loads(z.read('REPORT_ARCHIVE_MANIFEST.json'))
                self.assertTrue(all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in m['files'].items()))
            runner.verify_score_seal.assert_called_once();runner.backup.assert_called_once()
            (w/'reports/table.json').write_text('tamper')
            with self.assertRaises(RuntimeError):execution_reports.persist_reports(runner)
            self.assertEqual(len(transport.list('reports')),1)

if __name__=='__main__':
    suite=unittest.TestLoader().loadTestsFromTestCase(ReleaseTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
