"""CPU engineering tests only; production runner always requires CUDA."""
import contextlib,copy,io,json,tempfile,types,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import training_core,train_exp007,v2_data,v2_graph
from checks import model_checks
from common import *
from design import plans,decide
import run_exp007 as run

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'configs/exp007.json').read_text())
CHANNELS={r['channel']:r['mission'] for r in json.loads((ROOT/'provenance/EXP006_data_manifest.json').read_text())}
torch.set_num_threads(1)


class NoopLease:
    def check(self):pass


def synthetic_result(job):
    # Exercise the expected mixed mission selection: MSL fixed, SMAP dynamic.
    score={'fixed':1.,'static':.99,'dynamic':.95,'message_removed':1.02}[job['role']]
    if job['mission']=='MSL' and job['role']=='dynamic':score=.999
    return dict(**job,job_id=job_id(job),validation_mse=score if job['stage']!='robust' else None,
                assessment_mse=score if job['stage']=='robust' else None,
                gate_parameters=65 if job['gate_mode']=='dynamic' else (1 if job['gate_mode']=='static' else 0),
                parameters=100,best_epoch=2,epochs_ran=3,hit_epoch_ceiling=False,best_at_ceiling=False,
                elapsed_seconds=1.,gate_mean=1.,gate_std=.1)


class Engineering(unittest.TestCase):
    def test_model_identity_rf_gradients_and_removal(self):
        self.assertEqual(len(model_checks(torch.device('cpu'))),6)

    def test_frozen_source_functions(self):
        import ast
        original=ast.parse((ROOT/'provenance/EXP006_gpu_train.py').read_text())
        current=ast.parse((ROOT/'src/training_core.py').read_text())
        old={n.name:ast.dump(n,include_attributes=False) for n in original.body if isinstance(n,ast.FunctionDef)}
        for n in current.body:
            if isinstance(n,ast.FunctionDef):self.assertEqual(ast.dump(n,include_attributes=False),old[n.name])

    def test_training_roles_and_ablation(self):
        # Substitute device strings only inside temporary test modules, never on disk.
        core=types.ModuleType('cpu_test_core')
        text=(ROOT/'src/training_core.py').read_text().replace('torch.device("cuda")','torch.device("cpu")').replace('    torch.cuda.synchronize()','    pass')
        exec(compile(text,'cpu_test_core','exec'),core.__dict__)
        module=types.ModuleType('cpu_test_train')
        text=(ROOT/'src/train_exp007.py').read_text().replace('torch.device("cuda")','torch.device("cpu")').replace('.to("cuda")','.to("cpu")')
        exec(compile(text,'cpu_test_train','exec'),module.__dict__)
        module._train_with_early_stopping=core._train_with_early_stopping
        module._verify_initial_persistence=core._verify_initial_persistence
        module._load_robust_fold=core._load_robust_fold
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);old1,old2=v2_data.TRAIN_DIR,v2_graph.TRAIN_DIR
            try:
                v2_data.TRAIN_DIR=v2_graph.TRAIN_DIR=core.TRAIN_DIR=td
                cfg=copy.deepcopy(CONFIG);cfg['training_common']['max_epochs']=2
                rng=np.random.default_rng(9)
                for ch,width,mission in [('syntheticM',55,'MSL'),('syntheticS',25,'SMAP')]:
                    np.save(td/(ch+'.npy'),rng.normal(size=(400,width)).astype(np.float32))
                    cfg['baseline_by_mission'][mission]['hidden_by_mission'][mission]=8
                    for mode in ['fixed','static','dynamic']:
                        for stage in ['robust','canonical','ablation']:
                            job=dict(stage=stage,fold='R1' if stage=='robust' else 'canonical',mission=mission,
                                     channel=ch,seed=17,gate_mode=mode,role=mode,remove_messages=stage=='ablation')
                            model,result,trace=module.train_job(job,cfg)
                            self.assertTrue(np.isfinite(result['early_stop_mse']))
                            self.assertEqual(len(trace),40 if stage=='robust' else 80)
                            self.assertEqual(result['gate_parameters'],17 if mode=='dynamic' else (1 if mode=='static' else 0))
                            self.assertEqual(result['fit_end'],160 if stage=='robust' else 240)
                            self.assertEqual(result['stop_end'],200 if stage=='robust' else 320)
            finally:v2_data.TRAIN_DIR=old1;v2_graph.TRAIN_DIR=old2

    def test_gate_rules_and_no_post_ablation_selection(self):
        p=plans(CONFIG,CHANNELS);rows=[synthetic_result(j) for g in p.values() for j in g]
        selected,checks=decide(rows,CONFIG)
        self.assertEqual(selected['MSL']['gate_mode'],'fixed');self.assertEqual(selected['SMAP']['gate_mode'],'dynamic')
        # Static control can veto promotion even when the three original gates pass.
        for r in rows:
            if r['mission']=='SMAP' and r['role']=='static':
                r['validation_mse']=.90 if r['stage']=='canonical' else None
        selected,_=decide(rows,CONFIG);self.assertEqual(selected['SMAP']['gate_mode'],'fixed')
        with self.assertRaises(RuntimeError):decide(rows+[dict(stage='ablation')],CONFIG)
        with self.assertRaises(RuntimeError):decide(rows[:-1],CONFIG)

    def test_resume_identity_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);job=plans(CONFIG,CHANNELS)['robust'][0];archive=out/'partial.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for n,v in [('job.json',canonical(job)),('result.json',b'{}'),('history.json',b'[]'),('checkpoint.pt',b'x')]:z.writestr(n,v)
            self.assertIsNone(read_commit(out,job,'test'))
            commit_fit(out,job,'test',archive);self.assertEqual(read_commit(out,job,'test'),{})
            with self.assertRaises(RuntimeError):read_commit(out,job,'changed')
            (out/'fits'/job['stage']/(job_id(job)+'.zip')).write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError):read_commit(out,job,'test')
            freeze_json(out/'x.json',{'a':1})
            with self.assertRaises(RuntimeError):freeze_json(out/'x.json',{'a':2})

    def test_complete_orchestration_interruption_and_export(self):
        calls=[]
        def fake(job,config,out,rh,lease):
            calls.append(job_id(job));result=synthetic_result(job)
            with tempfile.TemporaryDirectory() as td:
                archive=Path(td)/'fit.zip'
                with zipfile.ZipFile(archive,'w') as z:
                    z.writestr('job.json',canonical(job));z.writestr('result.json',canonical(result))
                    z.writestr('history.json','[]');z.writestr('checkpoint.pt',b'engineering-only-placeholder')
                commit_fit(out,job,rh,archive)
            return result
        with tempfile.TemporaryDirectory() as td,patch.object(run,'execute',fake),contextlib.redirect_stdout(io.StringIO()):
            out=Path(td);p=plans(CONFIG,CHANNELS);self.assertEqual([len(p['robust']),len(p['canonical'])],[144,729])
            def interrupted(job,*args):
                if len(calls)==3:raise RuntimeError('simulated disconnect')
                return fake(job,*args)
            with patch.object(run,'execute',interrupted),self.assertRaisesRegex(RuntimeError,'simulated disconnect'):
                run.run_stage('robust',p['robust'],CONFIG,out,'test',NoopLease())
            rows=run.run_stage('robust',p['robust'],CONFIG,out,'test',NoopLease())
            rows+=run.run_stage('canonical',p['canonical'],CONFIG,out,'test',NoopLease())
            selected=run.write_selection(rows,CONFIG,out,'test')
            frozen=(out/'ARCHITECTURE_FREEZE.json').read_bytes()
            p=plans(CONFIG,CHANNELS,selected)
            self.assertEqual(len(p['ablation']),243)
            rows+=run.run_stage('ablation',p['ablation'],CONFIG,out,'test',NoopLease())
            run.finish(rows,CONFIG,selected,out,p)
            self.assertEqual((out/'ARCHITECTURE_FREEZE.json').read_bytes(),frozen)
            for stage in p:run.run_stage(stage,p[stage],CONFIG,out,'test',NoopLease())
            self.assertEqual(len(calls),1116);self.assertEqual(len(set(calls)),1116)
            self.assertEqual(len(json.loads((out/'final_checkpoint_index.json').read_text())),486)
            run.export_results(out)
            for name in ['EXP007_GPU_V1_summary.zip','EXP007_GPU_V1_results.zip']:
                with zipfile.ZipFile(out/'exports'/name) as z:self.assertIsNone(z.testzip())

    def test_data_and_graph_golden_records(self):
        from checks import preflight
        # Replace only the CUDA engineering part, preserving full real TRAIN audit.
        # Gold hashes and graph/scale dictionaries are checked separately here.
        golden=json.loads((ROOT/'provenance/EXP006_graph_manifest.json').read_text())
        for row in golden:
            ch=row['channel'];f=row['fit_fraction'];graph=v2_graph.build_channel_graph(ch,f)
            raw=np.load(v2_data.TRAIN_DIR/(ch+'.npy')).astype(np.float32)
            target=raw[:int(np.floor(len(raw)*f)),0];scale=float(target.std());scale=scale if scale>=1e-8 else 1.
            self.assertEqual(graph.feature_indices.tolist(),row['features'])
            self.assertEqual(graph.source_original_columns.tolist(),row['sources'])
            self.assertEqual(digest(graph.adjacency.tolist()),row['adjacency_sha256'])
            self.assertEqual(float(target.mean()),row['target_mean']);self.assertEqual(scale,row['target_scale'])
        self.assertEqual(len(golden),243)


if __name__=='__main__':unittest.main(verbosity=2)
