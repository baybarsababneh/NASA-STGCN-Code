"""Offline engineering tests. CPU substitutions exist only in this test process.

These tests do not produce research fits or authorize CPU use in production.
Run: python src/test_engineering.py
"""
import copy
import importlib.util
import json
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
from gpu_common import (commit_fit, digest, freeze_json, job_id, promotion,
                        read_commit, screen_selection)
from gpu_run import plans
import v2_data
import v2_graph
from v2_models_exp006 import TCNGCNFixedTuned

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/v2_exp006_gcn_tuning.json").read_text())
torch.set_num_threads(1)


def module_text(name, text):
    module = types.ModuleType(name)
    exec(compile(text, name, "exec"), module.__dict__)
    return module


class EngineeringTests(unittest.TestCase):
    def test_grid_and_full_plan(self):
        channels = {c: v2_data.infer_mission(np.load(v2_data.TRAIN_DIR / (c + ".npy")))
                    for c in v2_data.list_channels()}
        # Deliberately not the CPU shortlist: GPU selection must be unconstrained.
        selected = {m: [CFG["candidates"][0]["candidate_id"], CFG["candidates"][-1]["candidate_id"]]
                    for m in channels.values()}
        jobs = plans(CFG, channels, selected)
        self.assertEqual([len(jobs[k]) for k in ("screen", "robust", "reference", "confirm")], [288, 192, 243, 486])
        self.assertEqual(len({job_id(j) for v in jobs.values() for j in v}), 1209)

    def test_exact_tie_ranking(self):
        rows = []
        for m in ("MSL", "SMAP"):
            for cid, params in [("z", 100), ("b", 50), ("a", 50)]:
                for seed in (17, 29):
                    rows.append(dict(mission=m, candidate_id=cid, seed=seed,
                                     validation_mse=1.0, parameters=params))
        _, selected = screen_selection(rows, CFG)
        self.assertEqual(selected, {"MSL": ["a", "b"], "SMAP": ["a", "b"]})

    def test_promotion_all_three_gates(self):
        def rows_for(candidate, values, robust):
            rows = [dict(stage="confirm", role="gcn_candidate", mission="MSL", candidate_id=candidate,
                         seed=s, validation_mse=v, fold="canonical") for s, v in zip((17,29,43),values)]
            rows += [dict(stage="robust", role="gcn_candidate", mission="MSL", candidate_id=candidate,
                          seed=17, assessment_mse=v, fold=f) for f,v in zip(("R1","R2"),robust)]
            return rows
        ref = [dict(stage="reference", role="exp002_reference", mission="MSL", candidate_id="exp002_reference",
                    seed=s, validation_mse=1., fold="canonical") for s in (17,29,43)]
        ref += [dict(stage="robust", role="exp002_reference", mission="MSL", candidate_id="exp002_reference",
                     seed=17, assessment_mse=1., fold=f) for f in ("R1","R2")]
        rows = ref + rows_for("a", [.9,.9,.9], [.8,1.1]) + rows_for("b", [.8,.8,1.01], [.7,.7])
        checks, decisions = promotion(rows,{"MSL":["a","b"]},CFG)
        self.assertEqual(decisions["MSL"]["selected_candidate"],"a")
        self.assertFalse(checks[1]["passes_each_seed"])
        rows = ref + rows_for("a", [.99,.99,.99], [.9,.9]) + rows_for("b", [.8,.8,.8], [1.2,1.2])
        _, decisions = promotion(rows,{"MSL":["a","b"]},CFG)
        self.assertEqual(decisions["MSL"]["decision"],"retain_exp002_architecture")

    def test_commit_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); job={"stage":"screen","seed":17}; run="frozen"
            source=td/"fit.zip"
            with zipfile.ZipFile(source,"w") as z:
                for name,data in [("job.json",json.dumps(job)),("result.json",'{"validation_mse": 1}'),
                                  ("history.json","[]"),("checkpoint.pt","synthetic")]: z.writestr(name,data)
            self.assertIsNone(read_commit(td,job,run))
            # An archive without the final commit marker is intentionally incomplete.
            partial=td/"fits/screen"/(job_id(job)+".zip");partial.parent.mkdir(parents=True);partial.write_bytes(source.read_bytes())
            self.assertIsNone(read_commit(td,job,run))
            commit_fit(td,job,run,source)
            self.assertEqual(read_commit(td,job,run)["validation_mse"],1)
            with self.assertRaises(RuntimeError):read_commit(td,job,"changed")
            partial.write_bytes(b"corrupt")
            with self.assertRaises(RuntimeError):read_commit(td,job,run)

    def test_frozen_identity(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"freeze.json";freeze_json(p,{"a":1});freeze_json(p,{"a":1})
            with self.assertRaises(RuntimeError):freeze_json(p,{"a":2})

    def test_windows_are_causal_and_oriented(self):
        raw=np.arange(120*3,dtype=np.float32).reshape(120,3)
        x,y,p=v2_data._make_windows(raw,np.array([64,65,90]),64)
        self.assertEqual(x.shape,(3,3,64))
        np.testing.assert_array_equal(x[0],raw[:64].T)
        np.testing.assert_array_equal(x[2],raw[26:90].T)
        np.testing.assert_array_equal(y,raw[[64,65,90],0])
        np.testing.assert_array_equal(p,raw[[63,64,89],0])

    def test_fit_only_graph_invariance(self):
        rng=np.random.default_rng(8)
        raw=rng.normal(size=(200,55)).astype(np.float32);altered=raw.copy();altered[100:]*=100
        for mode in ("star","top4"):
            f1=v2_graph.select_varying_features(raw,100);f2=v2_graph.select_varying_features(altered,100)
            np.testing.assert_array_equal(f1,f2)
            a1,_=v2_graph.build_fixed_adjacency(raw[:100,f1],mode)
            a2,_=v2_graph.build_fixed_adjacency(altered[:100,f2],mode)
            np.testing.assert_array_equal(a1,a2)

    def test_training_loop_preserved_on_same_backend(self):
        frozen=module_text("frozen_trainer",(ROOT/"provenance/frozen_cpu/src/v2_train_exp006.py").read_text())
        # CPU-only mechanical substitution is confined to this engineering test.
        text=(ROOT/"src/gpu_train.py").read_text().replace('torch.device("cuda")','torch.device("cpu")')
        text=text.replace('    torch.cuda.synchronize()','    pass # CPU test substitution')
        current=module_text("current_trainer_cpu_test",text)
        rng=np.random.default_rng(9);x=rng.normal(size=(32,2,64)).astype(np.float32);y=rng.normal(size=32).astype(np.float32)
        common=dict(CFG["training_common"],batch_size=8,max_epochs=6)
        outputs=[]
        for mod in (frozen,current):
            mod.set_seed(17)
            model=TCNGCNFixedTuned(2,8,np.array([[0,1],[0,0]],dtype=np.float32))
            outputs.append(mod._train_with_early_stopping(model,x[:24],y[:24],x[24:],y[24:],17,.001,common))
        self.assertEqual(outputs[0][1:4],outputs[1][1:4])
        for k,v in outputs[0][0].state_dict().items():self.assertTrue(torch.equal(v,outputs[1][0].state_dict()[k]))

    def test_all_training_roles_synthetic(self):
        text=(ROOT/"src/gpu_train.py").read_text().replace('torch.device("cuda")','torch.device("cpu")')
        text=text.replace('    torch.cuda.synchronize()','    pass # CPU test substitution')
        current=module_text("role_smoke_cpu_test",text)
        with tempfile.TemporaryDirectory() as td:
            train=Path(td);rng=np.random.default_rng(13)
            old_data,old_graph=v2_data.TRAIN_DIR,v2_graph.TRAIN_DIR
            try:
                v2_data.TRAIN_DIR=v2_graph.TRAIN_DIR=current.TRAIN_DIR=train
                config=copy.deepcopy(CFG)
                for c in config["candidates"]:c["hidden_by_mission"]={"MSL":8,"SMAP":8}
                config["references"]["exp002"]["hidden_by_mission"]={"MSL":8,"SMAP":8}
                for c in config["references"]["exp005_selected_by_mission"].values():
                    c["hidden_by_mission"]={"MSL":8,"SMAP":8}
                for ch,width in [("syntheticMSL",55),("syntheticSMAP",25)]:
                    a=rng.normal(size=(400,width)).astype(np.float32);np.save(train/(ch+".npy"),a)
                    for role in ("gcn_candidate","exp002_reference"):
                        model,result=current.train_standard(ch,17,config,config["candidates"][0],smoke_epochs=2,role=role)
                        self.assertTrue(np.isfinite(result["validation_mse"]))
                        self.assertEqual(result["fit_end"],240)
                        self.assertEqual(result["stop_end"],320)
                        self.assertEqual(len(model.exp006_history),2)
                    for role in ("gcn_candidate","exp002_reference","exp005_reference"):
                        model,result=current.train_robustness(ch,17,config,role,"R1",config["candidates"][0],smoke_epochs=2)
                        self.assertTrue(np.isfinite(result["assessment_mse"]))
                        self.assertEqual((result["fit_end"],result["stop_end"],result["assess_end"]),(160,200,240))
                        self.assertEqual(len(model.exp006_history),2)
            finally:
                v2_data.TRAIN_DIR=old_data;v2_graph.TRAIN_DIR=old_graph

    def test_all_train_arrays_and_graphs(self):
        counts={"MSL":0,"SMAP":0}
        for ch in v2_data.list_channels():
            a=np.load(v2_data.TRAIN_DIR/(ch+".npy"),allow_pickle=False)
            self.assertTrue(np.isfinite(a).all());counts[v2_data.infer_mission(a)]+=1
            for frac in (.4,.5,.6):
                graph=v2_graph.build_channel_graph(ch,frac)
                self.assertEqual(graph.feature_indices[0],0)
                self.assertTrue(np.all(graph.adjacency[1:]==0))
                self.assertEqual(np.count_nonzero(graph.adjacency),graph.num_edges)
                if graph.num_edges:self.assertAlmostEqual(float(graph.adjacency.sum()),1,places=6)
        self.assertEqual(counts,{"MSL":27,"SMAP":54})


if __name__ == "__main__":
    unittest.main(verbosity=2)
