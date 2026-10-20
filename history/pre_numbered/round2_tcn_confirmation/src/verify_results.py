"""Verify saved validation predictions, boundaries, manifests and checkpoint loads."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from round2_train import ROOT,prepare,predict
from round2_models import make

def main():
    d=pd.read_csv(ROOT/'results/all_runs.csv');assert not d.duplicated(['configuration','channel','seed']).any();checked=0
    for r in d.itertuples():
        p=ROOT/'checkpoints'/r.configuration/f'{r.channel}_s{r.seed}';meta=json.loads(p.with_suffix('.json').read_text())['metadata'];v=np.load(p.with_suffix('.npz'))
        assert np.isfinite(v['prediction']).all()
        assert v['times'].min()==meta['fit_end'] and v['times'].max()==meta['validation_end']-1
        assert not meta['calibration_accessed']
        np.testing.assert_allclose(np.mean((v['prediction']-v['truth'])**2),r.val_mse,rtol=2e-5,atol=2e-7)
        checked+=1
    lock=json.loads((ROOT/'configs/locked_tcn.json').read_text());load_checks=0
    for mission,cfg in lock['missions'].items():
        channels=d[(d.configuration==cfg['name'])&(d.seed==17)].sort_values('val_mse',ascending=False).channel
        for seed in lock['seeds']:
            ch=channels.iloc[0];p=ROOT/'checkpoints'/cfg['name']/f'{ch}_s{seed}.pt';saved=torch.load(p,map_location='cpu',weights_only=False);model=make(cfg,len(saved['metadata']['active']),saved['adjacency']);model.load_state_dict(saved['state'],strict=True)
            _,X,y,t,meta=prepare(ch,cfg['lookback']);ix=np.flatnonzero(t>=meta['fit_end'])[:8];expected=np.load(p.with_suffix('.npz'))['prediction'][:8]
            np.testing.assert_allclose(predict(model,X[ix]),expected,rtol=2e-5,atol=2e-6);load_checks+=1
    summary=dict(validation_prediction_files_verified=checked,locked_checkpoint_reload_checks=load_checks,test_accessed=False,calibration_accessed=False)
    (ROOT/'results/final_verification.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
