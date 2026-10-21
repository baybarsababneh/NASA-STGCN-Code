"""Engineering source parity only; no new model fitting or detector selection."""
import ast,csv,json,sys,time
from pathlib import Path
import numpy as np
from exp016.detector import detect
from exp016.io import save_json
root=Path(sys.argv[1]);csv_path=Path(sys.argv[2]);out=Path(sys.argv[3]);receipt=[];seen=set();t=time.perf_counter()
for row in csv.DictReader(open(csv_path)):
 ch=row['chan_id']
 if ch in seen:continue
 seen.add(ch);test=np.load(root/'test'/(ch+'.npy'));pred=np.load(root/'2018-05-19_15.00.10/y_hat'/(ch+'.npy'))
 got=detect(test,pred,ch,'native_C2');expected=sorted(tuple(v) for v in ast.literal_eval(row['tp_sequences'])+ast.literal_eval(row['fp_sequences']))
 assert [list(v) for v in got['intervals']]==[list(v) for v in expected],ch
 receipt.append(dict(channel=ch,intervals_exact=True,warnings=got['warnings']))
 print('Native archived-interval parity:',ch,flush=True)
save_json(out,dict(channels=len(receipt),seconds=time.perf_counter()-t,checks=receipt))
