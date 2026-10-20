import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
from round2_train import ROOT,train_job

def main():
    p=argparse.ArgumentParser();p.add_argument('jobs');p.add_argument('--workers',type=int,default=3);p.add_argument('--output');args=p.parse_args()
    jobs=json.loads((ROOT/'configs'/args.jobs).read_text());csv=ROOT/'results'/(args.output or args.jobs.replace('_jobs.json','_runs.csv'))
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(train_job,job):job for job in jobs}
        for fut in as_completed(futures):
            r=fut.result();records.append(r)
            temporary=csv.with_suffix('.csv.tmp');pd.DataFrame(records).to_csv(temporary,index=False);temporary.replace(csv)
            print(f'{len(records)}/{len(jobs)} {r["configuration"]} {r["channel"]} s{r["seed"]}: mse={r["val_mse"]:.6g}, epoch={r["best_epoch"]}/{r["epochs"]}, {r["seconds"]:.1f}s',flush=True)

if __name__=='__main__':main()
