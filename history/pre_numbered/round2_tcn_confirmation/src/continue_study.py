"""Finish sequential selection/confirmation after the screening queue completes."""
import json,time,subprocess,sys
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[1]
def state(stage):
    (R/'results/study_state.json').write_text(json.dumps(dict(stage=stage,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2))
def run(script,*args):
    state(' '.join([script,*args]));subprocess.run([sys.executable,str(R/'src'/script),*args],check=True)
if __name__=='__main__':
    state('waiting_for_complete_screen')
    deadline=time.monotonic()+3600
    while True:
        try:
            if len(pd.read_csv(R/'results/screen_runs.csv'))==len(json.loads((R/'configs'/('screen_executed_jobs.json' if (R/'configs/screen_executed_jobs.json').exists() else 'screen_jobs.json')).read_text())):break
        except (OSError,pd.errors.EmptyDataError):pass
        if time.monotonic()>deadline:raise TimeoutError('Screen not complete within one hour')
        time.sleep(2)
    run('select_stages.py','screen')
    run('run_jobs.py','confirm_jobs.json','--workers','6')
    run('select_stages.py','confirm')
    run('run_jobs.py','robustness_jobs.json','--workers','6')
    run('analyze.py')
    state('complete')
