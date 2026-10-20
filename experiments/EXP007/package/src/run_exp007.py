"""Unattended GPU-native gate experiment, freeze, then final matched ablation."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
import argparse,gc,json,math,shutil,subprocess,tempfile,time,traceback,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import *
from execution import configure_cuda,install_data_guard,Lease,clean_json,export_results
from checks import preflight
from design import plans,decide,paired_tables
from train_exp007 import train_job

ROOT=Path(__file__).resolve().parents[1]


def execute(job,config,out,run_hash,lease):
    lease.check();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    model,result,trace=train_job(job,config)
    torch.cuda.synchronize()
    metric=result['assessment_mse'] if job['stage']=='robust' else result['validation_mse']
    if not math.isfinite(metric):raise RuntimeError('Non-finite fit result')
    result.update(job_id=job_id(job),run_hash=run_hash,fit_wall_seconds=time.perf_counter()-start,
                  gpu_peak_memory_bytes=torch.cuda.max_memory_allocated())
    result=clean_json(result)
    checkpoint={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'job':job,'spec':config['baseline_by_mission'][job['mission']],
                'result':result,'config':config,'run_hash':run_hash,
                'gate_trace':trace,'gate_trace_targets_start':result['gate_diagnostic_target_start'],
                'model_class':'models_exp007.MessageGatedGCN','format':'EXP007_GPU_V1-best-checkpoint'}
    with tempfile.TemporaryDirectory(prefix='exp007_fit_') as td:
        td=Path(td);torch.save(checkpoint,td/'checkpoint.pt')
        with zipfile.ZipFile(td/'fit.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            z.write(td/'checkpoint.pt','checkpoint.pt')
            z.writestr('result.json',canonical(result));z.writestr('job.json',canonical(job))
            z.writestr('history.json',canonical(model.exp006_history))
        lease.check();commit_fit(out,job,run_hash,td/'fit.zip')
    del model,checkpoint,trace;gc.collect();torch.cuda.empty_cache()
    return result


def run_stage(stage,jobs,config,out,run_hash,lease):
    freeze_json(out/'plans'/f'{stage}.json',jobs)
    rows=[]
    for index,job in enumerate(jobs,1):
        lease.check()
        result=read_commit(out,job,run_hash)
        if result is None:
            print(f"[{stage} {index}/{len(jobs)}] {job['mission']} {job['channel']} {job['role']} seed={job['seed']} {job['fold']}",flush=True)
            result=execute(job,config,out,run_hash,lease)
        else:print(f"[{stage} {index}/{len(jobs)}] verified resume {job_id(job)}",flush=True)
        rows.append(result)
        atomic_bytes(out/'tables'/f'{stage}_runs.csv',pd.DataFrame(rows).to_csv(index=False).encode())
        write_json(out/'STATUS.json',dict(stage=stage,completed=len(rows),expected=len(jobs),run_hash=run_hash))
    if {r['job_id'] for r in rows}!={job_id(j) for j in jobs}:raise RuntimeError('Stage job mismatch')
    return rows


def write_selection(rows,config,out,run_hash):
    selected,checks=decide(rows,config)
    freeze_json(out/'ARCHITECTURE_FREEZE.json',dict(run_hash=run_hash,selected=selected,
        selection_job_ids=sorted(r['job_id'] for r in rows),
        ablation_results_not_used=True,no_additional_architecture_search=True))
    freeze_json(out/'selected_models.json',selected)
    atomic_bytes(out/'tables/promotion_checks.csv',pd.DataFrame(checks).to_csv(index=False).encode())
    print('ARCHITECTURE FROZEN BEFORE MESSAGE-REMOVAL CONTROL:\n'+json.dumps(selected,indent=2),flush=True)
    return selected


def finish(rows,config,selected,out,jobs):
    df=pd.DataFrame(rows)
    if len(df)!=1116 or df.job_id.nunique()!=1116:raise RuntimeError('Final inventory incomplete')
    atomic_bytes(out/'tables/all_runs.csv',df.to_csv(index=False).encode())
    c=df[df.stage!='robust']
    summary=c.groupby(['mission','stage','role','gate_mode'],as_index=False).agg(
        channels=('channel','nunique'),fits=('seed','size'),validation_mse_mean=('validation_mse','mean'),
        validation_mse_median=('validation_mse','median'),parameters_mean=('parameters','mean'),
        gate_parameters=('gate_parameters','mean'),epochs_mean=('epochs_ran','mean'),
        best_epoch_mean=('best_epoch','mean'),ceiling_hits=('hit_epoch_ceiling','sum'),
        best_at_ceiling=('best_at_ceiling','sum'),training_seconds=('elapsed_seconds','sum'),
        gate_mean=('gate_mean','mean'),within_channel_gate_std_mean=('gate_std','mean'))
    atomic_bytes(out/'tables/mission_summary.csv',summary.to_csv(index=False).encode())
    for keys,name in [(['mission','stage','role','seed'],'seed_summary'),
                      (['mission','stage','role','channel'],'channel_summary')]:
        table=c.groupby(keys,as_index=False).agg(validation_mse_mean=('validation_mse','mean'))
        atomic_bytes(out/f'tables/{name}.csv',table.to_csv(index=False).encode())
    robust=df[df.stage=='robust'].groupby(['mission','fold','role'],as_index=False).agg(
        assessment_mse_mean=('assessment_mse','mean'),fits=('seed','size'))
    atomic_bytes(out/'tables/robustness_summary.csv',robust.to_csv(index=False).encode())
    pairs,ci=paired_tables(rows,selected,config)
    atomic_bytes(out/'tables/paired_channel_effects.csv',pairs.to_csv(index=False).encode())
    atomic_bytes(out/'tables/paired_bootstrap.csv',ci.to_csv(index=False).encode())
    family=pairs.groupby(['mission','comparison','family'],as_index=False).agg(
        channels=('channel','size'),absolute_improvement_mean=('absolute_improvement','mean'),
        relative_improvement_median_pct=('relative_improvement_pct','median'))
    atomic_bytes(out/'tables/family_effects.csv',family.to_csv(index=False).encode())
    # Link existing canonical checkpoints, rather than refitting winners after selection.
    index=[]
    for job in jobs['canonical']+jobs['ablation']:
        if job['stage']=='canonical' and job['gate_mode']!=selected[job['mission']]['gate_mode']:continue
        stem=f"fits/{job['stage']}/{job_id(job)}"
        commit=json.loads((out/(stem+'.done.json')).read_text())
        index.append(dict(channel=job['channel'],mission=job['mission'],seed=job['seed'],
            role='final_gcn' if job['stage']=='canonical' else 'message_removed',
            gate_mode=job['gate_mode'],archive=stem+'.zip',sha256=commit['archive_sha256']))
    assert len(index)==486
    write_json(out/'final_checkpoint_index.json',index)
    lines=['# EXP007_GPU_V1 completed','',
        '1116/1116 fits verified. Architecture selection used TRAIN only and was frozen before the final message-removal control.','']
    for mission,decision in selected.items():
        lines.append(f"- {mission}: {decision['decision']}. Gate={decision['gate_mode']}, RF={1+2*sum(decision['spec']['dilations'])}, width={decision['spec']['hidden_by_mission'][mission]}. Mean validation MSE={decision['validation_mse']:.9g}.")
    lines+=['','The static scalar is a mechanism control. Only the prespecified dynamic gate is eligible for promotion.',
        'Message-removal and bootstrap results are explanatory diagnostics and cannot reopen architecture selection.',
        'Positive improvement in message_removed_vs_* means removal forecasted better. It does not establish an anomaly-detection result.',
        'Stop GCN architecture optimisation. Audit these outputs, then freeze a separate detector-development protocol before using calibration data.',
        'NASA TEST remains retrospective. No NASA TEST scores or labels were read.']
    atomic_bytes(out/'SUMMARY.md','\n'.join(lines).encode());print('\n'.join(lines),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output).resolve()
    if out==ROOT or ROOT in out.parents:raise RuntimeError('Input and output must be separate')
    out.mkdir(parents=True,exist_ok=True)
    with Lease(out) as lease:
        validate_package(ROOT);env=configure_cuda()
        config=json.loads((ROOT/'configs/exp007.json').read_text())
        freeze=dict(experiment=config['experiment_id'],environment=env,
                    package_manifest_sha256=sha(ROOT/'PACKAGE_MANIFEST.json'),
                    protocol_sha256=sha(ROOT/'PROTOCOL.md'))
        freeze_json(out/'RUN_FREEZE.json',freeze);run_hash=digest(freeze)
        freeze_json(out/'effective_config.json',config)
        write_json(out/'sessions'/(time.strftime('%Y%m%dT%H%M%S')+'.json'),
                   dict(run_hash=run_hash,gpu_unit=subprocess.check_output(
                       ['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv'],text=True)))
        for folder in ['src','configs','provenance']:
            shutil.copytree(ROOT/folder,out/'reproducibility'/folder,dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__'))
        for name in ['PACKAGE_MANIFEST.json','PROTOCOL.md']:
            shutil.copy2(ROOT/name,out/'reproducibility'/name)
        install_data_guard();channels=preflight(ROOT,out,config)
        jobs=plans(config,channels)
        rows=run_stage('robust',jobs['robust'],config,out,run_hash,lease)
        rows+=run_stage('canonical',jobs['canonical'],config,out,run_hash,lease)
        selected=write_selection(rows,config,out,run_hash)
        jobs=plans(config,channels,selected)
        rows+=run_stage('ablation',jobs['ablation'],config,out,run_hash,lease)
        finish(rows,config,selected,out,jobs)
        write_json(out/'STATUS.json',dict(status='COMPLETE',fits=1116,run_hash=run_hash))
        lease.check();export_results(out)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        raise SystemExit('Stopped safely. Earlier completed fit commits are preserved. Do not alter scientific settings to bypass an error.')
