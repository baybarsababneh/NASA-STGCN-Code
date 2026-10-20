"""Persist final scientific outputs without changing their frozen evaluator."""
import hashlib,json,os,time,uuid,zipfile
from pathlib import Path
from core import ROOT,sha,read,write,manifest,configure
from drive_transport import AMENDMENT

def persist_reports(runner):
    work=runner.work;transport=runner.recovery.transport
    runner.verify_score_seal()
    manifest(work/'reports','COMPLETE.json')
    if read(work/'FINAL_AUDIT.json').get('status')!='PASS':raise RuntimeError('Final scientific audit must pass before delivery')
    identity={'amendment':AMENDMENT,'package_manifest_sha256':sha(ROOT/'PACKAGE_MANIFEST.json'),
              'study_freeze_sha256':sha(work/'STUDY_FREEZE.json'),
              'score_seal_sha256':sha(work/'scores/SCORE_SEAL.json'),
              'report_seal_sha256':sha(work/'reports/COMPLETE.json'),
              'environment':configure(runner.device)}
    audit_path=work/'E01_FINAL_EXECUTION_AUDIT.json'
    if audit_path.exists():
        audit=read(audit_path)
        if audit.get('identity')!=identity or audit.get('status')!='PASS':raise RuntimeError('Final engineering report identity differs')
    else:
        audit={'status':'PASS','identity':identity,'created_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'planned_fits':22,'completed_job_manifests':runner.completed(),
            'actual_GPU_engineering_gate':read(work/'E01_ENGINEERING_GATE.json'),
            'measured_performance_preflight':read(work/'PERFORMANCE_PREFLIGHT.json'),
            'verified_recovery_receipt_before_final_packaging':runner.recovery.latest()[0],
            'automatic_Drive_backup_required':True,'live_work_on_local_filesystem':True,
            'checkpoint_target_compute_seconds':runner.interval,'hard_wall_clock_loss_bound_claimed':False,
            'backup_seconds_measured_this_session':runner.measured_checkpoint_seconds,
            'EXP012_retraining_or_replay':0,'new_NASA_TEST_tuning':False,
            'scientific_success_inferred_from_engineering_PASS':False}
        write(audit_path,audit)
    runner.backup('final_scientific_and_execution_audits',force=True)
    payload={p.relative_to(work).as_posix():p for p in (work/'reports').rglob('*') if p.is_file()}
    for name in ['FINAL_AUDIT.json','E01_FINAL_EXECUTION_AUDIT.json','EXECUTION_PROGRESS.jsonl',
                 'ACCESS.jsonl','STUDY_FREEZE.json','CONFIRMATION_COMPLETE.json','RUN_FREEZE.json',
                 'DATA_FREEZE.json','ENGINEERING_GATE.json','E01_ENGINEERING_GATE.json',
                 'PERFORMANCE_PREFLIGHT.json','scores/SCORE_SEAL.json']:
        payload[name]=work/name
    for name in ['PROTOCOL.md','PROTOCOL.json','LITERATURE_AND_DESIGN.md','BUILD_AUDIT.json',
                 'E01_ENGINEERING_AUDIT.md','E01_START_HERE.md','E01_AMENDMENT_MANIFEST.json','PACKAGE_MANIFEST.json']:
        payload['release/'+name]=ROOT/name
    hashes={n:sha(p) for n,p in payload.items()}
    report=runner.cache/(AMENDMENT+'_reports.zip')
    tmp=report.with_name(report.name+'.tmp-'+uuid.uuid4().hex)
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for n,p in sorted(payload.items()):z.write(p,n)
        z.writestr('REPORT_ARCHIVE_MANIFEST.json',json.dumps({'identity':identity,'files':hashes},sort_keys=True,indent=2))
    with zipfile.ZipFile(tmp) as z:
        if set(z.namelist())!={*hashes,'REPORT_ARCHIVE_MANIFEST.json'}:raise RuntimeError('Final report archive roster differs')
        for n,h in hashes.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise RuntimeError('Final report archive member differs: '+n)
    os.replace(tmp,report);expected=sha(report);size=report.stat().st_size
    matches=[r for r in transport.list('reports') if r.get('properties',{}).get('status')=='complete' and r.get('properties',{}).get('sha256')==expected]
    if matches:remote=transport.verify(matches[0]['id'],expected,size)
    else:
        remote=transport.upload(report,report.name,'reports',runner.transfer_progress)
        remote=transport.finalize(remote['id'],report.name)
    transport.verify(remote['id'],expected,size)
    receipt=report.with_suffix(report.suffix+'.sha.json')
    write(receipt,{'sha256':expected,'bytes':size,'identity':identity,'Drive_file':remote})
    receipt_sha=sha(receipt);receipt_size=receipt.stat().st_size
    stored=[r for r in transport.list('report_receipt') if r.get('properties',{}).get('status')=='complete' and r.get('properties',{}).get('sha256')==receipt_sha]
    if stored:transport.verify(stored[0]['id'],receipt_sha,receipt_size)
    else:
        uploaded=transport.upload(receipt,receipt.name,'report_receipt')
        receipt_remote=transport.finalize(uploaded['id'],receipt.name)
        transport.verify(receipt_remote['id'],receipt_sha,receipt_size)
    runner.log_record({'phase':'FINAL_REPORT_AND_RECOVERY_DURABLE','report_sha256':expected,
        'Drive_report_name':remote['name'],'recovery_commit_name':runner.recovery.latest()[0]['name'],
        'scientific_claim':'Retrospective exploratory NASA; no independent confirmation inferred.'})
    return report,receipt
