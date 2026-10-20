"""Copy report-ready assets from a verified final run, without scoring or fitting."""
import argparse,shutil
from pathlib import Path
from integrity import read,sha,checked,manifest,files,write

def export(work,destination,allow_fixture=False):
    work=Path(work);destination=Path(destination)
    w,d=work.resolve(),destination.resolve()
    if w==d or w in d.parents or d in w.parents:raise RuntimeError('Export into a separate directory')
    audit=read(work/'FINAL_AUDIT.json');report=manifest(work/'reports','COMPLETE.json')
    checked(work/'SCORE_SEAL.json',audit['score_seal_sha256']);checked(work/'reports/COMPLETE.json',audit['report_seal_sha256'])
    if read(work/'SCORE_SEAL.json')['files']!=files(work/'units'):raise RuntimeError('Sealed score bytes changed')
    if audit['fixture'] and not allow_fixture:raise RuntimeError('Synthetic fixture is not a NASA result; use --allow-fixture only for an explicitly marked test export')
    if not audit['fixture'] and not audit['A100_validation']:raise RuntimeError('Production A100 validation missing')
    if destination.exists() and any(destination.iterdir()):raise RuntimeError('Export destination must be empty')
    destination.mkdir(parents=True,exist_ok=True)
    selected=['REPORT_READY_SUMMARY.md','SUMMARY.json','mission_metrics.csv','mission_seed_metrics.csv','common_detector_channels.csv','complete_pipeline_channels.csv','common_detector_event_notifications.csv','native_pipeline_event_notifications_frozen.csv','paired_channel_bootstrap.json','table_rows.tex','NASA_comparisons.png','NASA_comparisons.pdf','LABEL_ROSTER_AUDIT.json']
    for n in selected:
        shutil.copyfile(work/'reports'/n,destination/n);checked(destination/n,report['files'][n])
    write(destination/'EXPORT.json',{'status':audit['status'],'fixture':audit['fixture'],'report_seal_sha256':audit['report_seal_sha256'],'score_seal_sha256':audit['score_seal_sha256'],'files':files(destination),'new_scientific_computation':False})
    return destination

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--destination',type=Path,required=True);ap.add_argument('--allow-fixture',action='store_true');a=ap.parse_args()
    print(export(a.work,a.destination,a.allow_fixture))
