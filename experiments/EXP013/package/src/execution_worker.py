"""Unbuffered, hash-seed-pinned worker owned by the current Colab kernel."""
import argparse,os,sys
from pathlib import Path
from process_guard import arm_parent_death_guard

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['startup','engineering','preflight','develop-and-seal','verify-seal','evaluate','finish'])
    parser.add_argument('--work',type=Path,required=True);parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--folder-id',required=True);parser.add_argument('--parent-pid',type=int,required=True)
    parser.add_argument('--labels',type=Path)
    args=parser.parse_args()
    arm_parent_death_guard(args.parent_pid)  # Before imports that allocate GPU memory.
    if os.environ.get('PYTHONHASHSEED')!='17':raise RuntimeError('Original PYTHONHASHSEED=17 is required')
    from drive_transport import DriveTransport
    from execution import Runner
    transport=DriveTransport.from_existing_authorization(args.folder_id)
    runner=Runner(args.work,args.cache,transport,device='cuda',interval_seconds=600)
    with runner.active():
        if args.action=='startup':runner.startup()
        else:
            import pipeline
            pipeline.verify_run(runner.work,'cuda')
            if args.action=='engineering':runner.engineering()
            elif args.action=='preflight':runner.preflight()
            elif args.action=='develop-and-seal':runner.develop_and_seal()
            elif args.action=='verify-seal':runner.verify_score_seal();print('VERIFIED SCORE SEAL: NASA label request is now permitted.',flush=True)
            elif args.action=='evaluate':
                if args.labels is None and not (runner.work/'reports/COMPLETE.json').exists():raise RuntimeError('Post-seal annotation CSV required')
                runner.evaluate_and_finish(args.labels)
            elif args.action=='finish':
                from execution_reports import persist_reports
                paths=persist_reports(runner)
                print('FINAL_DOWNLOAD_FILES',*[str(p) for p in paths],flush=True)

if __name__=='__main__':main()
