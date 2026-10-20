"""Small notebook launcher; only enforces deadlines, forwards stdout, reaps worker."""
import json, os, selectors, signal, subprocess, time
from pathlib import Path

def expired_operation(status, pid, now):
    if status.get('pid')!=pid:return None
    return next((v for v in status.get('operations',[]) if now-v['start_monotonic']>v['deadline']),None)

def launch(command, cache, total_timeout=None):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    status=cache/'operation.json'; status.unlink(missing_ok=True)
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
        bufsize=1,start_new_session=True,
        env={**os.environ,'PYTHONHASHSEED':'17','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONUNBUFFERED':'1'})
    selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
    start=time.monotonic();reason=None
    try:
        with (cache/'worker_stdout.log').open('a') as log:
            while process.poll() is None:
                for key,_ in selector.select(timeout=1):
                    line=key.fileobj.readline()
                    if line: print(line,end='',flush=True);log.write(line);log.flush()
                now=time.monotonic()
                if total_timeout and now-start>total_timeout:
                    reason='Bounded invocation exceeded '+str(total_timeout)+' seconds';break
                if status.exists():
                    try: current=json.loads(status.read_text())
                    except (OSError,json.JSONDecodeError):continue
                    expired=expired_operation(current,process.pid,now)
                    if expired:
                        reason='Operation deadline exceeded: '+expired['name']+' ('+str(expired['deadline'])+' seconds)';break
            if reason:
                print(reason+'. Stop; inspect the last operation and verified backup. No gate is granted.',flush=True)
            else:
                remainder=process.stdout.read()
                if remainder:print(remainder,end='',flush=True);log.write(remainder)
        if reason or process.returncode:
            raise RuntimeError(reason or ('Worker exited '+str(process.returncode)+'. Cause is not inferred from exit code alone.'))
    finally:
        selector.close()
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print('Worker could not be reaped. Stop this runtime; do not start another writer.',flush=True)
        process.stdout.close()
