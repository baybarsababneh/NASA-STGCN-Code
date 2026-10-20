"""Isolated pinned install, tested with CPU wheels; production uses cu126 wheels."""
import os,sys,time,json,subprocess,argparse
from pathlib import Path
os.environ['MPLBACKEND']='Agg'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--venv',type=Path,required=True)
    ap.add_argument('--cpu-fixture',action='store_true',help='Development installer verification only; never a production device fallback')
    a=ap.parse_args();package=a.package.resolve();venv=a.venv.resolve();beg=time.monotonic()
    if sys.version_info[:2] not in [(3,12),(3,13)]:raise RuntimeError('Python 3.12 or 3.13 required')
    env=dict(os.environ,MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',UV_NATIVE_TLS='true',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(package/'src')+os.pathsep+str(package))
    subprocess.run([sys.executable,'-m','pip','install','--quiet','uv==0.12.23'],check=True,env=env)
    if not (venv/'bin/python').exists():subprocess.run([sys.executable,'-m','uv','venv','--python',sys.executable,'--seed',str(venv)],check=True,env=env)
    py=str(venv/'bin/python');index='https://download.pytorch.org/whl/cpu' if a.cpu_fixture else 'https://download.pytorch.org/whl/cu126'
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',py,'torch==2.6.0','--index-url',index],check=True,env=env)
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',py,'-r',str(package/'requirements-colab.txt')],check=True,env=env)
    probe="from integrity import configure; import json; print(json.dumps(configure('cpu' if " + repr(a.cpu_fixture) + " else 'cuda'," + repr(a.cpu_fixture) + "),sort_keys=True))"
    result=subprocess.run([py,'-c',probe],check=True,env=env,text=True,capture_output=True)
    receipt={'status':'PINNED_INSTALL_AND_RUNTIME_PROBE_PASS','cpu_fixture':a.cpu_fixture,
             'actual_A100_validation':not a.cpu_fixture,'environment':json.loads(result.stdout),
             'seconds':time.monotonic()-beg,'uv_version':'0.12.23','wheel_index':index,'MPLBACKEND':'Agg'}
    print(json.dumps(receipt,sort_keys=True),flush=True)
if __name__=='__main__':main()
