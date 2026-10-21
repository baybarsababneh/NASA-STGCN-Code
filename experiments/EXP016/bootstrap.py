"""Create an isolated worker environment; no notebook kernel restart needed."""
import argparse,json,os,subprocess,sys,venv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--venv',type=Path,required=True);p.add_argument('--cpu-fixture',action='store_true');a=p.parse_args()
if sys.version_info[:2] not in ((3,10),(3,11),(3,12)):
    raise RuntimeError('EXP016 pins PyTorch 2.5.1 and supports Python 3.10–3.12. Select a Colab runtime with supported Python. Do not silently change dependencies.')
root=Path(__file__).resolve().parent;py=a.venv/'bin/python';marker=a.venv/'EXP016_ENV.json'
expected={'torch':'2.5.1+cpu' if a.cpu_fixture else '2.5.1+cu121','numpy':'1.26.4','pandas':'2.2.3','h5py':'3.12.1','more_itertools':'10.5.0','pytest':'8.3.4','threadpoolctl':'3.5.0'}
env={**os.environ,'PYTHONNOUSERSITE':'1','PIP_DISABLE_PIP_VERSION_CHECK':'1'}
probe='import json,torch,numpy,pandas,h5py,more_itertools,pytest,threadpoolctl; print(json.dumps({m.__name__:m.__version__ for m in [torch,numpy,pandas,h5py,more_itertools,pytest,threadpoolctl]}))'
ready=False
if py.exists():
    r=subprocess.run([str(py),'-c',probe],env=env,text=True,capture_output=True)
    if r.returncode==0:ready=json.loads(r.stdout)==expected
if not ready:
    venv.EnvBuilder(with_pip=True,clear=True).create(a.venv)
    subprocess.run([str(py),'-m','pip','install','-r',str(root/'requirements.txt')],env=env,check=True)
    index='https://download.pytorch.org/whl/'+('cpu' if a.cpu_fixture else 'cu121')
    subprocess.run([str(py),'-m','pip','install','torch==2.5.1','--index-url',index],env=env,check=True)
actual=json.loads(subprocess.check_output([str(py),'-c',probe],env=env,text=True))
if actual!=expected:raise RuntimeError('Dependency verification failed: '+str(actual))
marker.write_text(json.dumps(actual,indent=2))
print('Verified isolated EXP016 worker:',py)
