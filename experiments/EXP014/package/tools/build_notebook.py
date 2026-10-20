"""Build the paired notebook after the immutable input bundle is finalised."""
import argparse,hashlib,json,textwrap
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def md(s):return {'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip().splitlines(keepends=True)}
def code(s):return {'cell_type':'code','metadata':{},'source':textwrap.dedent(s).strip().splitlines(keepends=True),'execution_count':None,'outputs':[]}

def build(bundle,manifest,output):
    cells=[md('''
    # EXP014 · final bounded NASA literature comparison

    **Select an A100 40 GB runtime, then choose Runtime → Run all.** Mount Drive and upload the matching input bundle once. On reconnect, use the same run slot and Run all again.

    This notebook performs **no new training**. It adds the exact frozen EXP012 Hundman detector to telemetry-target forecasts from two verified EXP013 GDN checkpoints. Frozen GCN/LSTM and native GDN/CATCH results remain unchanged. Common-detector evidence and complete-pipeline adaptations are reported separately.

    **Required original input:** the completed EXP013 `FINAL_RESULTS.zip` in Drive, normally at `MyDrive/EXP013_COMMAND_REFERENCE_V1_E03_CONV/main/FINAL_RESULTS.zip`. The E03 input bundle and report ZIP are insufficient. If necessary, edit `SOURCE_PATH` below. A smaller ZIP/unpacked directory with the exact 252 constituents is accepted; the bundle's input inventory lists them.

    Development tests were CPU-only. This notebook must establish actual A100 memory feasibility, exact checkpoint restoration and bitwise frozen native-score replay. A failed gate stops the comparison; missing weights are never substituted.
    '''),code('''
    # 1. Set up Drive, verify the paired bundle, and identify the runtime.
    from google.colab import drive, files
    from pathlib import Path
    import sys, os, json, hashlib, zipfile, shutil, subprocess, uuid

    RUN_SLOT = 'main'
    SOURCE_PATH = Path('/content/drive/MyDrive/EXP013_COMMAND_REFERENCE_V1_E03_CONV/main/FINAL_RESULTS.zip')
    VERSION = 'EXP014_LITERATURE_COMPARISON_V1'
    BUNDLE_NAME = 'EXP014_LITERATURE_COMPARISON_V1_input_bundle.zip'
    BUNDLE_SHA256 = '__BUNDLE_SHA256__'
    MANIFEST_SHA256 = '__MANIFEST_SHA256__'
    if not RUN_SLOT or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in RUN_SLOT):
        raise ValueError('Use a simple letters/numbers run slot.')
    if sys.version_info[:2] not in [(3,12),(3,13)]:
        raise RuntimeError('Python 3.12 or 3.13 is required; this notebook does not migrate the scientific code.')
    print('Notebook Python:',sys.version)
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True).strip()
    print('Detected GPU (name, MiB, driver):',gpu)
    fields = gpu.splitlines()[0].split(',')
    if 'A100' not in fields[0] or not 35000 <= int(fields[1].strip()) <= 45000:
        raise RuntimeError('Select an A100 40 GB Colab runtime. No model/batch-size substitution is permitted.')
    drive.mount('/content/drive')
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f'Provide the completed immutable EXP013 archive at {SOURCE_PATH}, or edit SOURCE_PATH to its actual location. No new training will replace it.')
    DEST = Path('/content/drive/MyDrive')/VERSION/RUN_SLOT
    PACKAGE = Path('/content')/(VERSION+'_package')
    WORK = Path('/content')/(VERSION+'_work_'+RUN_SLOT)
    CACHE = Path('/content')/(VERSION+'_cache_'+RUN_SLOT)
    VENV = Path('/content')/(VERSION+'_venv')
    DEST.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)

    def sha_file(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
        return h.hexdigest()

    def persist_verified(src,dst):
        expected=sha_file(src);dst=Path(dst)
        if dst.exists():
            if sha_file(dst)!=expected:raise RuntimeError('Existing persistent file differs: '+str(dst))
            return
        dst.parent.mkdir(parents=True,exist_ok=True)
        tmp=dst.with_name(dst.name+'.partial-'+uuid.uuid4().hex)
        try:
            with Path(src).open('rb') as f,tmp.open('wb') as g:
                shutil.copyfileobj(f,g,8*2**20);g.flush();os.fsync(g.fileno())
            if sha_file(tmp)!=expected:raise RuntimeError('Drive copy readback differs')
            os.replace(tmp,dst)
            if sha_file(dst)!=expected:raise RuntimeError('Drive publication readback differs')
        finally:tmp.unlink(missing_ok=True)

    local_bundle=CACHE/BUNDLE_NAME; saved_bundle=DEST/BUNDLE_NAME
    if saved_bundle.exists():
        print('Loading the verified cached bundle from Drive.')
        shutil.copyfile(saved_bundle,local_bundle)
    else:
        print('Upload the matching EXP014 input bundle ZIP (one file).')
        uploaded=files.upload()
        if len(uploaded)!=1:raise RuntimeError('Upload exactly one matching input bundle.')
        content=next(iter(uploaded.values()))
        if hashlib.sha256(content).hexdigest()!=BUNDLE_SHA256:raise RuntimeError('Uploaded bundle is not the notebook-matched release.')
        local_bundle.write_bytes(content)
        del uploaded,content
    if sha_file(local_bundle)!=BUNDLE_SHA256:raise RuntimeError('Bundle hash differs; use the paired notebook and bundle.')
    persist_verified(local_bundle,saved_bundle)
    # The package is a disposable local copy. Scientific work/cache are separate.
    if PACKAGE.exists():shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    with zipfile.ZipFile(local_bundle) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise RuntimeError('Duplicate bundle archive paths')
        for item in z.infolist():
            p=Path(item.filename)
            if p.is_absolute() or '..' in p.parts or chr(92) in item.filename or (item.external_attr>>16)&0o170000==0o120000:
                raise RuntimeError('Unsafe bundle archive path')
        z.extractall(PACKAGE)
    if sha_file(PACKAGE/'MANIFEST.json')!=MANIFEST_SHA256:raise RuntimeError('Pinned code/input manifest differs')
    manifest=json.loads((PACKAGE/'MANIFEST.json').read_text())['files']
    actual={p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    if actual!=set(manifest):raise RuntimeError('Bundle file roster differs')
    for n,h in manifest.items():
        if sha_file(PACKAGE/n)!=h:raise RuntimeError('Bundle file hash differs: '+n)
    contract=json.loads((PACKAGE/'INPUT_CONTRACT.json').read_text())
    needed=set(contract['known_required_files'])
    if SOURCE_PATH.is_file():
        with zipfile.ZipFile(SOURCE_PATH) as z:
            missing=needed-set(z.namelist())
    else:missing={n for n in needed if not (SOURCE_PATH/n).is_file()}
    if missing:raise FileNotFoundError('This source is missing required EXP013 files: '+', '.join(sorted(missing)))
    print('Paired bundle verified. Original inputs remain read-only. New outputs:',DEST)
    '''),code('''
    # 2. Install the frozen numerical environment in an isolated local venv.
    # Each subsequent stage uses a fresh process, so the notebook kernel need not restart.
    subprocess.run([sys.executable,'-m','pip','install','--quiet','uv==0.12.23'],check=True)
    if not (VENV/'bin/python').exists():
        subprocess.run([sys.executable,'-m','uv','venv','--python',sys.executable,'--seed',str(VENV)],check=True)
    PY=str(VENV/'bin/python')
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',PY,'torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu126'],check=True)
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',PY,'-r',str(PACKAGE/'requirements-colab.txt')],check=True)
    ENV=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(PACKAGE/'src')+os.pathsep+str(PACKAGE))
    subprocess.run([PY,'-c',"from integrity import configure; import json; print('PINNED RUNTIME',json.dumps(configure('cuda'),sort_keys=True))"],check=True,env=ENV)
    '''),code('''
    # 3. Run one bounded CPU regression. It never produces NASA/GPU results.
    test_report=CACHE/'NOTEBOOK_CPU_REGRESSION.json'
    test_log=CACHE/'NOTEBOOK_CPU_REGRESSION.log'
    result=subprocess.run([PY,'-u',str(PACKAGE/'tests/run_tests.py'),'--report',str(test_report),'--log',str(test_log)],env=ENV)
    if result.returncode:
        print(test_log.read_text()[-12000:] if test_log.exists() else 'Test process did not produce a log.')
        raise RuntimeError('Regression failed. Scientific inference has not started.')
    receipt=json.loads(test_report.read_text())
    if receipt['status']!='PASS_CPU_REGRESSION':raise RuntimeError('Regression receipt is not a PASS.')
    print('CPU tests:',receipt['tests_run'],'in',round(receipt['seconds'],2),'seconds. Actual A100 validation remains pending.')
    persist_verified(test_report,DEST/'engineering'/('CPU_REGRESSION_'+sha_file(test_report)[:16]+'.json'))
    persist_verified(test_log,DEST/'engineering'/('CPU_REGRESSION_'+sha_file(test_log)[:16]+'.log'))
    '''),code('''
    # 4. Execute or resume the scientific comparison. Progress is streamed live.
    # Real native full-batch memory/restore gates and native reference replay are mandatory.
    subprocess.run([PY,'-u',str(PACKAGE/'src/run.py'),
                    '--source',str(SOURCE_PATH),'--work',str(WORK),
                    '--backup',str(DEST),'--cache',str(CACHE)],check=True,env=ENV)
    '''),code('''
    # 5. Display the verified final summary and identify the durable archive.
    from IPython.display import display, Markdown, Image
    latest=json.loads((DEST/'LATEST_EXPORT.json').read_text())
    final_archive=DEST/latest['archive']
    if sha_file(final_archive)!=latest['sha256']:raise RuntimeError('Final Drive archive verification failed')
    display(Markdown((WORK/'reports/REPORT_READY_SUMMARY.md').read_text()))
    display(Image(filename=str(WORK/'reports/NASA_comparisons.png')))
    print('VERIFIED FINAL ARCHIVE:',final_archive)
    print('SHA-256:',latest['sha256'])
    print('Supply this archive to the report-writing task. EXP012 and EXP013 conclusions remain immutable.')
    '''),md('''
    ## Recovery and interpretation

    After a disconnect, select A100 40 GB and **Run all** again with the same run slot and unchanged source/bundle. Verified backups restore committed channels automatically; already validated work is skipped. A channel not durably backed up may be recomputed. Keep one active runtime per folder. Runtime/code/input mismatches stop recovery.

    Common-detector results do not isolate architecture: GDN has lookback 5, all within-file dimensions, shared mission weights and one seed; EXP012 has lookback 64, selected contexts, per-channel weights and three seeds. Complete-pipeline results retain EXP013's different TRAIN-selected native-score detector. Neither panel is a reproduction of paper leaderboard scoring. Historical TEST exposure and released TEST-extrema scaling remain disclosed. No point adjustment or TEST-label tuning is introduced.

    Read `docs/SCIENTIFIC_DECISION.md`, `docs/INPUTS.md`, `docs/RECOVERY.md` and `docs/COMPLETION_AUDIT.md` in the bundle. STGLR and OmniAnomaly are bounded fidelity/environment exclusions; a CATCH–Hundman reconstruction hybrid is unnecessary here. The output contains separate tables, notification timing, descriptive paired uncertainty, figures and a completion audit. No new training, architecture changes, calibration search or further model additions are scheduled.
    ''')]
    for c in cells:
        if c['cell_type']=='code':
            source=''.join(c['source']).replace('__BUNDLE_SHA256__',digest(bundle)).replace('__MANIFEST_SHA256__',digest(manifest))
            compile(source,'<notebook-cell>','exec');c['source']=source.splitlines(keepends=True)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':output.name,'provenance':[],'machine_shape':'hm'},'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'cells':cells}
    for i,c in enumerate(cells):c['id']=f'exp014-{i+1:02d}'
    output.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();build(a.bundle,a.manifest,a.output)
