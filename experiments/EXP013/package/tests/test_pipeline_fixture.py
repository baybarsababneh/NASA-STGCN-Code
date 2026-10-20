"""End-to-end integration with four fictional entities. Never NASA scientific results."""
import sys,tempfile,copy,zipfile,io,csv,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import core,data,train,scoring,pipeline,upstream
from core import *
from types import SimpleNamespace
from unittest.mock import patch

def published_resume(paths):
    # Separate private tiny config; scientific published settings are never edited.
    configure('cpu');real=train.model_for
    def tiny(role,mission,channels,nf,paths,device):
        m,c=real(role,mission,channels,nf,paths,device);c=copy.deepcopy(c);c.update(epochs=2,patience=20,batch_size=16)
        if role=='catch':
            c['config'].update(cf_dim=4,d_model=4,d_ff=8,e_layers=1,n_heads=1,head_dim=4,batch_size=16,num_epochs=2)
            cls=type(m);del m;gc.collect();m=cls(SimpleNamespace(**c['config']))
        return m,c
    gen=np.random.default_rng(71);raw={'fake':gen.normal(size=(480,6)).astype(np.float32)};meta={'fake':{'mission':'SMAP','rows':480,'columns':6,'fit_end':288,'stop_end':384,'cal_end':432}}
    with tempfile.TemporaryDirectory(prefix='EXP013_RESUME_FIXTURE_') as td,patch.object(train,'model_for',tiny),patch.object(train,'validate',return_value=1.) as validation:
        for role in ['gdn','catch']:
            validation.side_effect=[2.,1.,2.,1.] if role=='gdn' else None
            a=Path(td)/('whole_'+role);b=Path(td)/('resume_'+role);identity={'fixture_only':True}
            whole=train.train_job(a,raw,meta,'SMAP',role,17,paths,'cpu',identity)
            train.train_job(b,raw,meta,'SMAP',role,17,paths,'cpu',identity,epoch_limit=1)
            resumed=train.train_job(b,raw,meta,'SMAP',role,17,paths,'cpu',identity)
            assert whole['epoch']==2 and whole['state_sha256']==resumed['state_sha256']
            print('PASS PUBLISHED RESUME',role,'including CATCH carried mask gradients and RNG')

def main(paths):
    configure('cpu');real=train.model_for
    def tiny(role,mission,channels,nf,paths,device):
        m,c=real(role,mission,channels,nf,paths,device);c=copy.deepcopy(c);c.update(epochs=1,patience=20,batch_size=16)
        if role=='catch':
            c['config'].update(cf_dim=4,d_model=4,d_ff=8,e_layers=1,n_heads=1,head_dim=4,batch_size=16,num_epochs=1)
            cls=type(m);del m;gc.collect();m=cls(SimpleNamespace(**c['config']))
        return m,c
    with tempfile.TemporaryDirectory(prefix='EXP013_SYNTHETIC_ONLY_') as td,patch.object(train,'model_for',tiny):
        root=Path(td)/'package';root.mkdir();(root/'inputs').mkdir();w=Path(td)/'work';w.mkdir();fake={};meta={};testrows={};copypayload={}
        gen=np.random.default_rng(812)
        with zipfile.ZipFile(root/'inputs/NASA_arrays_only.zip','w',zipfile.ZIP_DEFLATED) as z:
            for mission in ['MSL','SMAP']:
                for entity in range(2):
                    ch='FICTIONAL_'+mission+'_'+str(entity);a=gen.normal(0,.3,(480,6)).astype(np.float32);a[:,0]=np.sin(np.arange(480)/17)+gen.normal(0,.05,480);b=a[:400].copy();b[280:310,0]+=2.5
                    meta[ch]={'mission':mission,'rows':480,'columns':6,'fit_end':288,'stop_end':384,'cal_end':432,'mean':0.,'scale':1.};fake[ch]=a
                    for split,arr in [('train',a),('test',b)]:
                        buf=io.BytesIO();np.save(buf,arr);payload=buf.getvalue();name='data/data/'+split+'/'+ch+'.npy';z.writestr(name,payload);copypayload[name]=hashlib.sha256(payload).hexdigest()
                    testrows[ch]={'mission':mission,'rows':400,'columns':6,'sha256':copypayload['data/data/test/'+ch+'.npy']}
        write(root/'TEST_ROSTER.json',testrows);write(root/'NASA_ARRAY_COPY_AUDIT.json',{'new_array_archive_sha256':sha(root/'inputs/NASA_arrays_only.zip'),'files':copypayload});write(root/'PROTOCOL.json',{'fixture_only':True})
        history=[]
        for ch in meta:
            for s in [17,29,43]:history.append({'mission':meta[ch]['mission'],'arm':'lstm','channel':ch,'seed':s,'detector':'hundman_adapt_v1','H':0,'M':1,'U':0,'FPt':0,'N0':120,'TP':0,'FP':0,'FN':30})
        buf=io.StringIO();wr=csv.DictWriter(buf,fieldnames=list(history[0]));wr.writeheader();wr.writerows(history)
        with zipfile.ZipFile(root/'inputs/EXP012_RETROSPECTIVE_EVALUATION_REPORTS.zip','w') as z:z.writestr('channel_metrics.csv',buf.getvalue());z.writestr('mission_metrics.csv','mission,arm\nMSL,fixture_only\n')
        seal(root,'PACKAGE_MANIFEST.json');write(w/'DATA_FREEZE.json',{'channels':meta,'fixture_only':True});identity={'version':VERSION,'package_manifest_sha256':sha(root/'PACKAGE_MANIFEST.json'),'protocol_sha256':sha(root/'PROTOCOL.json'),'data_freeze_sha256':sha(w/'DATA_FREEZE.json'),'environment':environment('cpu')};write(w/'RUN_FREEZE.json',identity);write(w/'ENGINEERING_GATE.json',{'status':'PASS','environment':identity['environment'],'package_manifest_sha256':identity['package_manifest_sha256'],'fixture_only':True});write(w/'official/UPSTREAM_FREEZE.json',upstream.PINS)
        originals={x:x.ROOT for x in [core,data,train,scoring,pipeline,upstream]}
        for x in originals:x.ROOT=root
        try:
            with patch.object(pipeline,'prepare',return_value=paths):
                pipeline.fit_and_select(w,'cpu');pipeline.verify_study(w);pipeline.confirm_all(w,'cpu');pipeline.export_test(w,'cpu');pipeline.verify_export(w)
                # A fake event mapping tests the report path without opening any NASA label CSV.
                mapping={ch:[(280,310)] for ch in meta}
                with patch.object(pipeline,'read_labels',return_value=(mapping,{'fixture_only':True,'events':4})):
                    pipeline.evaluate(w,Path(td)/'FICTIONAL_LABELS_ONLY');pipeline.audit(w)
                before=sha(w/'reports/COMPLETE.json');pipeline.fit_and_select(w,'cpu');pipeline.export_test(w,'cpu');pipeline.evaluate(w,Path(td)/'NEVER_READ');assert sha(w/'reports/COMPLETE.json')==before
                # Integrity failure blocks label access even after an earlier good evaluation.
                q=next((w/'scores/units').glob('*/scores.npz'));q.write_bytes(q.read_bytes()+b'corrupt')
                with patch.object(pipeline,'read_labels') as parser:
                    try:pipeline.evaluate(w,Path(td)/'NEVER_READ')
                    except RuntimeError:pass
                    else:raise AssertionError('Corrupt seal was accepted')
                    parser.assert_not_called()
            print('PASS FULL SYNTHETIC INTEGRATION:22 fixture fits, study freeze, confirmation,44 score streams, evaluation, report audit, verified reuse and tamper-blocked labels. No NASA research fits/labels/TEST arrays.')
        finally:
            for x,p in originals.items():x.ROOT=p

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--official-work',type=Path,required=True);args=ap.parse_args();paths=upstream.prepare(args.official_work);published_resume(paths);main(paths)
