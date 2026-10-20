"""Run on the actual Colab GPU BEFORE any research training. Synthetic inputs only."""
import argparse,subprocess,sys,gc,time,tempfile,copy
from core import *
from upstream import prepare,factory,catch_losses
from architectures import CommandReferenceGCN
from scoring import scores
from data import inject_batch

def gdn_reference(layer,x,embedding,edges):
    # Independent dense implementation of the original GraphLayer equations.
    n=len(x);edge=edges[:,edges[0]!=edges[1]];loops=torch.arange(n,device=x.device).repeat(2,1);edge=torch.cat([edge,loops],1)
    source,target=edge;z=layer.lin(x);zi=z[target][:,None,:];zj=z[source][:,None,:]
    ei=embedding[target][:,None,:];ej=embedding[source][:,None,:]
    ai=torch.cat([layer.att_i,layer.att_em_i],-1);aj=torch.cat([layer.att_j,layer.att_em_j],-1)
    alpha=torch.nn.functional.leaky_relu((torch.cat([zi,ei],-1)*ai).sum(-1)+(torch.cat([zj,ej],-1)*aj).sum(-1),.2)
    output=[]
    for i in range(n):
        idx=target==i;weight=alpha[idx].softmax(0);output.append((zj[idx]*weight[:,:,None]).sum(0).mean(0))
    return torch.stack(output)+layer.bias

def validate_official(paths,device):
    import copy
    records=[];seeds(17)
    for mission,nf in [('MSL',55),('SMAP',25)]:
        m,cfg=factory('gdn',mission,nf,paths,device);m.to(device).eval();layer=m.gnn_layers[0].gnn
        with torch.no_grad():layer.att_em_i.uniform_(-.2,.2);layer.att_em_j.uniform_(-.2,.2)
        x=torch.randn(nf,5,device=device,requires_grad=True);emb=m.embedding.weight.detach().clone().requires_grad_(True);edge=m.edge_index_sets[0].to(device)
        y=layer(x,edge,emb);loss=y.square().sum();loss.backward();grad=x.grad.clone();pgrad={k:p.grad.clone() for k,p in layer.named_parameters() if p.grad is not None}
        layer.zero_grad();x2=x.detach().clone().requires_grad_(True);e2=emb.detach().clone().requires_grad_(True);expected=gdn_reference(layer,x2,e2,edge);expected.square().sum().backward()
        torch.testing.assert_close(y,expected,rtol=1e-5,atol=3e-6);torch.testing.assert_close(grad,x2.grad,rtol=1e-5,atol=3e-6)
        torch.testing.assert_close(emb.grad,e2.grad,rtol=3e-5,atol=5e-6)
        for k,p in layer.named_parameters():
            if k in pgrad:torch.testing.assert_close(pgrad[k],p.grad,rtol=3e-5,atol=5e-6)
        # Runtime shape covers the published batch and actual mission node counts.
        m.train();m.zero_grad();xx=torch.randn(32,nf,5,device=device);op=torch.optim.Adam(m.parameters(),lr=.001);l=m(xx,None).square().mean();l.backward();op.step()
        records.append({'model':'GDN','mission':mission,'forward_and_gradient_equation_parity':'PASS','batch':32,'loss':float(l.detach())})
        del m,op;gc.collect()
        if device=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        m,cfg=factory('catch',mission,nf,paths,device);m.to(device);B=128 if device=='cuda' else 2
        xx=torch.randn(B,192,nf,device=device);aux,crit=catch_losses(paths,cfg['config']);m.train();out,z,d=m(xx);c=cfg['config'];l=(out-xx).square().mean()+c['dc_lambda']*d+c['auxi_lambda']*aux(z,m.revin_layer(xx,'transform'))
        if not torch.isfinite(l):raise RuntimeError('Official CATCH full-shape loss is nonfinite')
        l.backward()
        if any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise RuntimeError('Official CATCH gradient is nonfinite')
        # Adam state allocation is included in the actual GPU memory gate.
        op=torch.optim.Adam([p for k,p in m.named_parameters() if 'mask_generator' not in k],lr=c['lr']);op.step()
        with torch.no_grad():
            m.eval();sample=xx[:1];rec,_,_=m(sample);v=((sample-rec).square().mean(-1)+c['score_lambda']*crit(sample,rec).mean(-1))[:,-1]
            if not torch.isfinite(v).all():raise RuntimeError('Official CATCH native-score formula is nonfinite')
            for sparse in [False,True]:
                edge_case=torch.zeros(2,192,nf,device=device)
                if sparse:
                    edge_case[:,:,0]=torch.sin(torch.arange(192,device=device)/13.)
                    edge_case[:,::17,1]=1.
                er,_,_=m(edge_case);ev=((edge_case-er).square().mean(-1)+c['score_lambda']*crit(edge_case,er).mean(-1))[:,-1]
                if not torch.isfinite(ev).all():raise RuntimeError('Official CATCH constant/binary-command score is nonfinite')
        peak=torch.cuda.max_memory_allocated() if device=='cuda' else None
        records.append({'model':'CATCH','mission':mission,'unmodified_native_forward_loss_score':'PASS','constant_and_sparse_binary_command_score':'PASS','parameters':sum(p.numel() for p in m.parameters()),'batch':B,'full_published_batch_tested':device=='cuda','peak_GPU_bytes':peak})
        del m,op,xx,out,z,d,l,rec,v,edge_case,er,ev;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    return records

def validate_gpu_resume(paths):
    import train
    from unittest.mock import patch
    from types import SimpleNamespace
    real=train.model_for
    def tiny(role,mission,channels,nf,paths,device):
        m,c=real(role,mission,channels,nf,paths,device);c=copy.deepcopy(c);c.update(epochs=2,patience=20,batch_size=16)
        if role=='catch':
            c['config'].update(cf_dim=4,d_model=4,d_ff=8,e_layers=1,n_heads=1,head_dim=4,batch_size=16,num_epochs=2)
            cls=type(m);del m;gc.collect();m=cls(SimpleNamespace(**c['config']))
        return m,c
    gen=np.random.default_rng(712);raw={'fictional':gen.normal(size=(480,6)).astype(np.float32)};meta={'fictional':{'mission':'SMAP','rows':480,'columns':6,'fit_end':288,'stop_end':384,'cal_end':432}}
    result=[]
    with tempfile.TemporaryDirectory(prefix='EXP013_GPU_RESUME_') as td,patch.object(train,'model_for',tiny):
        for role in ['graph','gdn','catch']:
            with patch.object(train,'validate',side_effect=[2.,1.,2.,1.]):
                whole=train.train_job(Path(td)/('whole_'+role),raw,meta,'SMAP',role,17,paths,'cuda',{'synthetic_runtime_fixture':True})
                train.train_job(Path(td)/('resume_'+role),raw,meta,'SMAP',role,17,paths,'cuda',{'synthetic_runtime_fixture':True},epoch_limit=1)
                resumed=train.train_job(Path(td)/('resume_'+role),raw,meta,'SMAP',role,17,paths,'cuda',{'synthetic_runtime_fixture':True})
            if whole['epoch']!=2 or whole['state_sha256']!=resumed['state_sha256']:raise RuntimeError('Actual GPU epoch resume differs: '+role)
            result.append({'role':role,'actual_GPU_resume_state_byte_identity':'PASS','isolated_tiny_fixture_only':True})
            gc.collect();torch.cuda.empty_cache()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args();w=a.work.resolve();env=configure(a.device)
    # No prior generated receipt can authorise a different source/runtime.
    if (w/'ENGINEERING_GATE.json').exists():
        g=read(w/'ENGINEERING_GATE.json')
        if g['environment']==env and g['package_manifest_sha256']==sha(ROOT/'PACKAGE_MANIFEST.json') and g['status']=='PASS':print('REUSE VERIFIED ENGINEERING GATE');return
        raise RuntimeError('Engineering gate runtime/source differs')
    manifest(ROOT,'PACKAGE_MANIFEST.json');paths=prepare(w)
    r=subprocess.run([sys.executable,str(ROOT/'tests/test_science.py')],capture_output=True,text=True)
    atomic(w/'ENGINEERING_TEST_LOG.txt',(r.stdout+r.stderr).encode())
    if r.returncode:raise RuntimeError('Synthetic engineering tests failed; see ENGINEERING_TEST_LOG.txt')
    source=validate_official(paths,a.device)
    resume=[]
    if a.device=='cuda':
        for nf,entities in [(55,27),(25,54)]:
            for role in ['graph','removed','fusion']:
                m=CommandReferenceGCN(nf,entities,role).to('cuda');x=torch.randn(256,65,nf,device='cuda');e=torch.arange(256,device='cuda')%entities
                ref,l=m(x,e);inj,truth=inject_batch(x);_,li=m(inj,e)
                loss=torch.nn.functional.huber_loss(ref,x[:,-1,0])+.5*(torch.nn.functional.binary_cross_entropy_with_logits(l,torch.zeros_like(l))+torch.nn.functional.binary_cross_entropy_with_logits(li,truth));loss.backward()
                if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise RuntimeError('New-model full GPU loss/gradient is nonfinite')
                del m,x,e,ref,l,inj,truth,li,loss;gc.collect();torch.cuda.empty_cache()
        resume=validate_gpu_resume(paths)
    result={'status':'PASS','environment':env,'package_manifest_sha256':sha(ROOT/'PACKAGE_MANIFEST.json'),'synthetic_orchestration_and_scientific_tests':'PASS','official_source_validation':source,'actual_GPU_resume_validation':resume,'NASA_labels_opened':False,'NASA_TEST_float_arrays_opened':False,'scientific_fits':0,'full_GPU_validation':a.device=='cuda'}
    write(w/'ENGINEERING_GATE.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
