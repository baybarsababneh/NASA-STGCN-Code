"""Pinned official sources; two GDN PyG API compatibility edits retain its equations."""
import ast, sys, types, importlib, urllib.request, zipfile
from types import SimpleNamespace
from core import *

PINS={
 'gdn':{'repository':'d-ailin/GDN','commit':'9853899da860682669a134e4af315d036aab4eca','sha256':'8e4fa40b492f07a2d010fa26776cd2e78f318931dfa2cc63092d8ba10a848e9e'},
 'catch':{'repository':'decisionintelligence/CATCH','commit':'3647c69be5eb56649b072596cf89098e689e20c3','sha256':'d464a528bd312ec747dc2ad45c5ee7d5313f0d542eb778478f0fa443392be4c5'}}

def prepare(work):
    base=Path(work)/'official';base.mkdir(parents=True,exist_ok=True);paths={}
    for name,p in PINS.items():
        zpath=base/(name+'.zip')
        if not zpath.exists():
            u='https://codeload.github.com/'+p['repository']+'/zip/'+p['commit']
            with urllib.request.urlopen(u,timeout=60) as r: atomic(zpath,r.read())
        checked(zpath,p['sha256']);dst=base/name
        if not (dst/'SOURCE_MANIFEST.json').exists():
            dst.mkdir(exist_ok=True)
            with zipfile.ZipFile(zpath) as z:
                if z.comment.decode()!=p['commit']: raise RuntimeError('Upstream commit differs')
                prefix=z.namelist()[0].split('/')[0]+'/'
                for n in z.namelist():
                    rel=n[len(prefix):]
                    if not rel or n.endswith('/'): continue
                    # Only required source, README, licence and run-script files.
                    keep=(name=='gdn' and (rel.startswith(('models/','util/')) or rel in ['LICENSE','README.md','run.sh','evaluate.py','train.py'])) or (name=='catch' and (rel.startswith('ts_benchmark/baselines/catch/') or rel.startswith('scripts/multivariate_detection/detect_score/') or rel in ['README.md','requirements.txt','LICENSE']))
                    if keep:
                        q=dst/rel
                        if dst.resolve() not in q.resolve().parents: raise RuntimeError('Unsafe upstream path')
                        atomic(q,z.read(n))
            # Positional arg 3 was num_nodes in old PyG; it is ptr in modern PyG.
            if name=='gdn':
                q=dst/'models/graph_layer.py';b=q.read_text()
                old='softmax(alpha, edge_index_i, size_i)'
                if b.count(old)!=1: raise RuntimeError('Unexpected GDN source')
                b=b.replace(old,'softmax(alpha, edge_index_i, num_nodes=size_i)')
                init="super(GraphLayer, self).__init__(aggr='add', **kwargs)"
                if b.count(init)!=1: raise RuntimeError('Unexpected GDN constructor')
                b=b.replace(init,"super(GraphLayer, self).__init__(aggr='add', node_dim=0, **kwargs)")
                atomic(q,b.encode())
            seal(dst,'SOURCE_MANIFEST.json',{'upstream':p,'runtime_patch':'GDN num_nodes keyword; node_dim=0 to retain PyG1 node axis' if name=='gdn' else None})
        sm=manifest(dst,'SOURCE_MANIFEST.json')
        if sm['upstream']!=p:raise RuntimeError('Upstream provenance changed')
        # Re-derive file hashes from the hash-pinned original archive, so editing a
        # cached source AND its local manifest cannot silently change a baseline.
        with zipfile.ZipFile(zpath) as z:
            prefix=z.namelist()[0].split('/')[0]+'/'
            for rel,h in sm['files'].items():
                expected=z.read(prefix+rel)
                if name=='gdn' and rel=='models/graph_layer.py':
                    expected=expected.replace(b'softmax(alpha, edge_index_i, size_i)',b'softmax(alpha, edge_index_i, num_nodes=size_i)').replace(b"super(GraphLayer, self).__init__(aggr='add', **kwargs)",b"super(GraphLayer, self).__init__(aggr='add', node_dim=0, **kwargs)")
                if hashlib.sha256(expected).hexdigest()!=h:raise RuntimeError('Official source bytes changed: '+rel)
        paths[name]=dst
    freeze(base/'UPSTREAM_FREEZE.json',PINS)
    return paths

def namespace(name,path):
    if name not in sys.modules:
        mod=types.ModuleType(name);mod.__path__=[str(path)];sys.modules[name]=mod
    elif list(getattr(sys.modules[name],'__path__',[]))!=[str(path)]: raise RuntimeError('Conflicting upstream package '+name)

def factory(kind,mission,n_features,paths,device):
    if kind=='gdn':
        p=paths['gdn'];sys.path.insert(0,str(p))
        from util.env import set_device
        set_device(device)
        # Alias packages only refer to the pinned official tree.
        namespace('official_gdn',p);namespace('official_gdn.models',p/'models')
        GDN=importlib.import_module('official_gdn.models.GDN').GDN
        edge=torch.tensor([(j,i) for i in range(n_features) for j in range(n_features) if i!=j],dtype=torch.long).T
        m=GDN([edge],n_features,dim=64,out_layer_inter_dim=128,input_dim=5,out_layer_num=1,topk=5)
        return m,{'epochs':30,'patience':15,'batch_size':32,'lr':.001,'lookback':5,'source':'official run.sh; common chronological splits and TRAIN-only native-score normalization'}
    p=paths['catch'];parts=['ts_benchmark','ts_benchmark.baselines','ts_benchmark.baselines.catch']
    for i,n in enumerate(parts): namespace(n,p/('/'.join(n.split('.'))))
    mod=importlib.import_module('ts_benchmark.baselines.catch.models.CATCH_model')
    tree=ast.parse((p/'ts_benchmark/baselines/catch/CATCH.py').read_text())
    defaults=next(ast.literal_eval(a.value) for a in tree.body if isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS' for t in a.targets))
    # Exact mission detect_score script overrides from this pinned official commit.
    if mission=='MSL': defaults.update(Mlr=5e-5,batch_size=128,cf_dim=64,d_ff=256,d_model=128,e_layers=3,head_dim=64,lr=.0005,n_heads=2,num_epochs=5,patch_size=16,patch_stride=8,seq_len=192)
    else: defaults.update(Mlr=.01,auxi_lambda=1,batch_size=128,cf_dim=16,d_ff=32,d_model=64,dc_lambda=1,dropout=.4,e_layers=3,head_dim=64,inference_patch_size=4,lr=.005,n_heads=4,num_epochs=10,patch_size=16,patch_stride=8,score_lambda=1e-6,seq_len=192)
    defaults.update(c_in=n_features,enc_in=n_features,dec_in=n_features,c_out=n_features,task_name='anomaly_detection',pred_len=192,learning_rate=defaults['lr'])
    cfg=SimpleNamespace(**defaults);m=mod.CATCHModel(cfg)
    return m,{'epochs':cfg.num_epochs,'patience':cfg.patience,'batch_size':cfg.batch_size,'lr':cfg.lr,'lookback':191,'config':defaults,'source':'official detect_score mission script; unmodified forward, frequency losses and stochastic eval masks'}

def catch_losses(paths,cfg):
    from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_loss,frequency_criterion
    return frequency_loss(SimpleNamespace(**cfg)),frequency_criterion(SimpleNamespace(**cfg))
