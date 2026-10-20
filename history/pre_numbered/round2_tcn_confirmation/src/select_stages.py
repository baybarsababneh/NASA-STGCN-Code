"""Apply recorded validation-only promotion rules; never accesses test/labels."""
import argparse,json
import numpy as np
import pandas as pd
from round2_train import ROOT,dump

def reference():
    d=pd.read_csv(ROOT/'inputs/original_validation_runs.csv')
    return d[((d.spacecraft=='MSL')&(d.configuration=='gcn_l64_top4') | (d.spacecraft=='SMAP')&(d.configuration=='gcn_l64_h32'))&(d.seed==17)].set_index('channel').val_mse

def score(rows,base):
    x=rows.set_index('channel').val_mse; b=base.reindex(x.index);mask=b>1e-4
    return float(np.log((x[mask]+1e-4)/(b[mask]+1e-4)).clip(-np.log(4),np.log(4)).mean())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','confirm']);args=ap.parse_args()
    p=json.loads((ROOT/'configs/screen_protocol.json').read_text());ch=pd.read_csv(ROOT/'inputs/channels.csv');base=reference();jobs=[];decisions={};table=[]
    if args.stage=='screen':
        d=pd.read_csv(ROOT/'results/screen_runs.csv');pruning={}
        for mission,mp in p['missions'].items():
            scores={c['name']:score(d[d.configuration==c['name']],base) for c in mp['candidates'] if len(d[d.configuration==c['name']])==len(mp['screen_channels'])}
            inc=mp['candidates'][0];assert inc['name'] in scores
            alt=min([c for c in mp['candidates'][1:] if c['name'] in scores],key=lambda c:scores[c['name']])
            # Exact lower-bound pruning: missing scores cannot be below -log(4).
            # This changes computation only, never the recorded winning criterion.
            for cfg in mp['candidates']:
                if cfg['name'] in scores:continue
                z=d[d.configuration==cfg['name']].set_index('channel');ref=base.reindex(mp['screen_channels']);assert (ref>1e-4).all()
                observed=np.log((z.val_mse+1e-4)/(base.reindex(z.index)+1e-4)).clip(-np.log(4),np.log(4))
                missing=sorted(set(mp['screen_channels'])-set(z.index));bound=float((observed.sum()-len(missing)*np.log(4))/len(mp['screen_channels']))
                assert bound>scores[alt['name']], 'Incomplete candidate could still win; finish screening'
                pruning[cfg['name']]=dict(completed=len(z),missing_channels=missing,best_possible_score=bound,best_complete_alternative=alt['name'],alternative_score=scores[alt['name']],reason='Cannot win even with maximally favorable remaining clipped scores; unfinished fits interrupted')
            promote=scores[alt['name']]<=scores[inc['name']]+np.log(.98)
            kept=[inc]+([alt] if promote else []);decisions[mission]=dict(scores=scores,incumbent=inc,challenger=alt if promote else None)
            for cfg in kept:
                jobs.extend([[c,mission,cfg,17] for c in ch[ch.spacecraft==mission].channel])
            # Matched budget fixed GCN is a control, not a new architecture search.
            control=dict(name=mission+'_fixed80',kind='gcn',lookback=64,hidden=inc['hidden'],graph=inc['graph'],epochs=80,lr=.001)
            jobs.extend([[c,mission,control,s] for s in [17,29,43] for c in ch[ch.spacecraft==mission].channel])
            for cfg in mp['candidates']:
                name=cfg['name'];sc=scores.get(name,np.nan)
                table.append(dict(spacecraft=mission,configuration=name,score=sc,ratio_to_incumbent=np.exp(sc-scores[inc['name']]),promoted=name in [c['name'] for c in kept],completed_runs=len(d[d.configuration==name]),pruned=name in pruning,best_possible_score=pruning.get(name,{}).get('best_possible_score',sc)))
        dump(ROOT/'configs/screen_pruning_note.json',pruning);dump(ROOT/'configs/screen_decisions.json',decisions);dump(ROOT/'configs/confirm_jobs.json',jobs);pd.DataFrame(table).to_csv(ROOT/'results/screen_summary.csv',index=False)
    else:
        prev=json.loads((ROOT/'configs/screen_decisions.json').read_text());d=pd.read_csv(ROOT/'results/confirm_runs.csv');assert len(d)==len(json.loads((ROOT/'configs/confirm_jobs.json').read_text()))
        for mission,mp in prev.items():
            inc=mp['incumbent'];alt=mp['challenger'];keep=False;checks={}
            if alt:
                for subset in ['all','nonscreen']:
                    sub=d[d.spacecraft==mission]
                    if subset=='nonscreen':sub=sub[~sub.channel.isin(p['missions'][mission]['screen_channels'])]
                    si=score(sub[sub.configuration==inc['name']],base);sa=score(sub[sub.configuration==alt['name']],base);checks[subset]=dict(incumbent=si,challenger=sa,ratio=np.exp(sa-si))
                keep=all(v['ratio']<=.98 for v in checks.values())
            kept=[inc]+([alt] if keep else []);decisions[mission]=dict(incumbent=inc,challenger=alt if keep else None,checks=checks)
            for cfg in kept:jobs.extend([[c,mission,cfg,s] for s in [29,43] for c in ch[ch.spacecraft==mission].channel])
        dump(ROOT/'configs/confirmation_decisions.json',decisions);dump(ROOT/'configs/robustness_jobs.json',jobs)
    print(json.dumps(decisions,indent=2));print('Jobs:',len(jobs))
if __name__=='__main__':main()
