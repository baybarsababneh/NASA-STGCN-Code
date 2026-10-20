"""Frozen plans and promotion logic. No test-data input or adaptive search."""
import numpy as np
import pandas as pd


def plans(config, channels, selected=None):
    jobs={"robust":[], "canonical":[], "ablation":[]}
    for mission, roster in config["screening_channels"].items():
        for fold in config["folds"]:
            for role in config["roles"]:
                for ch in roster:
                    for seed in config["robust_seeds"]:
                        jobs["robust"].append(dict(stage="robust",fold=fold,mission=mission,channel=ch,
                            seed=seed,role=role,gate_mode=role,remove_messages=False))
    for ch, mission in channels.items():
        for role in config["roles"]:
            for seed in config["canonical_seeds"]:
                jobs["canonical"].append(dict(stage="canonical",fold="canonical",mission=mission,channel=ch,
                    seed=seed,role=role,gate_mode=role,remove_messages=False))
        if selected is not None:
            for seed in config["canonical_seeds"]:
                jobs["ablation"].append(dict(stage="ablation",fold="canonical",mission=mission,channel=ch,
                    seed=seed,role="message_removed",gate_mode=selected[mission]["gate_mode"],remove_messages=True))
    expected={"robust":144,"canonical":729,"ablation":243 if selected is not None else 0}
    if {k:len(v) for k,v in jobs.items()}!=expected:raise RuntimeError("Frozen plan count mismatch")
    return jobs


def decide(rows, config):
    df=pd.DataFrame(rows)
    if set(df.stage) != {"canonical","robust"}:raise RuntimeError("Selection must precede final ablation")
    if len(df)!=873:raise RuntimeError("Incomplete study cannot select a model")
    selected={};checks=[]
    for mission in config["missions"]:
        c=df[(df.stage=="canonical")&(df.mission==mission)]
        means=c.groupby("role").validation_mse.mean().to_dict()
        if not all(np.isfinite(v) and v>0 for v in [means['fixed'],means['static']]):
            raise RuntimeError("Relative promotion denominator undefined")
        improvement=100*(means['fixed']-means['dynamic'])/means['fixed']
        byseed=[]
        for seed in config["canonical_seeds"]:
            ss=c[c.seed==seed].groupby("role").validation_mse.mean()
            byseed.append(float(ss['fixed']-ss['dynamic']))
        robust={r:[] for r in ['fixed','static']}
        for fold in config['folds']:
            m=df[(df.stage=='robust')&(df.mission==mission)&(df.fold==fold)].groupby('role').assessment_mse.mean()
            for ref in robust:
                if not m[ref]>0:raise RuntimeError("Undefined robustness denominator")
                robust[ref].append(float(100*(m[ref]-m['dynamic'])/m[ref]))
        gates={'passes_2pct_vs_fixed':improvement>=config['promotion_minimum_pct'],
               'passes_each_seed_vs_fixed':all(v>0 for v in byseed),
               'passes_robustness_vs_fixed':np.mean(robust['fixed'])>=0,
               'passes_canonical_vs_static':means['dynamic']<means['static'],
               'passes_robustness_vs_static':np.mean(robust['static'])>=0}
        promoted=bool(all(gates.values()))
        selected[mission]={'gate_mode':'dynamic' if promoted else 'fixed',
            'decision':'promote_dynamic_gate' if promoted else 'retain_EXP006_selected_architecture',
            'spec':config['baseline_by_mission'][mission],
            'validation_mse':float(means['dynamic'] if promoted else means['fixed']),
            'selected_without_detector_metrics':True}
        checks.append(dict(mission=mission, fixed_mse=means['fixed'],static_mse=means['static'],
            dynamic_mse=means['dynamic'],improvement_pct=improvement,
            robust_improvement_pct_vs_fixed=float(np.mean(robust['fixed'])),
            robust_improvement_pct_vs_static=float(np.mean(robust['static'])),
            **{f'seed{s}_absolute_improvement_vs_fixed':v for s,v in zip(config['canonical_seeds'],byseed)},
            **{k:bool(v) for k,v in gates.items()},promoted=promoted))
    return selected,checks


def paired_tables(rows, selected, config):
    df=pd.DataFrame(rows)
    canonical=df[df.stage=='canonical']
    pairs=[];boot=[]
    for mission in config['missions']:
        part=canonical[canonical.mission==mission]
        comparisons=[('dynamic','fixed'),('dynamic','static')]
        for candidate,reference in comparisons+ [('message_removed',selected[mission]['gate_mode'])]:
            if candidate=='message_removed':
                cand=df[(df.stage=='ablation')&(df.mission==mission)]
            else:cand=part[part.role==candidate]
            ref=part[part.role==reference]
            merged=cand[['channel','seed','validation_mse']].merge(ref[['channel','seed','validation_mse']],
                on=['channel','seed'],suffixes=('_candidate','_reference'),validate='one_to_one')
            if len(merged)!=len(ref):raise RuntimeError('Unpaired channel/seed data')
            per=merged.groupby('channel',as_index=False)[['validation_mse_candidate','validation_mse_reference']].mean()
            per['mission']=mission;per['comparison']=candidate+'_vs_'+reference
            per['family']=per.channel.str.split('-').str[0]
            per['absolute_improvement']=per.validation_mse_reference-per.validation_mse_candidate
            per['relative_improvement_pct']=np.where(per.validation_mse_reference>1e-8,
                100*per.absolute_improvement/per.validation_mse_reference,np.nan)
            pairs.append(per)
            # Resample paired channels, after averaging their three seeds.
            rng=np.random.default_rng(config['bootstrap_seed'])
            delta=per.absolute_improvement.to_numpy()
            samples=delta[rng.integers(0,len(delta),size=(config['bootstrap_replicates'],len(delta)))].mean(axis=1)
            boot.append(dict(mission=mission,comparison=candidate+'_vs_'+reference,
                channels=len(delta),mean_absolute_improvement=float(delta.mean()),
                lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)),
                positive_channels=int((delta>0).sum()),negative_channels=int((delta<0).sum()),
                diagnostic_only_not_a_promotion_gate=True))
    return pd.concat(pairs,ignore_index=True),pd.DataFrame(boot)
