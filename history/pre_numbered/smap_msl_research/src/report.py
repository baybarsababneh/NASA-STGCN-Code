"""Generate a concise research brief from the saved experiment tables."""
from pathlib import Path
import json,html
import pandas as pd
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import mm
from common import ROOT

def pct(v):return f'{100*v:.1f}%'
def num(v):return f'{v:.3f}'

def main():
    s=pd.read_csv(ROOT/'results/summary.csv');primary=s[(s.seed==17)&(s.protocol=='cal_s10_q0.995')];compat=s[(s.seed==17)&(s.protocol=='telemanom_compatibility')]
    selected=json.loads((ROOT/'configs/frozen_selection.json').read_text())['selected']
    train=pd.read_csv(ROOT/'results/training.csv');rep=pd.read_csv(ROOT/'results/repeats.csv');runs=pd.concat([train,rep],ignore_index=True)
    boot=pd.read_csv(ROOT/'results/paired_bootstrap.csv')
    models=['persistence','no_edges','MLP','LSTM_small','GCN'];names={'persistence':'Persistence','no_edges':'GCN, edges removed','MLP':'Lag MLP','LSTM_small':'Compact LSTM','GCN':'ST-GCN','archived_LSTM':'Archived LSTM replay'}
    def get(f,m,mission):return f[(f.model==m)&(f.spacecraft==mission)].iloc[0]
    graphrows=[]
    for mission in ['SMAP','MSL']:
        r=get(primary,'GCN',mission);graphrows.append([mission,pct(r.event_precision),pct(r.event_recall),num(r.event_f1),num(r.event_f05),num(r.point_f1)])
    compare=[]
    for model in models:
        a=get(primary,model,'SMAP');b=get(primary,model,'MSL');compare.append([names[model],num(a.event_f05),num(b.event_f05),num(a.point_f1),num(b.point_f1)])
    legacyrows=[]
    published={'SMAP':(.855,.855,.71),'MSL':(.926,.694,.69)}
    for mission in ['SMAP','MSL']:
        p,r,printed=published[mission];f05=1.25*p*r/(.25*p+r)
        legacyrows.append([mission,'Paper (published)',pct(p),pct(r),num(f05)])
        for model in ['archived_LSTM','GCN']:
            r=get(compat,model,mission);legacyrows.append([mission,names[model],pct(r.event_precision),pct(r.event_recall),num(r.event_f05)])
    seedrows=[]
    for mission in ['SMAP','MSL']:
        for model in ['GCN','no_edges']:
            g=s[(s.spacecraft==mission)&(s.model==model)&(s.protocol=='cal_s10_q0.995')]
            seedrows.append(dict(spacecraft=mission,model=model,mean=g.event_f05.mean(),sd=g.event_f05.std(),minimum=g.event_f05.min(),maximum=g.event_f05.max()))
    pd.DataFrame(seedrows).to_csv(ROOT/'results/seed_summary.csv',index=False)
    gains=[]
    for mission in ['SMAP','MSL']:
        r=boot[(boot.spacecraft==mission)&(boot.comparison=='GCN minus no_edges')&(boot.metric=='event_f05')].iloc[0]
        gains.append(f'{mission}: graph-minus-no-edges event F0.5 = {r.difference:+.3f}, channel-bootstrap 95% interval [{r.lower95:+.3f}, {r.upper95:+.3f}].')
    fullprimary=primary[['spacecraft','model','event_precision','event_recall','event_f1','event_f05','point_precision','point_recall','point_f1','point_f05','macro_average_precision','false_events_per_1000_steps']]
    best=[]
    for mission,c in selected.items():
        tr=train[(train.spacecraft==mission)&(train.configuration==c['name'])]
        best.append(f'{mission}: lookback {c["lookback"]}, hidden width {c["hidden"]}, {c["graph"]} graph; median selected epoch {tr.best_epoch.median():g}, parameter count {tr.parameters.min():,}-{tr.parameters.max():,}.')
    runtime=runs.seconds.sum()/60
    md=f'''# SMAP/MSL spatiotemporal GCN: initial research results

Prepared for discussion with the research team by Baybars Ababneh. This is a completed
initial local experiment, not a claim of a new state of the art or an exact
retraining of the 2018 LSTM. Code-assisted analysis.

## Main finding

The present ST-GCN does not establish an advantage over the original LSTM. Under
the common legacy detector, its event F0.5 is 0.840 (SMAP) and 0.741 (MSL), versus
0.847 and 0.893 for the archived LSTM replay. The fixed calibration detector has
low event precision and fragmented false alarms; its primary GCN event F0.5 is
0.042 and 0.120. This is a useful negative result and a warning that forecast
quality and detector calibration must be assessed separately.

## Method and graph

Audited all seven supplied sources. Each file contains one telemetry target and
binary command context. The graph has a telemetry node and active command-indicator
nodes, with directed command-to-target edges; it is not a physical sensor network.
A shared temporal convolution, indicator-specific gains, one relational graph
convolution and residual readout forecast the next telemetry value. We trained one
model per stream. The matched edge-free control isolates the graph contribution.

The CSV has a duplicated P-2 entry, leaving 54 unique labelled SMAP streams and 27
MSL streams. Main scoring unions the conflicting P-2 intervals and excludes unlabelled
T-10: 68 SMAP and 36 MSL events, across 488495 common evaluated points.

Training uses chronological 60/20/20 fit/validation/calibration splits. Only fit
data determine preprocessing and graphs. Five targeted graph configurations vary
lookback, width and neighbour selection. The minimum median per-channel validation
MSE chooses the graph for each mission. Test labels are used only for evaluation.
Thresholds use the held-out calibration 99.5th percentile and causal EWMA span 10.
No point adjustment or test-optimal threshold is used.

## Best tested graph configurations

{chr(10).join('- '+b for b in best)}

All models: AdamW lr=.001, weight decay=.0001, batch 256, at most 40 epochs,
patience 6, gradient clipping 1. The official train/test files remain separate.
There are {len(runs)} saved model fits, including 405 graph candidate fits, fresh
controls, and repeated graph/edge-free seeds. Aggregate recorded fitting time is
{runtime:.1f} CPU-process minutes; wall time and evaluation overhead differ.
Seeds 17,29,43 are retained; seed 17 is primary, never chosen by test score.

## Primary results: fixed calibration thresholds

{fullprimary.to_markdown(index=False,floatfmt='.3f')}

Event metrics follow the paper's any-overlap definition. Raw point metrics are
reported separately, without filling entire anomalous intervals after one hit.
Macro average precision averages per-channel ranking quality. False-event rates
are per 1000 evaluated time steps, not a deployment-calibrated false-alarm rate.

## Does the graph help?

{' '.join(gains)}

These intervals resample whole streams and are conditional on this small benchmark;
channels may still be correlated. They do not quantify new-mission generalisation.
The complete three-seed outcomes are:

{pd.DataFrame(seedrows).to_markdown(index=False,floatfmt='.3f')}

Do not select the best test-performing seed or sensitivity setting after seeing
these tables. Lower validation forecast error is not necessarily better anomaly
detection: residual magnitudes, regime shifts and threshold calibration also matter.

## Comparison with the original work

{pd.DataFrame(legacyrows,columns=['Mission','Source / model','Event precision','Event recall','Standard F0.5']).to_markdown(index=False)}

Paper values are its published 82-row population, which counts P-2 twice. The
replay and GCN rows use the corrected 81-stream population and the same retained
upstream detector. The detector uses retrospective startup smoothing and expanded
intervals; these results are not the clean fixed-threshold primary result.
Saved LSTM weights reproduce archived forecasts within 4e-7 on checked windows.
The full archived-error replay agrees within 4.1e-7 with saved smoothed errors.

The paper prints F0.5=.71 (SMAP) and .69 (MSL), inconsistent with its own precision
and recall under the standard formula. The table above recomputes standard F0.5
from published rounded P/R; it does not silently substitute the printed values.
The archive log also differs from the paper in MSL false positives. Current
upstream code corrects a 250-index alignment issue and has details absent from the
paper; exact historical equivalence is therefore not claimed.

## Interpretation and limitations

- This release is chiefly a single-target, command-conditioned forecasting task.
  No verified synchronisation or physical adjacency supports a cross-file sensor graph.
- The compact residual predictor and fresh LSTM control differ from the original
  2x80 LSTM trained on shuffled windows with 250 history steps and 10 output horizons.
  We reuse original archived predictions; we do not claim a full fresh 2018 retraining.
- Constant fitting targets occur in 12 SMAP and 5 MSL streams. A persistence skip
  can miss sustained level anomalies after their transition. Sparse command data,
  short calibration segments and test-regime changes limit calibration reliability.
- The README reports test-extrema scaling, but 11 MSL test arrays exceed [-1,1],
  so uniform preprocessing provenance is uncertain. Fit-only standardisation cancels
  a positive affine transform on nonconstant targets; constant fallback and legacy
  raw-unit guardrails retain caveats. Physical units are unavailable.
- Event overlap can conceal late detections and broad alarms. Use point scores,
  alarm coverage, traces and delays alongside event F-scores. This event-centred
  dataset cannot establish real operational precision over long quiet periods.
- The supplied reference notebooks use synthetic anomalies, cross-file windows or
  test-derived thresholds, so their headline accuracy is not a valid comparator.

## Sensible next steps

1. Review the data/graph interpretation and P-2 ambiguity with the research team. Confirm
   whether she wants a command-context GCN or access to synchronised multichannel telemetry.
2. Use a new prespecified development/holdout protocol to test absolute versus
   residual forecasting, multi-horizon losses and more robust calibration. These
   are motivated follow-ups, not adjustments to the completed test benchmark.
3. Seek longer normal calibration histories and a second dataset with known
   synchronisation/topology before claiming a general benefit from graph modelling.

## Reproduction and evidence

README.md gives executable commands. All checkpoint weights, fit-only preprocessing,
adjacencies, seeds, curves, predictions and threshold sensitivities are retained.
AUDIT.md explains all source and scoring discrepancies. No external account or
website content was changed.

Sources: supplied Hundman et al. arXiv:1802.04431v3, attached email/CSV/ZIP/notebooks;
read-only authors' code at https://github.com/khundman/telemanom. The original code
snapshot and its Apache licence are retained under vendor/.
'''
    (ROOT/'Technical_Report.md').write_text(md)
    fontdir=Path(reportlab.__file__).parent/'fonts'
    pdfmetrics.registerFont(TTFont('Vera',str(fontdir/'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('VeraBold',str(fontdir/'VeraBd.ttf')))
    pdfmetrics.registerFontFamily('Vera',normal='Vera',bold='VeraBold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodySmall',fontName='Vera',fontSize=8.5,leading=11,spaceAfter=6,textColor=colors.HexColor('#283746')))
    styles.add(ParagraphStyle(name='SectionSmall',fontName='VeraBold',fontSize=10.5,leading=14,spaceBefore=9,spaceAfter=5,textColor=colors.HexColor('#146b7b')))
    styles['Title'].fontName='VeraBold';styles['Title'].fontSize=18;styles['Title'].leading=24;styles['Title'].textColor=colors.HexColor('#153b4b')
    story=[]
    def P(text):story.append(Paragraph(text,styles['BodySmall']))
    def H(text):story.append(Paragraph(text,styles['SectionSmall']))
    def T(header,rows,widths):
        data=[[Paragraph(str(x),styles['BodySmall']) for x in header]]+[[Paragraph(str(x),styles['BodySmall']) for x in row] for row in rows]
        table=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2eef1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f6f8')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
        story.append(table);story.append(Spacer(1,5))
    story.append(Paragraph('Spacecraft anomaly detection<br/>A first spatiotemporal GCN experiment',styles['Title']))
    P('Baybars Ababneh | Research discussion with the research team | 17 September 2026')
    P('<b>Finding:</b> this ST-GCN does not establish an improvement over the original LSTM. Fixed calibration produces excessive false alarms; using the original detector yields substantially higher event scores, showing that thresholding strongly affects the comparison.')
    H('Method and defensible graph')
    P('Each file contains <b>one telemetry target plus binary command context</b>. The graph connects active command-indicator nodes to the telemetry node; it does not claim physical sensor connectivity. A temporal convolution, typed graph aggregation and residual readout predict the next telemetry value. Models are fitted separately for each stream.')
    P('Main evaluation: <b>54 unique SMAP + 27 MSL streams</b>, 104 labelled events. Duplicate P-2 intervals are unioned; unlabelled T-10 is excluded. Training is split chronologically 60% fit / 20% validation / 20% calibration. Model selection and thresholds use no test labels. Causal EWMA span 10 and the calibration 99.5th percentile define the primary detector.')
    H('Selected configurations and experiment coverage')
    for b in best:P(b)
    P(f'Five targeted graph settings; persistence, MLP and compact LSTM controls; matched graph-edge ablations; seeds 17/29/43 for selected graphs and ablations. <b>{len(runs)} model fits</b> retained. AdamW .001, batch 256, at most 40 epochs; early stopping restores the best validation checkpoint.')
    H('Primary ST-GCN results (declared seed 17)')
    T(['Mission','Event P','Event R','Event F1','Event F0.5','Point F1'],graphrows,[58,69,69,69,83,73])
    H('Controlled comparison: fixed calibration thresholds')
    T(['Model','SMAP event F0.5','MSL event F0.5','SMAP point F1','MSL point F1'],compare,[133,72,72,72,72])
    P('Event scores use any overlap with a true interval. Point scores are unadjusted: a single hit does not fill the entire labelled event. The edge-free model keeps the same temporal architecture, allowing a direct test of graph contribution.')
    story.append(PageBreak())
    H('Graph contribution and seed stability')
    for g in gains:P(g)
    story.append(Image(str(ROOT/'figures/seed_ablation.png'),width=421,height=159))
    H('Original LSTM comparison: separate compatibility track')
    T(['Mission','Source / model','Event P','Event R','F0.5*'],legacyrows,[49,172,67,67,66])
    P('*Standard F0.5. The paper prints .71 for SMAP and .69 for MSL, which disagree with its P/R under the standard formula. Published rows count duplicate P-2; local rows use unique streams. The archived detector includes retrospective startup and buffered intervals; these scores must not be mixed with primary calibration scores.')
    H('What limits the conclusions')
    P('The release has uncertain scaling provenance (11 MSL test arrays exceed the documented range) and ambiguous labels; current code also differs from the historical run. Archived weights and forecasts were verified, but the original 2018 network was not retrained. Our compact LSTM is a fresh controlled baseline. Sparse command activity, constant training targets and short calibration histories limit learning. Persistence residuals can miss sustained level shifts. Event overlap is lenient; point scores and false-alarm coverage matter.')
    H('Next discussion with the research team')
    P('Confirm whether a command-context graph meets the intended task and resolve P-2. For a separately preregistered follow-up, test absolute versus residual prediction and improved normal-data calibration. Obtain synchronised multichannel telemetry or a second dataset before making broader claims about graph benefits.')
    P('Deliverables: full audit, executable code, configurations, checkpoints, curves, per-channel predictions, threshold/label sensitivities and bootstrap tables. All development was local; external pages remained read-only. Sources: supplied Hundman et al. (2018) paper and files; authors\' Telemanom repository. Full details and links: Technical_Report.md and AUDIT.md.')
    def footer(canvas,doc):
        canvas.setFont('Vera',7);canvas.setFillColor(colors.HexColor('#657787'));canvas.drawString(17*mm,12*mm,'SMAP/MSL | Initial research experiment | Reproducible evidence in the accompanying package');canvas.drawRightString(193*mm,12*mm,str(doc.page))
    SimpleDocTemplate(str(ROOT/'Technical_Summary.pdf'),title='SMAP/MSL: Initial Spatiotemporal GCN Experiment',author='Baybars Ababneh',pagesize=(210*mm,297*mm),rightMargin=17*mm,leftMargin=17*mm,topMargin=16*mm,bottomMargin=19*mm).build(story,onFirstPage=footer,onLaterPages=footer)
    print('Technical report and PDF brief generated')

if __name__=='__main__':main()
