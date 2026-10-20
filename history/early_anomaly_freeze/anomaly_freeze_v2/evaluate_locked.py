from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import load_protocol, package_dir, project_root, read_csv_rows, sha256_file, write_csv_rows, write_json
from metrics import (
    alarms_to_events, average_precision, corrected_event_metrics, event_counts,
    interval_mask, merge_intervals, pointwise_counts, pointwise_metrics, detection_delays,
)
from scores import strict_alarm


def validate_frozen_and_scores():
    root = package_dir()
    freeze = json.loads((root / "FREEZE_MANIFEST.json").read_text())
    for name, expected in freeze["files_sha256"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"frozen file changed: {name}")
    score_manifest = read_csv_rows(root / "test_score_manifest.csv")
    for r in score_manifest:
        p = root / r["score_path"]
        if sha256_file(p) != r["sha256"]:
            raise RuntimeError(f"TEST score file hash mismatch: {p}")
    return score_manifest


def load_labels_first_row_per_channel():
    # This is deliberately the sole anomaly-label read in the package.
    labels_path = project_root().parent / "labeled_anomalies.csv"
    df = pd.read_csv(labels_path)
    # Preserve existing V2 convention: one unique channel, first row retained.
    df = df.drop_duplicates("chan_id", keep="first").copy()
    if df["chan_id"].nunique() != 81:
        raise RuntimeError("Expected 81 unique labelled channels")
    out = {}
    for _, row in df.iterrows():
        inclusive = ast.literal_eval(row["anomaly_sequences"])
        half_open = [(int(a), int(b) + 1) for a, b in inclusive]
        out[str(row["chan_id"])] = {
            "mission": str(row["spacecraft"]),
            "events": merge_intervals(half_open),
            "source_num_values": int(row["num_values"]),
        }
    return out


def threshold_lookup():
    rows = read_csv_rows(package_dir() / "thresholds.csv")
    d = {}
    for r in rows:
        key = (r["model_key"], r["channel"], str(r["seed"]), r["scorer"], float(r["quantile"]))
        d[key] = float(r["threshold"])
    return d


def clip_events_with_censor(events, start, end):
    clipped = []
    left_censored_indices = set()
    for a, b in events:
        aa, bb = max(a, start), min(b, end)
        if bb > aa:
            if a < start:
                left_censored_indices.add(len(clipped))
            clipped.append((aa, bb))
    return clipped, left_censored_indices


def per_stream_metrics(channel, mission, model_key, seed, scorer, quantile, window_start, score_file, threshold, true_events):
    z = np.load(score_file)
    idx = z["target_idx"].astype(np.int64)
    scores = z[scorer].astype(np.float64)
    mask = idx >= int(window_start)
    idx = idx[mask]
    scores = scores[mask]
    if len(idx) == 0:
        raise RuntimeError(f"{channel}: no scores in evaluation window")
    alarms = strict_alarm(scores, threshold)
    domain_start, domain_end = int(idx[0]), int(idx[-1]) + 1
    clipped_true, left_censored = clip_events_with_censor(true_events, domain_start, domain_end)
    pred_events = alarms_to_events(idx, alarms)
    c = event_counts(clipped_true, pred_events)
    positive_mask = interval_mask(idx, clipped_true)
    tp, fp, fn, tn = pointwise_counts(positive_mask, alarms)
    N0 = int(np.sum(~positive_mask))
    FPt = fp
    em = corrected_event_metrics(c["H"], c["M"], c["U"], FPt, N0)
    pm = pointwise_metrics(tp, fp, fn)
    delays = detection_delays(clipped_true, pred_events, left_censored=left_censored)
    false_runs = c["U"]
    return {
        "channel": channel, "mission": mission, "model_key": model_key, "seed": seed,
        "scorer": scorer, "quantile": quantile, "window_start": window_start,
        "threshold": threshold, "n_scored": len(idx), "true_events": len(clipped_true),
        "H": c["H"], "M": c["M"], "U": c["U"], "fragmentation": c["fragmentation"],
        "FPt": FPt, "N0": N0, "point_tp": tp, "point_fp": fp, "point_fn": fn, "point_tn": tn,
        "average_precision": average_precision(positive_mask, scores),
        "false_alarm_runs_per_10000_nominal": (false_runs / N0 * 10000.0) if N0 else np.nan,
        **em, **pm, **delays,
    }


def aggregate_mission(stream_rows):
    # Aggregate event/point counts first. AP and delay/burden summaries are channel-macro diagnostics.
    H = sum(int(r["H"]) for r in stream_rows); M = sum(int(r["M"]) for r in stream_rows); U = sum(int(r["U"]) for r in stream_rows)
    FPt = sum(int(r["FPt"]) for r in stream_rows); N0 = sum(int(r["N0"]) for r in stream_rows)
    tp = sum(int(r["point_tp"]) for r in stream_rows); fp = sum(int(r["point_fp"]) for r in stream_rows); fn = sum(int(r["point_fn"]) for r in stream_rows)
    out = {"H":H,"M":M,"U":U,"FPt":FPt,"N0":N0,"point_tp":tp,"point_fp":fp,"point_fn":fn}
    out.update(corrected_event_metrics(H,M,U,FPt,N0))
    out.update(pointwise_metrics(tp,fp,fn))
    for key in ["average_precision","false_alarm_runs_per_10000_nominal","fragmentation","delay_mean","delay_median","recall_within_1","recall_within_5","recall_within_10","recall_within_30"]:
        vals = np.asarray([float(r[key]) for r in stream_rows if np.isfinite(float(r[key]))], dtype=float)
        out[key + "_channel_macro"] = float(vals.mean()) if len(vals) else np.nan
    return out



def _seed_mean_corrected(rows):
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[str(r["seed"])].append(r)
    vals = []
    for seed, rs in sorted(by_seed.items()):
        H=sum(int(x["H"]) for x in rs); M=sum(int(x["M"]) for x in rs); U=sum(int(x["U"]) for x in rs)
        FPt=sum(int(x["FPt"]) for x in rs); N0=sum(int(x["N0"]) for x in rs)
        vals.append(corrected_event_metrics(H,M,U,FPt,N0)["corrected_event_f0_5"])
    vals=np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    return float(vals.mean()) if len(vals) else np.nan


def paired_channel_bootstrap(primary_rows, model_a, model_b, mission, reps=10000, rng_seed=2026):
    rows = [r for r in primary_rows if r["mission"] == mission and r["model_key"] in {model_a, model_b}]
    channels = sorted({r["channel"] for r in rows})
    seeds = sorted({str(r["seed"]) for r in rows if r["model_key"] == model_a})
    lookup = {(r["model_key"],r["channel"],str(r["seed"])): r for r in rows}
    for m in [model_a, model_b]:
        for ch in channels:
            for sd in seeds:
                if (m,ch,sd) not in lookup:
                    raise RuntimeError(f"bootstrap pairing missing {(m,ch,sd)}")
    observed_a = _seed_mean_corrected([lookup[(model_a,ch,sd)] for ch in channels for sd in seeds])
    observed_b = _seed_mean_corrected([lookup[(model_b,ch,sd)] for ch in channels for sd in seeds])
    observed = observed_b - observed_a
    rng = np.random.default_rng(int(rng_seed))
    deltas = np.empty(int(reps), dtype=float)
    n = len(channels)
    for i in range(int(reps)):
        sampled = rng.integers(0, n, size=n)
        a_rows=[]; b_rows=[]
        for j in sampled:
            ch=channels[int(j)]
            for sd in seeds:
                a_rows.append(lookup[(model_a,ch,sd)])
                b_rows.append(lookup[(model_b,ch,sd)])
        deltas[i] = _seed_mean_corrected(b_rows) - _seed_mean_corrected(a_rows)
    lo, hi = np.quantile(deltas, [0.025,0.975])
    return {
        "mission": mission, "model_a": model_a, "model_b": model_b,
        "delta_definition": "model_b_minus_model_a",
        "observed_delta_corrected_event_f0_5": float(observed),
        "bootstrap_replicates": int(reps), "rng_seed": int(rng_seed),
        "ci95_low": float(lo), "ci95_high": float(hi),
        "bootstrap_mean": float(np.mean(deltas)),
    }


def leave_one_channel_out_influence(primary_rows, model_a, model_b, mission):
    rows = [r for r in primary_rows if r["mission"] == mission and r["model_key"] in {model_a, model_b}]
    channels = sorted({r["channel"] for r in rows})
    full_a = _seed_mean_corrected([r for r in rows if r["model_key"]==model_a])
    full_b = _seed_mean_corrected([r for r in rows if r["model_key"]==model_b])
    full_delta = full_b-full_a
    out=[]
    for ch in channels:
        a=_seed_mean_corrected([r for r in rows if r["model_key"]==model_a and r["channel"]!=ch])
        b=_seed_mean_corrected([r for r in rows if r["model_key"]==model_b and r["channel"]!=ch])
        d=b-a
        out.append({
            "mission":mission,"model_a":model_a,"model_b":model_b,"excluded_channel":ch,
            "full_delta_model_b_minus_a":full_delta,"leave_one_out_delta_model_b_minus_a":d,
            "influence_full_minus_leave_one_out":full_delta-d,
        })
    return out

def main():
    score_manifest = validate_frozen_and_scores()
    labels = load_labels_first_row_per_channel()
    thresholds = threshold_lookup()
    protocol = load_protocol()
    root = package_dir()

    # Index score files by model/channel/seed.
    sm = {(r["model_key"], r["channel"], str(r["seed"])): root / r["score_path"] for r in score_manifest}

    settings = []
    # Primary + fixed raw/EWMA x 99/99.5 grid at index 250.
    for scorer in ["raw", "ewma_span10"]:
        for q in [0.99, 0.995]:
            settings.append((250, scorer, q, "primary_grid"))
    # Index-64 internal coverage check; keep primary raw/q99 operating rule only.
    settings.append((64, "raw", 0.99, "internal_index64_sensitivity"))

    stream_rows = []
    for window_start, scorer, q, panel in settings:
        for model_key in ["persistence","exp001","exp002","exp003","exp004","exp005"]:
            if panel == "internal_index64_sensitivity" and model_key not in {"exp002","exp003","exp004","exp005"}:
                continue
            for channel, lab in labels.items():
                seeds = ["deterministic"] if model_key == "persistence" else [str(x) for x in protocol["seeds"]]
                for seed in seeds:
                    tkey = (model_key, channel, seed, scorer, q)
                    if tkey not in thresholds:
                        raise RuntimeError(f"missing threshold {tkey}")
                    skey = (model_key, channel, seed)
                    if skey not in sm:
                        raise RuntimeError(f"missing score stream {skey}")
                    r = per_stream_metrics(channel, lab["mission"], model_key, seed, scorer, q, window_start, sm[skey], thresholds[tkey], lab["events"])
                    r["panel"] = panel
                    stream_rows.append(r)

    write_csv_rows(root / "evaluation_stream_metrics.csv", stream_rows)

    # Mission metrics per seed/settings.
    groups = defaultdict(list)
    for r in stream_rows:
        key = (r["panel"], r["window_start"], r["scorer"], r["quantile"], r["model_key"], r["mission"], str(r["seed"]))
        groups[key].append(r)
    mission_seed_rows = []
    for key, rs in groups.items():
        panel, w, scorer, q, model_key, mission, seed = key
        agg = aggregate_mission(rs)
        mission_seed_rows.append({"panel":panel,"window_start":w,"scorer":scorer,"quantile":q,"model_key":model_key,"mission":mission,"seed":seed,"channels":len(rs),**agg})
    write_csv_rows(root / "evaluation_mission_seed_metrics.csv", mission_seed_rows)

    # Seed means. Persistence has one deterministic row.
    groups2 = defaultdict(list)
    for r in mission_seed_rows:
        key = (r["panel"], r["window_start"], r["scorer"], r["quantile"], r["model_key"], r["mission"])
        groups2[key].append(r)
    summary = []
    metric_keys = ["corrected_event_f0_5","event_precision","event_recall","hundman_event_f0_5","point_precision","point_recall","point_f1","point_f0_5","average_precision_channel_macro","false_alarm_runs_per_10000_nominal_channel_macro","nominal_alarm_fraction"]
    for key, rs in groups2.items():
        row = dict(zip(["panel","window_start","scorer","quantile","model_key","mission"], key))
        row["seed_rows"] = len(rs)
        for m in metric_keys:
            vals = np.asarray([float(x[m]) for x in rs if np.isfinite(float(x[m]))], dtype=float)
            row[m + "_seed_mean"] = float(vals.mean()) if len(vals) else np.nan
        summary.append(row)
    write_csv_rows(root / "evaluation_summary.csv", summary)

    # Paired channel bootstrap for the two preregistered primary contrasts, primary operating point only.
    primary_rows = [r for r in stream_rows if r["panel"]=="primary_grid" and int(r["window_start"])==250 and r["scorer"]=="raw" and abs(float(r["quantile"])-0.99)<1e-12 and r["model_key"] != "persistence"]
    bootstrap_rows=[]
    influence_rows=[]
    reps=int(protocol["uncertainty"]["paired_channel_bootstrap_replicates"])
    rng_seed=int(protocol["uncertainty"]["bootstrap_rng_seed"])
    for model_a, model_b in protocol["primary_internal_comparisons"]:
        for mission in ["MSL","SMAP"]:
            bootstrap_rows.append(paired_channel_bootstrap(primary_rows, model_a, model_b, mission, reps=reps, rng_seed=rng_seed))
            influence_rows.extend(leave_one_channel_out_influence(primary_rows, model_a, model_b, mission))
    write_csv_rows(root / "primary_paired_bootstrap.csv", bootstrap_rows)
    write_csv_rows(root / "primary_channel_influence.csv", influence_rows)

    write_json(root / "evaluation_report.json", {
        "status":"PASS",
        "historical_test_exposure":"This is a locked retrospective NASA benchmark, not a virgin prospective holdout.",
        "primary_setting":{"window_start":250,"scorer":"raw","quantile":0.99},
        "warning":"Do not choose a different sensitivity cell as the headline result after inspecting outcomes."
    })
    print("LOCKED LABELLED EVALUATION: COMPLETE")
    print("Primary setting is index 250, raw residual, q=0.99. Other fixed cells are sensitivities.")


if __name__ == "__main__":
    main()
