from __future__ import annotations

import math
import numpy as np


def merge_intervals(intervals):
    intervals = sorted((int(a), int(b)) for a, b in intervals if int(b) > int(a))
    merged = []
    for a, b in intervals:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:  # overlap or adjacency in half-open convention
            merged[-1][1] = max(merged[-1][1], b)
    return [tuple(x) for x in merged]


def clip_intervals(intervals, start, end):
    out = []
    for a, b in intervals:
        aa, bb = max(int(a), int(start)), min(int(b), int(end))
        if bb > aa:
            out.append((aa, bb))
    return merge_intervals(out)


def alarms_to_events(target_idx, alarms):
    target_idx = np.asarray(target_idx, dtype=np.int64)
    alarms = np.asarray(alarms, dtype=bool)
    if target_idx.shape != alarms.shape:
        raise ValueError("target_idx and alarms shape mismatch")
    if len(target_idx) and np.any(np.diff(target_idx) != 1):
        raise ValueError("target_idx must be consecutive")
    events = []
    start = None
    prev = None
    for idx, flag in zip(target_idx, alarms):
        if flag and start is None:
            start = int(idx)
        if not flag and start is not None:
            events.append((start, int(prev) + 1))
            start = None
        prev = int(idx)
    if start is not None:
        events.append((start, int(prev) + 1))
    return events


def overlaps(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def event_counts(true_events, pred_events):
    true_events = list(true_events)
    pred_events = list(pred_events)
    hit = sum(any(overlaps(t, p) for p in pred_events) for t in true_events)
    missed = len(true_events) - hit
    unmatched_pred = sum(not any(overlaps(t, p) for t in true_events) for p in pred_events)
    fragmentation = sum(max(0, sum(overlaps(t, p) for p in pred_events) - 1) for t in true_events)
    return dict(H=int(hit), M=int(missed), U=int(unmatched_pred), fragmentation=int(fragmentation))


def safe_div(num, den, undefined=np.nan):
    return float(num) / float(den) if den else undefined


def f_beta(precision, recall, beta=0.5):
    if not np.isfinite(precision) or not np.isfinite(recall):
        return np.nan
    b2 = beta * beta
    den = b2 * precision + recall
    if den == 0:
        return 0.0
    return (1.0 + b2) * precision * recall / den


def corrected_event_metrics(H, M, U, FPt, N0, beta=0.5):
    # Required convention: if no predicted event hit exists, precision/F are 0.
    pe = 0.0 if (H + U) == 0 else H / (H + U)
    re = np.nan if (H + M) == 0 else H / (H + M)
    nominal_frac = np.nan if N0 == 0 else FPt / N0
    pc = np.nan if not np.isfinite(nominal_frac) else pe * (1.0 - nominal_frac)
    return {
        "event_precision": float(pe),
        "event_recall": float(re) if np.isfinite(re) else np.nan,
        "nominal_alarm_fraction": float(nominal_frac) if np.isfinite(nominal_frac) else np.nan,
        "corrected_precision": float(pc) if np.isfinite(pc) else np.nan,
        "corrected_event_f0_5": f_beta(pc, re, beta=beta),
        "hundman_event_f0_5": f_beta(pe, re, beta=beta),
    }


def pointwise_counts(true_positive_mask, alarm_mask):
    y = np.asarray(true_positive_mask, dtype=bool)
    a = np.asarray(alarm_mask, dtype=bool)
    if y.shape != a.shape:
        raise ValueError("mask shape mismatch")
    tp = int(np.sum(y & a))
    fp = int(np.sum(~y & a))
    fn = int(np.sum(y & ~a))
    tn = int(np.sum(~y & ~a))
    return tp, fp, fn, tn


def pointwise_metrics(tp, fp, fn):
    p = 0.0 if tp + fp == 0 else tp / (tp + fp)
    r = np.nan if tp + fn == 0 else tp / (tp + fn)
    f1 = f_beta(p, r, beta=1.0)
    f05 = f_beta(p, r, beta=0.5)
    return {"point_precision": p, "point_recall": r, "point_f1": f1, "point_f0_5": f05}


def average_precision(y_true, scores):
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("shape mismatch")
    positives = int(y.sum())
    if positives == 0:
        return np.nan
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order].astype(np.int64)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / (tp + fp)
    return float(np.sum(precision * y_sorted) / positives)


def interval_mask(target_idx, intervals):
    idx = np.asarray(target_idx, dtype=np.int64)
    mask = np.zeros(len(idx), dtype=bool)
    for a, b in intervals:
        mask |= (idx >= a) & (idx < b)
    return mask


def false_alarm_runs(pred_events, true_events):
    return sum(not any(overlaps(t, p) for t in true_events) for p in pred_events)


def detection_delays(true_events, pred_events, left_censored=None):
    left_censored = set(left_censored or [])
    delays = []
    detected_within = {1: 0, 5: 0, 10: 0, 30: 0}
    eligible = 0
    for i, t in enumerate(true_events):
        if i in left_censored:
            continue
        eligible += 1
        hits = []
        for p in pred_events:
            if overlaps(t, p):
                first = max(t[0], p[0])
                hits.append(max(0, first - t[0]))
        if hits:
            d = min(hits)
            delays.append(d)
            for h in detected_within:
                if d <= h:
                    detected_within[h] += 1
    return {
        "delay_eligible_events": eligible,
        "detected_delay_count": len(delays),
        "delay_mean": float(np.mean(delays)) if delays else np.nan,
        "delay_median": float(np.median(delays)) if delays else np.nan,
        **{f"recall_within_{h}": (detected_within[h] / eligible if eligible else np.nan) for h in detected_within},
    }
