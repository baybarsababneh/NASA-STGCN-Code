from __future__ import annotations

import numpy as np


def absolute_residual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    out = np.abs(y_true - y_pred)
    if not np.isfinite(out).all():
        raise ValueError("non-finite residual score")
    return out


def causal_ewma(scores, span=10):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("scores must be 1D")
    if len(scores) == 0:
        return scores.copy()
    if span <= 0:
        raise ValueError("span must be > 0")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    alpha = 2.0 / (float(span) + 1.0)
    out = np.empty_like(scores)
    out[0] = scores[0]
    for i in range(1, len(scores)):
        out[i] = alpha * scores[i] + (1.0 - alpha) * out[i - 1]
    return out


def empirical_quantile_higher(scores, q):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("scores must be a non-empty 1D array")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must be in [0,1]")
    try:
        return float(np.quantile(scores, float(q), method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(scores, float(q), interpolation="higher"))


def strict_alarm(scores, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    return scores > float(threshold)
