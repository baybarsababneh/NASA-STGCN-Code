from __future__ import annotations

import numpy as np

from metrics import (
    alarms_to_events, average_precision, clip_intervals, corrected_event_metrics,
    event_counts, interval_mask, merge_intervals, pointwise_counts, pointwise_metrics,
)
from scores import causal_ewma, empirical_quantile_higher, strict_alarm
from io_utils import package_dir, write_json


def main():
    # Half-open merge: overlap and adjacency merge.
    assert merge_intervals([(1, 3), (3, 5), (10, 12), (11, 14)]) == [(1, 5), (10, 14)]
    assert clip_intervals([(0, 10), (20, 30)], 5, 25) == [(5, 10), (20, 25)]

    idx = np.arange(10, 20)
    alarms = np.array([0,1,1,0,0,1,0,1,1,1], dtype=bool)
    assert alarms_to_events(idx, alarms) == [(11,13),(15,16),(17,20)]

    true = [(11, 14), (17, 19)]
    pred = [(11, 12), (13, 14), (15, 16), (17, 20)]
    c = event_counts(true, pred)
    assert c == {"H": 2, "M": 0, "U": 1, "fragmentation": 1}

    # Always-on is penalised by nominal-time correction.
    idx2 = np.arange(0, 10)
    true2 = [(4, 6)]
    y2 = interval_mask(idx2, true2)
    pred2 = alarms_to_events(idx2, np.ones(10, dtype=bool))
    c2 = event_counts(true2, pred2)
    FPt = int(np.sum(~y2))
    N0 = int(np.sum(~y2))
    m2 = corrected_event_metrics(c2["H"], c2["M"], c2["U"], FPt, N0)
    assert m2["corrected_event_f0_5"] == 0.0

    # Never-on: missed event, zero F.
    c3 = event_counts(true2, [])
    m3 = corrected_event_metrics(c3["H"], c3["M"], c3["U"], 0, N0)
    assert m3["corrected_event_f0_5"] == 0.0

    # Pointwise exact case.
    a = np.array([0,1,1,0], dtype=bool)
    y = np.array([0,1,0,0], dtype=bool)
    tp, fp, fn, tn = pointwise_counts(y, a)
    assert (tp, fp, fn, tn) == (1,1,0,2)
    pm = pointwise_metrics(tp, fp, fn)
    assert abs(pm["point_precision"] - 0.5) < 1e-12
    assert abs(pm["point_recall"] - 1.0) < 1e-12

    # AP ranking: perfect ranking must be 1.
    assert abs(average_precision([1,0,1,0], [4,1,3,2]) - 1.0) < 1e-12

    # EWMA adjust=False exact recurrence, and higher quantile strict exceedance.
    z = causal_ewma([1.0, 0.0, 0.0], span=10)
    alpha = 2/11
    assert abs(z[0] - 1.0) < 1e-12
    assert abs(z[1] - (1-alpha)) < 1e-12
    scores = np.arange(100, dtype=float)
    thr = empirical_quantile_higher(scores, 0.99)
    assert thr == 99.0
    assert strict_alarm(scores, thr).sum() == 0

    # Ties/constant-score safeguard: no adaptive fallback.
    const = np.ones(50)
    thr2 = empirical_quantile_higher(const, 0.99)
    assert thr2 == 1.0 and strict_alarm(const, thr2).sum() == 0

    write_json(package_dir() / "metric_test_report.json", {"status":"PASS","scope":"synthetic only; no NASA TEST arrays or anomaly labels accessed"})
    print("SYNTHETIC METRIC TESTS: PASS")


if __name__ == "__main__":
    main()
