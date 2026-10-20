from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "archive" / "data" / "data"
TRAIN_DIR = DATA_ROOT / "train"

EXCLUDED_CHANNELS = {"T-10"}

MISSION_BY_COLUMNS = {
    25: "SMAP",
    55: "MSL",
}


@dataclass
class ChannelData:
    channel: str
    mission: str

    x_fit: np.ndarray
    y_fit: np.ndarray
    persistence_fit: np.ndarray

    x_val: np.ndarray
    y_val: np.ndarray
    persistence_val: np.ndarray

    x_cal: np.ndarray
    y_cal: np.ndarray
    persistence_cal: np.ndarray

    fit_end: int
    val_end: int

    target_mean: float
    target_scale: float

    num_features: int
    num_values: int


def list_channels():
    """
    Return the 81 labelled-study channels without opening
    anomaly labels or NASA TEST data.

    T-10 is excluded because the V2 audit established that
    it is the single TRAIN/TEST stream without anomaly labels.
    """

    paths = sorted(TRAIN_DIR.glob("*.npy"))

    channels = [
        path.stem
        for path in paths
        if path.stem not in EXCLUDED_CHANNELS
    ]

    if len(channels) != 81:
        raise RuntimeError(
            f"Expected 81 study channels, found {len(channels)}."
        )

    return channels


def infer_mission(array):
    """
    Infer mission from input dimensionality.

    The independent V2 audit established:
      SMAP -> 25 columns
      MSL  -> 55 columns
    """

    num_columns = array.shape[1]

    if num_columns not in MISSION_BY_COLUMNS:
        raise ValueError(
            f"Unexpected input dimension: {num_columns}"
        )

    return MISSION_BY_COLUMNS[num_columns]


def _make_windows(
    standardized,
    target_times,
    lookback,
):
    """
    Build causal windows ending immediately before each target time.

    Input:
        standardized: [time, features]

    Output:
        x: [samples, features, lookback]
        y: standardized next telemetry value
        persistence: previous standardized telemetry value
    """

    all_windows = np.lib.stride_tricks.sliding_window_view(
        standardized,
        window_shape=lookback,
        axis=0,
    )

    window_indices = target_times - lookback

    x = all_windows[window_indices].copy().astype(np.float32)

    y = standardized[target_times, 0].copy().astype(np.float32)

    persistence = (
        standardized[target_times - 1, 0]
        .copy()
        .astype(np.float32)
    )

    return x, y, persistence


def load_channel(
    channel,
    lookback=64,
    fit_fraction=0.60,
    validation_end_fraction=0.80,
):
    """
    Load one channel using TRAIN data only.

    Chronological split:
        first 60%  -> fitting
        next 20%   -> validation
        final 20%  -> calibration

    Only the fitting block determines telemetry scaling.

    Column 0 is the telemetry target.
    Remaining columns are command-context features and are left
    unchanged because they are binary/encoded command indicators.
    """

    path = TRAIN_DIR / f"{channel}.npy"

    if not path.exists():
        raise FileNotFoundError(path)

    if channel in EXCLUDED_CHANNELS:
        raise ValueError(
            f"{channel} is excluded from the 81-channel V2 study."
        )

    raw = np.load(path).astype(np.float32)

    if raw.ndim != 2:
        raise ValueError(
            f"{channel}: expected a 2D array, got shape {raw.shape}."
        )

    mission = infer_mission(raw)

    n = len(raw)

    fit_end = int(np.floor(n * fit_fraction))
    val_end = int(np.floor(n * validation_end_fraction))

    if fit_end <= lookback:
        raise ValueError(
            f"{channel}: fitting block is too short for "
            f"lookback={lookback}."
        )

    if not (lookback < fit_end < val_end < n):
        raise ValueError(
            f"{channel}: invalid split boundaries "
            f"(lookback={lookback}, fit_end={fit_end}, "
            f"val_end={val_end}, n={n})."
        )

    # ---------------------------------------------------------
    # Fit-only telemetry standardisation.
    # ---------------------------------------------------------

    fit_target = raw[:fit_end, 0]

    target_mean = float(fit_target.mean())
    target_scale = float(fit_target.std())

    # Constant / near-constant channel safeguard.
    if not np.isfinite(target_scale) or target_scale < 1e-8:
        target_scale = 1.0

    standardized = raw.copy()

    standardized[:, 0] = (
        standardized[:, 0] - target_mean
    ) / target_scale

    # ---------------------------------------------------------
    # Target times.
    #
    # Validation/calibration may use earlier observed history as
    # causal context, but their target values never enter fitting.
    # ---------------------------------------------------------

    fit_times = np.arange(
        lookback,
        fit_end,
        dtype=np.int64,
    )

    val_times = np.arange(
        fit_end,
        val_end,
        dtype=np.int64,
    )

    cal_times = np.arange(
        val_end,
        n,
        dtype=np.int64,
    )

    x_fit, y_fit, persistence_fit = _make_windows(
        standardized,
        fit_times,
        lookback,
    )

    x_val, y_val, persistence_val = _make_windows(
        standardized,
        val_times,
        lookback,
    )

    x_cal, y_cal, persistence_cal = _make_windows(
        standardized,
        cal_times,
        lookback,
    )

    return ChannelData(
        channel=channel,
        mission=mission,

        x_fit=x_fit,
        y_fit=y_fit,
        persistence_fit=persistence_fit,

        x_val=x_val,
        y_val=y_val,
        persistence_val=persistence_val,

        x_cal=x_cal,
        y_cal=y_cal,
        persistence_cal=persistence_cal,

        fit_end=fit_end,
        val_end=val_end,

        target_mean=target_mean,
        target_scale=target_scale,

        num_features=raw.shape[1],
        num_values=n,
    )


if __name__ == "__main__":
    channels = list_channels()

    mission_counts = {
        "SMAP": 0,
        "MSL": 0,
    }

    for channel in channels:
        data = load_channel(channel)

        mission_counts[data.mission] += 1

    print("V2 TRAIN-only loader OK")
    print(f"Channels: {len(channels)}")
    print(f"SMAP: {mission_counts['SMAP']}")
    print(f"MSL:  {mission_counts['MSL']}")