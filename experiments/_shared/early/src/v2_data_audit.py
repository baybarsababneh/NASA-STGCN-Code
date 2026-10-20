from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "archive" / "data" / "data"

TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
LABELS_FILE = ROOT.parent / "labeled_anomalies.csv"


def main():
    labels = pd.read_csv(LABELS_FILE)

    train_files = sorted(TRAIN_DIR.glob("*.npy"))
    test_files = sorted(TEST_DIR.glob("*.npy"))

    train_channels = {p.stem for p in train_files}
    test_channels = {p.stem for p in test_files}
    labelled_channels = set(labels["chan_id"])

    duplicate_labels = (
        labels["chan_id"]
        .value_counts()
        .loc[lambda x: x > 1]
    )

    unique_labels = (
        labels
        .drop_duplicates("chan_id")
    )

    print("=" * 60)
    print("V2 DATA AUDIT")
    print("=" * 60)

    print(f"TRAIN files: {len(train_files)}")
    print(f"TEST files:  {len(test_files)}")
    print(f"Label rows:  {len(labels)}")
    print(
        f"Unique labelled channels: "
        f"{labels['chan_id'].nunique()}"
    )

    print("\nUnique labelled channels by mission:")
    print(
        unique_labels
        .groupby("spacecraft")["chan_id"]
        .nunique()
    )

    print("\nDuplicate label rows:")
    if duplicate_labels.empty:
        print("None")
    else:
        print(duplicate_labels)

    print("\nTRAIN channels without labels:")
    print(sorted(train_channels - labelled_channels))

    print("\nLabelled channels missing TRAIN files:")
    print(sorted(labelled_channels - train_channels))

    print("\nLabelled channels missing TEST files:")
    print(sorted(labelled_channels - test_channels))

    dimensions = {}
    lengths = []

    for path in train_files:
        array = np.load(path)

        dimensions[array.shape[1]] = (
            dimensions.get(array.shape[1], 0) + 1
        )

        lengths.append(array.shape[0])

    print("\nTRAIN input dimensions:")
    for dim, count in sorted(dimensions.items()):
        print(f"{dim} columns: {count} channels")

    print("\nTRAIN sequence lengths:")
    print(f"minimum: {min(lengths)}")
    print(f"maximum: {max(lengths)}")
    print(f"mean:    {np.mean(lengths):.1f}")

    assert train_channels == test_channels
    assert labels["chan_id"].nunique() == 81

    print("\nDATA AUDIT PASSED")


if __name__ == "__main__":
    main()