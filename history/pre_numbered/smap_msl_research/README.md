# Spacecraft anomaly detection: reproducible research package

Start with **AUDIT.md**
for source discrepancies, scientific decisions and limitations. The full numerical
tables are under `results/`; the experiment is local and requires no Kaggle login.

## Reproduce

Python 3.12 and a CPU are sufficient. A clean environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/audit_sources.py
python src/test_invariants.py
python src/verify_archive_models.py
python src/legacy.py
python src/train.py
python src/select_and_repeat.py
python src/evaluate.py
python src/diagnostics.py
python src/report.py
```

Commands work from the package directory or any working directory because source
paths resolve relative to the package. Training and evaluation resume completed
runs from CSV files. To rerun from scratch, first make a separate copy, then remove
that copy's `results/`, `checkpoints/` and `configs/frozen_selection.json`, recreate
empty results/checkpoints directories, and run the commands. Never change the
frozen protocol and append results into the same experiment directories.

The archive includes the extracted original data, model weights and saved forecasts.
For a fresh source checkout without extracted data, put the supplied `archive.zip`
with the other six source files and run `python src/audit_sources.py --sources PATH`.
The original source files are not overwritten. The original archive's hash is in
`audit/source_manifest.json`; its duplicate ZIP bytes are not included in this
package because the complete extracted contents are present.

The primary environment used PyTorch's CPU wheel; `requirements.txt` includes its
official wheel index. Plot/report generation also needs the listed `tabulate` and
ReportLab dependencies. PDF visual inspection uses Poppler (`pdftoppm`), separately
installed as a system tool.

## Organisation

- `src/common.py`: interval definitions, metrics and data policies.
- `src/models.py`: readable temporal-graph network, MLP and compact LSTM.
- `src/train.py`: only official training arrays; chronological splits and checkpoints.
- `src/select_and_repeat.py`: graph selection before test evaluation; matched ablations.
- `src/evaluate.py`: independent calibration and separate legacy-compatible evaluation.
- `src/legacy.py`: authors' detector with a documented NumPy index-type patch.
- `src/verify_archive_models.py`: legacy hard-sigmoid inference consistency check.
- `src/diagnostics.py`: plots, seed comparisons, label sensitivity and paired bootstrap.
- `configs/`: prespecified protocol and frozen mission-specific graph selections.
- `checkpoints/`: state dictionaries, fit-only preprocessing and adjacency matrices;
  matching NPZ files contain predictions over official training data; JSON files
  retain complete training/validation curves.
- `results/training.csv`, `repeats.csv`: individual runs, settings, time and validation.
- `results/per_channel.csv`, `summary.csv`: full per-stream and mission metrics.
- `results/predictions/`: raw time indices, predictions, errors, masks and thresholds.
- `audit/`: supplied-source hashes, inventory, legacy replay and original text extraction.
- `vendor/`: downloaded authors' code and Apache licence; source snapshots are hashed.

## How to explain the model

For one stream, the history of a telemetry value and each active command indicator
is encoded over time. A graph layer combines command histories into the telemetry
node. The readout predicts the next telemetry value. Large forecast residuals,
smoothed in time, trigger an alarm using a threshold learned from a separate normal
training segment. Removing graph edges leaves the same temporal model; comparing
that control tests whether the command graph adds value.

Do not equate the clean fixed-threshold metrics with the paper's retrospective
detector. Do not use the provided test labels to retune the reported experiment.
Follow-up model development now needs a fresh prespecified protocol or a new held-out
dataset; repeated decisions based on these test results would contaminate evaluation.

Original work: Hundman et al., *Detecting Spacecraft Anomalies Using LSTMs and
Nonparametric Dynamic Thresholding* (2018), arXiv:1802.04431v3;
https://github.com/khundman/telemanom. New research code and findings are separate
from the authors' archived experiment. The graph is a command-context modelling
assumption, not a recovered spacecraft wiring diagram.
