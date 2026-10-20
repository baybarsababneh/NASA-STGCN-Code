#!/usr/bin/env python3
"""Copy recorded inputs into an empty workspace; never launch a scientific job."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require_artifacts(prefixes):
    manifest = json.loads((ROOT / 'provenance/ARTIFACT_PATHS.json').read_text())
    absent = [name for name in manifest if name.startswith(tuple(prefixes)) and not (ROOT / name).is_file()]
    if absent:
        raise RuntimeError(f'Restore companion artefacts first ({len(absent):,} required files absent).')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('study', choices=[f'EXP{i:03d}' for i in range(1, 16)] + ['followups'])
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--historical-report', type=Path, help='Optional original historical PDF required for full legacy source verification when staging followups')
    a = p.parse_args()
    dest = a.destination.resolve()
    if dest == ROOT or ROOT in dest.parents or dest in ROOT.parents:
        raise RuntimeError('Use a separate destination outside the preserved repository.')
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Destination must be empty; nothing will be overwritten.')
    if a.study == 'followups':
        require_artifacts(['followups/', 'experiments/EXP012/preparation/package/', 'data/raw/archive.zip', 'inputs/original_results/EXP012_LABEL_BLIND_SCORE_EXPORT.zip', 'inputs/original_bundles/EXP012_ANOMALY_BENCHMARK_PREPARATION_V1_input_bundle.zip'])
        for source, name in [('followups/released_lstm', 'lstm_benchmark_audit'), ('followups/native_detector', 'hundman_native_crossover')]:
            shutil.copytree(ROOT / source, dest / name)
        items = [('data/raw/archive.zip', 'project_sources/03-archive.zip'), ('data/labels/labeled_anomalies.csv', 'project_sources/02-labeled_anomalies.csv'), ('inputs/original_results/EXP012_LABEL_BLIND_SCORE_EXPORT.zip', 'lstm_benchmark_audit/materialized/NASA Research Project/EXP012_LABEL_BLIND_SCORE_EXPORT.zip'), ('inputs/original_bundles/EXP012_ANOMALY_BENCHMARK_PREPARATION_V1_input_bundle.zip', 'lstm_benchmark_audit/materialized/NASA Research Project/EXP012_ANOMALY_BENCHMARK_PREPARATION_V1_input_bundle.zip')]
        for src, target in items:
            path = dest / target
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / src, path)
        if a.historical_report is not None:
            pdf = a.historical_report.expanduser().resolve()
            expected = 'fcd846cc5afcb5bbda2b7e92721f5593f5254090679cd0257c9bea80a66d9218'
            # The expected SHA-256 is the recorded original input's archive hash.
            actual = hashlib.sha256(pdf.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'Historical report hash mismatch, expected {expected}, got {actual}.')
            target = dest / 'upload/NASA_Project_SMAP_MSL_STGCN_Research_Report.pdf'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf, target)
        else:
            print('NOTE: Original historical report PDF is intentionally excluded. Strict legacy report-input verification needs --historical-report with the exact original PDF.')
        # The verifier expects the original full preparation capsule, not just src/.
        shutil.copytree(ROOT / 'experiments/EXP012/preparation/package', dest / 'lstm_benchmark_audit/exp012_source', dirs_exist_ok=True)
        print('Staged both follow-ups under their recorded directory names. See docs/REPRODUCTION.md.')
        return
    number = int(a.study[3:])
    if number <= 5:
        require_artifacts(['data/raw/nasa/data/data/train/'])
        shutil.copytree(ROOT / 'experiments/_shared/early', dest / 'Code')
        shutil.copytree(ROOT / 'data/raw/nasa/data/data/train', dest / 'archive/data/data/train')
        print(f'Staged shared early source and TRAIN only at {dest}. No saved results copied; read docs/REPRODUCTION.md.')
    elif number in [6, 7, 8, 10, 11]:
        prefix = f'experiments/{a.study}/package/'
        require_artifacts([prefix])
        shutil.copytree(ROOT / prefix, dest / 'package')
        if number == 11:
            require_artifacts(['inputs/original_results/EXP010_MECHANISMS_V1_results.zip'])
            shutil.copy2(ROOT / 'inputs/original_results/EXP010_MECHANISMS_V1_results.zip', dest / 'EXP010_MECHANISMS_V1_results.zip')
        print(f'Staged immutable capsule at {dest / "package"}; choose a different output directory when running.')
    else:
        raise RuntimeError('Use the original notebook and study-specific instructions in docs/REPRODUCTION.md; this study has additional recovery gates.')


if __name__ == '__main__':
    main()
