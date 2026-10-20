#!/usr/bin/env python3
"""Check archival bytes without importing scientific dependencies or running models."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', action='store_true', help='Require all restored large files too')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'provenance/REPOSITORY_MANIFEST.json').read_text())
    errors = []
    checks = dict(manifest['files'])
    artifacts = json.loads((ROOT / 'provenance/ARTIFACT_PATHS.json').read_text())
    if args.artifacts:
        checks.update(artifacts)
    for name, value in checks.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name:
            errors.append('UNSAFE manifest path ' + name)
            continue
        path = ROOT / name
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != ROOT.parent):
            errors.append('SYMLINK ' + name)
        elif not path.is_file():
            errors.append('MISSING ' + name)
        elif path.stat().st_size != value['bytes'] or digest(path) != value['sha256']:
            errors.append('CHANGED ' + name)
    allowed = set(manifest['files']) | set(artifacts) | {'provenance/REPOSITORY_MANIFEST.json'}
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if '.git' in relative.parts or '__pycache__' in relative.parts:
            continue
        if path.is_symlink():
            errors.append('SYMLINK ' + relative.as_posix())
        elif path.is_file() and relative.as_posix() not in allowed:
            errors.append('UNEXPECTED ' + relative.as_posix())
    for i in range(1, 16):
        if not (ROOT / f'experiments/EXP{i:03d}/README.md').is_file():
            errors.append(f'MISSING experiment index EXP{i:03d}')
    for path in ROOT.rglob('*.ipynb'):
        nb = json.loads(path.read_text())
        if any(c.get('outputs') or c.get('execution_count') is not None for c in nb['cells'] if c['cell_type'] == 'code'):
            errors.append('UNSANITISED notebook ' + str(path.relative_to(ROOT)))
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(checks):,} archival files match SHA-256 and size; 15 experiment indexes present.')
    print('This verifies the package, not scientific execution or GPU compatibility.')


if __name__ == '__main__':
    main()
