#!/usr/bin/env python3
"""Restore hash-verified large files from the companion object ZIPs. No code is executed."""
import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def safe_target(name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name:
        raise ValueError('Unsafe archive path: ' + name)
    path = ROOT / name
    if any(p.is_symlink() for p in [path, *path.parents] if p != ROOT.parent):
        raise ValueError('Refusing symlink in destination: ' + name)
    return path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parts', nargs='+', type=Path)
    parser.add_argument('--prefix', default='', help='Optional logical path prefix, e.g. experiments/EXP007/')
    args = parser.parse_args()
    mapping = json.loads((ROOT / 'provenance/ARTIFACT_PATHS.json').read_text())
    selected = {n: v for n, v in mapping.items() if n.startswith(args.prefix)}
    if not selected:
        raise SystemExit('No artefacts match that prefix.')
    archives = []
    try:
        available = {}
        for part in args.parts:
            archive = zipfile.ZipFile(part)
            archives.append(archive)
            for info in archive.infolist():
                p = PurePosixPath(info.filename)
                if len(p.parts) == 4 and p.parts[:2] == ('objects', 'sha256'):
                    available[p.name] = (archive, info)
        # Check all required objects and destinations before writing anything.
        missing = []
        pending = []
        for name, value in selected.items():
            path = safe_target(name)
            if path.exists():
                if not path.is_file() or path.stat().st_size != value['bytes'] or digest(path) != value['sha256']:
                    raise ValueError('Existing file differs; no overwrite: ' + name)
                continue
            h = value['sha256']
            if h not in available:
                missing.append(h)
            elif available[h][1].file_size != value['bytes']:
                raise ValueError('Wrong declared object size: ' + h)
            else:
                pending.append((path, value))
        if missing:
            raise ValueError(f'{len(set(missing))} required objects absent; supply all companion parts.')
        done = 0
        for path, value in pending:
            archive, info = available[value['sha256']]
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
                    tmp = Path(out.name)
                    hasher = hashlib.sha256()
                    with archive.open(info) as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            out.write(block)
                            hasher.update(block)
                if tmp.stat().st_size != value['bytes'] or hasher.hexdigest() != value['sha256']:
                    raise ValueError('Corrupt object: ' + value['sha256'])
                # No concurrent restore or scientific process should use this tree.
                if path.exists():
                    raise ValueError('Destination appeared during restore: ' + str(path))
                os.replace(tmp, path)
            finally:
                if tmp is not None and tmp.exists():
                    tmp.unlink()
            done += 1
        print(f'Restored {done:,} files; {len(selected) - done:,} already verified. No experiment was run.')
    finally:
        for archive in archives:
            archive.close()


if __name__ == '__main__':
    main()
