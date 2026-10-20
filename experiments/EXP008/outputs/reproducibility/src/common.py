"""Integrity/commit helpers reused unchanged from completed EXP006."""
import hashlib

import io

import json

import os

import shutil

import time

import uuid

import zipfile

from pathlib import Path

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with open(temp, "wb") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def write_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode())

def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"Frozen artifact mismatch: {path}. Do not mix executions.")
    else:
        write_json(path, value)

def validate_package(root):
    root = Path(root)
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts
              and p.name != "PACKAGE_MANIFEST.json"}
    if actual != set(manifest["files"]):
        raise RuntimeError("Bundle file roster differs from its manifest")
    for rel, expected in manifest["files"].items():
        if sha(root / rel) != expected:
            raise RuntimeError(f"Bundle checksum mismatch: {rel}")
    npys = sorted(root.rglob("*.npy"))
    if len(npys) != 81 or any(p.parent != root / "train" for p in npys):
        raise RuntimeError("Only the 81 allowlisted TRAIN arrays may be present")
    return manifest

def job_id(job):
    return digest(job)[:24]

def read_commit(out, job, run_hash):
    base = Path(out) / "fits" / job["stage"] / job_id(job)
    done = base.with_suffix(".done.json")
    if not done.exists():
        return None
    commit = json.loads(done.read_text())
    archive = base.with_suffix(".zip")
    if commit["run_hash"] != run_hash or commit["job"] != job:
        raise RuntimeError(f"Commit identity mismatch: {base}")
    if not archive.exists() or sha(archive) != commit["archive_sha256"]:
        raise RuntimeError(f"Corrupt committed fit: {base}; preserve and investigate")
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != {"result.json", "checkpoint.pt", "history.json", "job.json"}:
            raise RuntimeError(f"Incomplete fit archive: {base}")
        if json.loads(z.read("job.json")) != job:
            raise RuntimeError("Archive job mismatch")
        result = json.loads(z.read("result.json"))
    return result

def commit_fit(out, job, run_hash, local_archive):
    base = Path(out) / "fits" / job["stage"] / job_id(job)
    base.parent.mkdir(parents=True, exist_ok=True)
    final = base.with_suffix(".zip")
    temp = base.with_suffix(".zip.tmp-" + uuid.uuid4().hex)
    expected = sha(local_archive)
    with open(local_archive, "rb") as src, open(temp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    if sha(temp) != expected:
        raise RuntimeError("Durable checkpoint copy verification failed")
    os.replace(temp, final)
    write_json(base.with_suffix(".done.json"),
               {"run_hash": run_hash, "job": job, "archive_sha256": expected})
