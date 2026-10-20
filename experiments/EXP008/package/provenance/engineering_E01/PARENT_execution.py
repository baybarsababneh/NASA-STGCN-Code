"""Execution safeguards derived from the successfully completed EXP006 run."""
import os, sys, json, uuid, time, threading, platform, subprocess, importlib.metadata, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import *
from v2_data import TRAIN_DIR

def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; CPU fallback is prohibited")
    if torch.__version__ != "2.11.0+cu130" or np.__version__ != "2.1.3" or pd.__version__ != "2.2.3":
        raise RuntimeError("Pinned package versions do not match the GPU protocol")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0].strip()
    packages = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
    return {"python": platform.python_version(), "platform_machine": platform.machine(),
            "libc": list(platform.libc_ver()), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "driver": driver, "gpu": props.name,
            "capability": list(torch.cuda.get_device_capability(device)), "gpu_total_memory": props.total_memory,
            "packages": packages, "torch_build": torch.__config__.show(),
            "dtype": "float32", "tf32": False, "amp": False, "compile": False,
            "deterministic": True, "cudnn_benchmark": False, "cublas_workspace": ":4096:8",
            "optimizer_foreach": False, "optimizer_fused": False, "threads": 1}

def install_data_guard():
    """Defense in depth: the supplied bundle contains no TEST/label data."""
    def guard(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0]))
        lower = p.name.lower()
        if lower.endswith(".npy") and p.resolve().parent != TRAIN_DIR.resolve():
            raise PermissionError(f"Array read outside TRAIN allowlist denied: {p}")
        if "labeled_anomal" in lower or "labelled_anomal" in lower:
            raise PermissionError("Anomaly label access is prohibited")
    sys.addaudithook(guard)

class Lease:
    """Single active notebook only. Lease helps detect accidental concurrent runs.

    Drive is not a distributed transactional filesystem. Do not run two notebooks.
    An abandoned lease expires after 10 minutes; every commit checks ownership.
    """
    def __init__(self, out):
        self.path = Path(out) / "ACTIVE_RUN.json"
        self.token = uuid.uuid4().hex
        self.stop = threading.Event()
        self.error = None

    def __enter__(self):
        if self.path.exists():
            old = json.loads(self.path.read_text())
            if time.time() - old["heartbeat"] < 600:
                raise RuntimeError("Another run may be active. Use one notebook only. After a crashed runtime, wait 10 minutes and Run All again.")
            self.path.rename(self.path.with_name("abandoned_lease_" + old["token"] + ".json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        value = {"token": self.token, "heartbeat": time.time(), "pid": os.getpid()}
        with open(self.path, "x") as f:
            json.dump(value, f)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def check(self):
        if self.error:
            raise RuntimeError("Durable run lease failed") from self.error
        if json.loads(self.path.read_text())["token"] != self.token:
            raise RuntimeError("Lost run lease; refusing to commit")

    def _heartbeat(self):
        while not self.stop.wait(30):
            try:
                self.check()
                write_json(self.path, {"token": self.token, "heartbeat": time.time(), "pid": os.getpid()})
            except Exception as exc:
                self.error = exc
                return

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.path.exists() and json.loads(self.path.read_text())["token"] == self.token:
            self.path.unlink()

def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value

def export_results(out):
    export = out / "exports"
    export.mkdir(exist_ok=True)
    entries = [p for p in out.rglob("*") if p.is_file() and "exports" not in p.relative_to(out).parts
               and not p.name.startswith("ACTIVE_RUN") and ".tmp-" not in p.name]
    for full in (False, True):
        name = "EXP007_GPU_V1_results.zip" if full else "EXP007_GPU_V1_summary.zip"
        target = export / name
        temp = target.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            for p in sorted(entries):
                if full or "fits" not in p.relative_to(out).parts:
                    z.write(p, p.relative_to(out))
        os.replace(temp, target)
        atomic_bytes(export / (name + ".sha256"), (sha(target) + "  " + name + "\n").encode())
