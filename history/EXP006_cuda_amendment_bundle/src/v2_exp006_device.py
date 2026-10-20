import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_device(device_name):
    name = str(device_name).lower().strip()
    if name not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Enable a GPU runtime before continuing."
            )

        # Must be set before deterministic CUDA GEMM operations are launched.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        # Preserve full float32 behaviour; do not permit TF32 shortcuts.
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

        # Newer PyTorch versions may expose precision strings instead.
        try:
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                torch.backends.cuda.matmul.fp32_precision = "ieee"
        except Exception:
            pass
        try:
            if hasattr(torch.backends.cudnn, "conv") and hasattr(
                torch.backends.cudnn.conv, "fp32_precision"
            ):
                torch.backends.cudnn.conv.fp32_precision = "ieee"
        except Exception:
            pass

    return torch.device(name)


def optimizer_execution_kwargs():
    # Force the scalar/for-loop AdamW implementation on both CPU and CUDA so
    # the backend does not silently switch to foreach/fused implementations.
    kwargs = {"foreach": False}
    try:
        import inspect

        sig = inspect.signature(torch.optim.AdamW)
        if "fused" in sig.parameters:
            kwargs["fused"] = False
    except Exception:
        pass
    return kwargs


def portable_state_dict(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def environment_payload(root, device_name):
    root = Path(root)
    device = configure_device(device_name)

    tracked = [
        "src/v2_models_exp006.py",
        "src/v2_train_exp006.py",
        "src/v2_run_exp006.py",
        "src/v2_exp006_device.py",
        "src/v2_verify_exp006_device.py",
        "src/v2_train_exp006_frozen_cpu.py",
        "src/v2_run_exp006_frozen_cpu.py",
        "configs/v2_exp006_gcn_tuning.json",
        "configs/v2_exp006_screening_manifest.csv",
        "EXP006_PRETRAIN_FREEZE.json",
        "EXP006_DEVICE_AMENDMENT.json",
    ]
    hashes = {}
    for rel in tracked:
        p = root / rel
        if p.exists():
            hashes[rel] = sha256_file(p)

    gpu_name = None
    cuda_capability = None
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(idx)
        cuda_capability = list(torch.cuda.get_device_capability(idx))

    deterministic_enabled = bool(torch.are_deterministic_algorithms_enabled())
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "device_requested": str(device_name),
        "device_resolved": str(device),
        "dtype": "float32",
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpu_model": gpu_name,
        "gpu_compute_capability": cuda_capability,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": deterministic_enabled,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": (
            bool(torch.backends.cuda.matmul.allow_tf32)
            if hasattr(torch.backends.cuda.matmul, "allow_tf32")
            else None
        ),
        "cudnn_allow_tf32": (
            bool(torch.backends.cudnn.allow_tf32)
            if hasattr(torch.backends.cudnn, "allow_tf32")
            else None
        ),
        "optimizer_foreach": False,
        "optimizer_fused": False,
        "tracked_file_sha256": hashes,
    }

    canonical = dict(payload)
    canonical.pop("created_utc", None)
    env_id = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    payload["execution_environment_id"] = f"exp006_env_{env_id}"
    return payload


def write_environment_record(root, device_name):
    root = Path(root)
    payload = environment_payload(root, device_name)
    path = root / "exp006_execution_environment.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
