import argparse
import json
import math
import os
import sys
from pathlib import Path

# Deterministic cuBLAS setting must exist before CUDA work begins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn

from v2_data import TRAIN_DIR, load_channel
from v2_graph import build_channel_graph
from v2_models_exp006 import TCNGCNFixedTuned, count_parameters
import v2_train_exp006 as patched
import v2_train_exp006_frozen_cpu as frozen_cpu
from v2_exp006_device import (
    configure_device,
    optimizer_execution_kwargs,
    portable_state_dict,
    write_environment_record,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "v2_exp006_gcn_tuning.json"
AMENDMENT_PATH = ROOT / "EXP006_DEVICE_AMENDMENT.json"
SELECTED_PATH = ROOT / "results" / "v2_exp006_gcn_tuned" / "screen" / "selected_top2.json"
OUT_CSV = ROOT / "exp006_device_verification.csv"
SELECTION_PATH = ROOT / "exp006_device_verification_selection.json"
VERIFY_DIR = ROOT / "results" / "v2_exp006_gcn_tuned" / "device_verification"

PRED_RTOL = 1e-5
PRED_ATOL = 1e-6
GRAD_RTOL = 1e-4
GRAD_ATOL = 1e-6


def _load_json(path):
    return json.loads(Path(path).read_text())


def _candidate_lookup(config):
    return {c["candidate_id"]: c for c in config["candidates"]}


def _first_nonzero_variance_channel(config, mission):
    for channel in config["screening"]["channels"][mission]:
        raw = np.load(TRAIN_DIR / f"{channel}.npy", mmap_mode="r")
        fit_end = int(np.floor(len(raw) * float(config["dataset"]["fit_fraction"])))
        variance = float(np.var(np.asarray(raw[:fit_end, 0], dtype=np.float64)))
        if np.isfinite(variance) and variance > 0.0:
            return channel, variance
    raise RuntimeError(f"No non-zero fit-block telemetry variance channel found for {mission}")


def _make_model(channel, candidate, seed):
    patched.set_seed(seed, device_name="cpu")
    data = load_channel(
        channel=channel,
        lookback=int(candidate["lookback"]),
        fit_fraction=0.60,
        validation_end_fraction=0.80,
    )
    graph = build_channel_graph(channel=channel, fit_fraction=0.60)
    x_fit = patched.subset_features(data.x_fit, graph.feature_indices)
    x_val = patched.subset_features(data.x_val, graph.feature_indices)
    model = TCNGCNFixedTuned(
        num_nodes=graph.num_nodes,
        hidden_size=int(candidate["hidden_by_mission"][data.mission]),
        adjacency=graph.adjacency,
        kernel_size=int(candidate["kernel_size"]),
        dilations=tuple(int(d) for d in candidate["dilations"]),
    )
    return model, data, graph, x_fit, x_val


def _state_max_abs_diff(a, b):
    keys = list(a.keys())
    if keys != list(b.keys()):
        raise RuntimeError("State dict keys differ")
    max_diff = 0.0
    exact = True
    for k in keys:
        ta = a[k].detach().cpu()
        tb = b[k].detach().cpu()
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            return float("inf"), False
        if not torch.equal(ta, tb):
            exact = False
        if ta.numel():
            d = float((ta - tb).abs().max().item())
            max_diff = max(max_diff, d)
    return max_diff, exact


def _append(rows, **kwargs):
    row = {
        "check_group": kwargs.pop("check_group"),
        "mission": kwargs.pop("mission", None),
        "candidate_id": kwargs.pop("candidate_id", None),
        "channel": kwargs.pop("channel", None),
        "metric": kwargs.pop("metric"),
        "cpu_value": kwargs.pop("cpu_value", None),
        "cuda_value": kwargs.pop("cuda_value", None),
        "abs_diff": kwargs.pop("abs_diff", None),
        "tolerance": kwargs.pop("tolerance", None),
        "passed": bool(kwargs.pop("passed")),
        "notes": kwargs.pop("notes", ""),
    }
    row.update(kwargs)
    rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)


def _close(a, b, rtol, atol):
    return bool(np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol, equal_nan=False))


def _tensor_close(a, b, rtol, atol):
    return bool(torch.allclose(a.detach().cpu(), b.detach().cpu(), rtol=rtol, atol=atol))


def _manual_parity(channel, candidate, trained_cpu_state, seed, rows):
    mission = load_channel(channel, lookback=int(candidate["lookback"])).mission
    model0, data, graph, x_fit, _ = _make_model(channel, candidate, seed)
    model0.load_state_dict(trained_cpu_state)

    # Ensure this is not the zero-head persistence initialisation.
    head_nonzero = bool(
        torch.count_nonzero(model0.head.weight.detach()).item()
        or torch.count_nonzero(model0.head.bias.detach()).item()
    )
    _append(
        rows,
        check_group="numerical_parity",
        mission=mission,
        candidate_id=candidate["candidate_id"],
        channel=channel,
        metric="trained_checkpoint_nonzero_head",
        cpu_value=int(head_nonzero),
        cuda_value=int(head_nonzero),
        abs_diff=0.0,
        tolerance=0.0,
        passed=head_nonzero,
        notes="Parity uses a trained checkpoint, not the initial persistence head.",
    )

    configure_device("cuda")
    cpu = TCNGCNFixedTuned(
        num_nodes=graph.num_nodes,
        hidden_size=int(candidate["hidden_by_mission"][mission]),
        adjacency=graph.adjacency,
        kernel_size=int(candidate["kernel_size"]),
        dilations=tuple(int(d) for d in candidate["dilations"]),
    )
    gpu = TCNGCNFixedTuned(
        num_nodes=graph.num_nodes,
        hidden_size=int(candidate["hidden_by_mission"][mission]),
        adjacency=graph.adjacency,
        kernel_size=int(candidate["kernel_size"]),
        dilations=tuple(int(d) for d in candidate["dilations"]),
    )
    cpu.load_state_dict(trained_cpu_state)
    gpu.load_state_dict(trained_cpu_state)
    cpu = cpu.to("cpu")
    gpu = gpu.to("cuda")

    n = min(64, len(x_fit))
    xb_cpu = torch.from_numpy(x_fit[:n].astype(np.float32))
    yb_cpu = torch.from_numpy(data.y_fit[:n].astype(np.float32))
    xb_gpu = xb_cpu.to("cuda")
    yb_gpu = yb_cpu.to("cuda")
    criterion = nn.MSELoss()

    cpu.zero_grad(set_to_none=True)
    gpu.zero_grad(set_to_none=True)
    pred_cpu = cpu(xb_cpu)
    pred_gpu = gpu(xb_gpu)
    loss_cpu = criterion(pred_cpu, yb_cpu)
    loss_gpu = criterion(pred_gpu, yb_gpu)
    loss_cpu.backward()
    loss_gpu.backward()

    pred_diff = float((pred_cpu.detach() - pred_gpu.detach().cpu()).abs().max().item())
    pred_pass = _tensor_close(pred_cpu, pred_gpu, PRED_RTOL, PRED_ATOL)
    _append(
        rows,
        check_group="numerical_parity",
        mission=mission,
        candidate_id=candidate["candidate_id"],
        channel=channel,
        metric="prediction_max_abs_diff",
        cpu_value=float(pred_cpu.detach().abs().max().item()),
        cuda_value=float(pred_gpu.detach().abs().max().item()),
        abs_diff=pred_diff,
        tolerance=f"rtol={PRED_RTOL},atol={PRED_ATOL}",
        passed=pred_pass,
    )

    loss_diff = abs(float(loss_cpu.item()) - float(loss_gpu.item()))
    loss_pass = math.isclose(float(loss_cpu.item()), float(loss_gpu.item()), rel_tol=PRED_RTOL, abs_tol=PRED_ATOL)
    _append(
        rows,
        check_group="numerical_parity",
        mission=mission,
        candidate_id=candidate["candidate_id"],
        channel=channel,
        metric="loss_abs_diff",
        cpu_value=float(loss_cpu.item()),
        cuda_value=float(loss_gpu.item()),
        abs_diff=loss_diff,
        tolerance=f"rtol={PRED_RTOL},atol={PRED_ATOL}",
        passed=loss_pass,
    )

    max_grad_diff = 0.0
    grad_pass = True
    for (name_c, p_c), (name_g, p_g) in zip(cpu.named_parameters(), gpu.named_parameters()):
        if name_c != name_g:
            raise RuntimeError("Parameter ordering differs during parity check")
        if p_c.grad is None or p_g.grad is None:
            if p_c.grad is not None or p_g.grad is not None:
                grad_pass = False
            continue
        d = float((p_c.grad.detach() - p_g.grad.detach().cpu()).abs().max().item())
        max_grad_diff = max(max_grad_diff, d)
        if not _tensor_close(p_c.grad, p_g.grad, GRAD_RTOL, GRAD_ATOL):
            grad_pass = False
    _append(
        rows,
        check_group="numerical_parity",
        mission=mission,
        candidate_id=candidate["candidate_id"],
        channel=channel,
        metric="gradient_max_abs_diff",
        cpu_value=None,
        cuda_value=None,
        abs_diff=max_grad_diff,
        tolerance=f"rtol={GRAD_RTOL},atol={GRAD_ATOL}",
        passed=grad_pass,
    )

    opt_cpu = torch.optim.AdamW(
        cpu.parameters(),
        lr=float(candidate["learning_rate"]),
        weight_decay=1e-4,
        **optimizer_execution_kwargs(),
    )
    opt_gpu = torch.optim.AdamW(
        gpu.parameters(),
        lr=float(candidate["learning_rate"]),
        weight_decay=1e-4,
        **optimizer_execution_kwargs(),
    )
    opt_cpu.step()
    opt_gpu.step()

    max_param_diff = 0.0
    param_pass = True
    for (name_c, p_c), (name_g, p_g) in zip(cpu.named_parameters(), gpu.named_parameters()):
        if name_c != name_g:
            raise RuntimeError("Parameter ordering differs during update check")
        d = float((p_c.detach() - p_g.detach().cpu()).abs().max().item())
        max_param_diff = max(max_param_diff, d)
        if not _tensor_close(p_c, p_g, GRAD_RTOL, GRAD_ATOL):
            param_pass = False
    _append(
        rows,
        check_group="numerical_parity",
        mission=mission,
        candidate_id=candidate["candidate_id"],
        channel=channel,
        metric="one_step_parameter_max_abs_diff",
        cpu_value=None,
        cuda_value=None,
        abs_diff=max_param_diff,
        tolerance=f"rtol={GRAD_RTOL},atol={GRAD_ATOL}",
        passed=param_pass,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA verification requires a GPU runtime")

    config = _load_json(CONFIG_PATH)
    amendment = _load_json(AMENDMENT_PATH)
    selected = _load_json(SELECTED_PATH)
    locked = amendment["locked_shortlist"]
    if selected != locked:
        raise RuntimeError("selected_top2.json does not match the locked device-amendment shortlist")

    lookup = _candidate_lookup(config)
    rows = []
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    # Fix the verification channels before viewing any backend comparison.
    selection = {}
    for mission in config["dataset"]["missions"]:
        ch, var = _first_nonzero_variance_channel(config, mission)
        selection[mission] = {
            "channel": ch,
            "fit_block_telemetry_variance": var,
            "selection_rule": "first screening channel in frozen roster with non-zero fit-block telemetry variance",
        }
    SELECTION_PATH.write_text(json.dumps(selection, indent=2) + "\n")
    print("Verification channels fixed before backend comparison:")
    print(json.dumps(selection, indent=2))

    # ------------------------------------------------------------------
    # 1. Data/model identity and CPU regression. Keep CUDA deterministic
    # settings untouched until these CPU checks have completed.
    # ------------------------------------------------------------------
    trained_smoke_states = {}
    cpu_bridge = {}

    for mission in config["dataset"]["missions"]:
        channel = selection[mission]["channel"]

        data_a = load_channel(channel, lookback=64, fit_fraction=0.60, validation_end_fraction=0.80)
        data_b = load_channel(channel, lookback=64, fit_fraction=0.60, validation_end_fraction=0.80)
        graph_a = build_channel_graph(channel, fit_fraction=0.60)
        graph_b = build_channel_graph(channel, fit_fraction=0.60)

        identity_pass = (
            data_a.fit_end == data_b.fit_end
            and data_a.val_end == data_b.val_end
            and data_a.num_values == data_b.num_values
            and np.array_equal(graph_a.feature_indices, graph_b.feature_indices)
            and np.array_equal(graph_a.source_original_columns, graph_b.source_original_columns)
            and np.array_equal(graph_a.adjacency, graph_b.adjacency)
        )
        _append(
            rows,
            check_group="data_model_identity",
            mission=mission,
            channel=channel,
            metric="split_graph_exact_equality",
            cpu_value=f"fit_end={data_a.fit_end};val_end={data_a.val_end};n={data_a.num_values}",
            cuda_value=f"fit_end={data_b.fit_end};val_end={data_b.val_end};n={data_b.num_values}",
            abs_diff=0.0 if identity_pass else None,
            tolerance="exact",
            passed=identity_pass,
            notes=f"feature_indices={graph_a.feature_indices.tolist()}; adjacency compared exactly",
        )

        for cid in locked[mission]:
            cand = lookup[cid]
            m1, _, g1, _, _ = _make_model(channel, cand, args.seed)
            state1 = portable_state_dict(m1)
            m2, _, g2, _, _ = _make_model(channel, cand, args.seed)
            state2 = portable_state_dict(m2)
            maxdiff, exact = _state_max_abs_diff(state1, state2)
            param_count_equal = count_parameters(m1) == count_parameters(m2)
            init_pass = bool(exact and param_count_equal and np.array_equal(g1.adjacency, g2.adjacency))
            _append(
                rows,
                check_group="data_model_identity",
                mission=mission,
                candidate_id=cid,
                channel=channel,
                metric="initial_state_and_parameter_count",
                cpu_value=count_parameters(m1),
                cuda_value=count_parameters(m2),
                abs_diff=maxdiff,
                tolerance="exact",
                passed=init_pass,
            )

        # CPU regression uses first shortlisted configuration for each mission.
        cid = locked[mission][0]
        cand = lookup[cid]
        model_old, res_old = frozen_cpu.train_standard(
            channel=channel,
            seed=args.seed,
            base_config=config,
            candidate=cand,
            smoke_epochs=3,
        )
        model_new, res_new = patched.train_standard(
            channel=channel,
            seed=args.seed,
            base_config=config,
            candidate=cand,
            smoke_epochs=3,
            device_name="cpu",
            execution_environment_id="cpu_regression",
        )
        old_state = portable_state_dict(model_old)
        new_state = portable_state_dict(model_new)
        maxdiff, exact = _state_max_abs_diff(old_state, new_state)
        mse_diff = abs(float(res_old["validation_mse"]) - float(res_new["validation_mse"]))
        cpu_reg_pass = bool(exact and mse_diff == 0.0 and res_old["best_epoch"] == res_new["best_epoch"])
        _append(
            rows,
            check_group="cpu_regression",
            mission=mission,
            candidate_id=cid,
            channel=channel,
            metric="three_epoch_original_vs_patched",
            cpu_value=float(res_old["validation_mse"]),
            cuda_value=float(res_new["validation_mse"]),
            abs_diff=max(maxdiff, mse_diff),
            tolerance="exact",
            passed=cpu_reg_pass,
            notes=f"best_epoch old/new={res_old['best_epoch']}/{res_new['best_epoch']}; state_exact={exact}",
        )
        trained_smoke_states[mission] = portable_state_dict(model_new)

    # Full CPU side of all four bridge pairs, before enabling CUDA settings.
    for mission in config["dataset"]["missions"]:
        channel = selection[mission]["channel"]
        for cid in locked[mission]:
            cand = lookup[cid]
            model_cpu, result_cpu = patched.train_standard(
                channel=channel,
                seed=args.seed,
                base_config=config,
                candidate=cand,
                device_name="cpu",
                execution_environment_id="colab_cpu_bridge",
            )
            cpu_bridge[(mission, cid)] = {
                "result": result_cpu,
                "state": portable_state_dict(model_cpu),
            }
            del model_cpu

    # ------------------------------------------------------------------
    # 2. Enable deterministic CUDA and run numerical parity + full bridges.
    # ------------------------------------------------------------------
    configure_device("cuda")
    env = write_environment_record(ROOT, "cuda")
    env_id = env["execution_environment_id"]
    print(f"CUDA execution environment: {env_id}")

    for mission in config["dataset"]["missions"]:
        channel = selection[mission]["channel"]
        cid = locked[mission][0]
        _manual_parity(channel, lookup[cid], trained_smoke_states[mission], args.seed, rows)

    gpu_bridge_models = {}
    gpu_bridge_results = {}
    for mission in config["dataset"]["missions"]:
        channel = selection[mission]["channel"]
        for cid in locked[mission]:
            cand = lookup[cid]
            model_gpu, result_gpu = patched.train_standard(
                channel=channel,
                seed=args.seed,
                base_config=config,
                candidate=cand,
                device_name="cuda",
                execution_environment_id=env_id,
            )
            cpu_result = cpu_bridge[(mission, cid)]["result"]
            cpu_mse = float(cpu_result["validation_mse"])
            gpu_mse = float(result_gpu["validation_mse"])
            diff = abs(cpu_mse - gpu_mse)
            threshold = max(1e-6, 0.01 * abs(cpu_mse))
            passed = diff <= threshold
            _append(
                rows,
                check_group="full_training_bridge",
                mission=mission,
                candidate_id=cid,
                channel=channel,
                metric="validation_mse",
                cpu_value=cpu_mse,
                cuda_value=gpu_mse,
                abs_diff=diff,
                tolerance=threshold,
                passed=passed,
                notes=f"best_epoch cpu/cuda={cpu_result['best_epoch']}/{result_gpu['best_epoch']}",
            )
            gpu_bridge_results[(mission, cid)] = result_gpu
            gpu_bridge_models[(mission, cid)] = model_gpu

    # One same-environment GPU repeat per mission using the first shortlist item.
    for mission in config["dataset"]["missions"]:
        channel = selection[mission]["channel"]
        cid = locked[mission][0]
        cand = lookup[cid]
        model_repeat, result_repeat = patched.train_standard(
            channel=channel,
            seed=args.seed,
            base_config=config,
            candidate=cand,
            device_name="cuda",
            execution_environment_id=env_id,
        )
        first = gpu_bridge_results[(mission, cid)]
        first_mse = float(first["validation_mse"])
        repeat_mse = float(result_repeat["validation_mse"])
        diff = abs(first_mse - repeat_mse)
        threshold = max(1e-6, 0.01 * abs(first_mse))
        _append(
            rows,
            check_group="cuda_repeatability",
            mission=mission,
            candidate_id=cid,
            channel=channel,
            metric="validation_mse_repeat",
            cpu_value=first_mse,
            cuda_value=repeat_mse,
            abs_diff=diff,
            tolerance=threshold,
            passed=diff <= threshold,
            notes=f"best_epoch first/repeat={first['best_epoch']}/{result_repeat['best_epoch']}",
        )
        del model_repeat
        torch.cuda.empty_cache()

    # Checkpoint portability: first shortlisted GPU bridge per mission.
    for mission in config["dataset"]["missions"]:
        channel = selection[mission]["channel"]
        cid = locked[mission][0]
        cand = lookup[cid]
        model_gpu = gpu_bridge_models[(mission, cid)]
        state_cpu = portable_state_dict(model_gpu)
        ckpt = VERIFY_DIR / f"portable_{mission}_{cid}_seed{args.seed}.pt"
        torch.save({"state_dict": state_cpu, "candidate_id": cid, "channel": channel}, ckpt)
        loaded = torch.load(ckpt, map_location="cpu")

        model_cpu, data, graph, _, x_val = _make_model(channel, cand, args.seed)
        model_cpu.load_state_dict(loaded["state_dict"])
        model_cpu.eval()
        model_gpu.eval()
        n = min(64, len(x_val))
        xb_cpu = torch.from_numpy(x_val[:n].astype(np.float32))
        xb_gpu = xb_cpu.to("cuda")
        with torch.no_grad():
            p_cpu = model_cpu(xb_cpu)
            p_gpu = model_gpu(xb_gpu).cpu()
        diff = float((p_cpu - p_gpu).abs().max().item())
        passed = _tensor_close(p_cpu, p_gpu, PRED_RTOL, PRED_ATOL)
        _append(
            rows,
            check_group="checkpoint_portability",
            mission=mission,
            candidate_id=cid,
            channel=channel,
            metric="reloaded_prediction_max_abs_diff",
            cpu_value=float(p_cpu.abs().max().item()),
            cuda_value=float(p_gpu.abs().max().item()),
            abs_diff=diff,
            tolerance=f"rtol={PRED_RTOL},atol={PRED_ATOL}",
            passed=passed,
            notes=str(ckpt.relative_to(ROOT)),
        )

    # Clean GPU bridge models only after portability check.
    gpu_bridge_models.clear()
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    failures = df[~df["passed"].astype(bool)]
    print("\nEXP006 DEVICE VERIFICATION SUMMARY")
    print(df[["check_group", "mission", "candidate_id", "metric", "abs_diff", "tolerance", "passed"]].to_string(index=False))
    print(f"\nVerification rows: {len(df)}")
    print(f"Failures: {len(failures)}")
    print(f"CSV: {OUT_CSV}")

    if len(failures):
        raise SystemExit(
            "EXP006 device verification did not pass. Pause the migration and investigate; do not loosen tolerances."
        )

    print("EXP006 DEVICE VERIFICATION: PASS")


if __name__ == "__main__":
    main()
