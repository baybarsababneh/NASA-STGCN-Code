"""All roles use the completed EXP006 training loop without numerical changes."""
import json
import numpy as np
import torch
from models_exp007 import MessageGatedGCN
from v2_data import load_channel
from v2_graph import build_channel_graph
from training_core import (set_seed, subset_features, _train_with_early_stopping,
                           _verify_initial_persistence, _load_robust_fold, make_loader,
                           evaluate_mse)


def train_job(job, config):
    set_seed(job["seed"])
    spec = config["baseline_by_mission"][job["mission"]]
    lookback = spec["lookback"]
    common = config["training_common"]
    if job["stage"] == "robust":
        fold = config["folds"][job["fold"]]
        arrays = _load_robust_fold(job["channel"], lookback, fold)
        fit_fraction = fold["fit_end_fraction"]
        xfit, yfit = arrays["x_fit"], arrays["y_fit"]
        xstop, ystop = arrays["x_stop"], arrays["y_stop"]
        xe, ye, pe = arrays["x_assess"], arrays["y_assess"], arrays["p_assess"]
        metadata = {k: arrays[k] for k in ("fit_end", "stop_end", "assess_end", "target_mean", "target_scale", "num_values")}
        mission = arrays["mission"]
        eval_start = metadata["stop_end"]
    else:
        fit_fraction = config["dataset"]["fit_fraction"]
        data = load_channel(job["channel"], lookback, fit_fraction,
                            config["dataset"]["validation_end_fraction"])
        xfit, yfit, xstop, ystop = data.x_fit, data.y_fit, data.x_val, data.y_val
        xe, ye, pe = xstop, ystop, data.persistence_val
        metadata = dict(fit_end=data.fit_end, stop_end=data.val_end,
                        target_mean=data.target_mean, target_scale=data.target_scale, num_values=data.num_values)
        mission = data.mission
        eval_start = data.fit_end
    if mission != job["mission"]:
        raise RuntimeError("Mission mismatch")
    graph = build_channel_graph(job["channel"], fit_fraction=fit_fraction)
    xfit = subset_features(xfit, graph.feature_indices)
    xstop = subset_features(xstop, graph.feature_indices)
    xe = subset_features(xe, graph.feature_indices)
    model = MessageGatedGCN(graph.num_nodes, spec["hidden_by_mission"][mission], graph.adjacency,
                           spec["kernel_size"], spec["dilations"], job["gate_mode"], job["remove_messages"])
    _verify_initial_persistence(model, xfit[:min(8, len(xfit))])
    model, stop_mse, best_epoch, epochs, elapsed = _train_with_early_stopping(
        model, xfit, yfit, xstop, ystop, job["seed"], spec["learning_rate"], common)
    loader = make_loader(xe, ye, common["batch_size"], False, job["seed"])
    metric = evaluate_mse(model, loader, torch.nn.MSELoss(), torch.device("cuda"))
    gates = []
    with torch.no_grad():
        model.eval()
        for xb, _ in loader:
            _, gate = model.forward_with_gate(xb.to("cuda"))
            gates.append(gate.cpu())
    gate_trace = torch.cat(gates)
    g = gate_trace.numpy()
    result = dict(experiment_id=config["experiment_id"], **job, **metadata,
        gate_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith("gate_")),
        parameters=sum(p.numel() for p in model.parameters()),
        lookback=lookback, receptive_field=model.receptive_field,
        hidden_size=spec["hidden_by_mission"][mission], learning_rate=spec["learning_rate"],
        graph_num_nodes=graph.num_nodes, graph_num_edges=graph.num_edges,
        graph_features=graph.feature_indices.tolist(), graph_sources=graph.source_original_columns.tolist(),
        fit_samples=len(yfit), stop_samples=len(ystop), assessment_samples=len(ye),
        best_epoch=best_epoch, epochs_ran=epochs, hit_epoch_ceiling=epochs>=common["max_epochs"],
        best_at_ceiling=best_epoch>=common["max_epochs"], early_stop_mse=stop_mse,
        validation_mse=metric if job["stage"] != "robust" else None,
        assessment_mse=metric if job["stage"] == "robust" else None,
        persistence_mse=float(np.mean((pe-ye)**2)), elapsed_seconds=elapsed,
        gate_mean=float(g.mean()), gate_std=float(g.std()), gate_p05=float(np.quantile(g,.05)),
        gate_p95=float(np.quantile(g,.95)), gate_fraction_below_01=float(np.mean(g<.1)),
        gate_fraction_above_19=float(np.mean(g>1.9)), gate_diagnostic_target_start=eval_start)
    return model, result, gate_trace
