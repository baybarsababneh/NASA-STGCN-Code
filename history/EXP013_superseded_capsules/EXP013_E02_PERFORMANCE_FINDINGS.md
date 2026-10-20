# EXP013 graph performance: current findings

The source confirms your workload hypothesis: each batch encodes 13,824 MSL command histories through five causal convolution layers, twice. Injection leaves command histories identical. Redundancy explains repeated work, but cannot explain the 18-second cost of an individual forward by itself.

The leading backend hypothesis is deterministic dilated convolution dispatch to a slow native fallback. This is not yet verified on your A100. The earlier GPU workflows used the same padding/Conv1d approach with fewer selected nodes, fewer MSL layers, one forward pass, and a different pinned PyTorch/CUDA build. They also retained determinism and disabled TF32.

Implemented a graph-only source correction: dilations 2/4/8/16 use the same kernel weights on explicitly lagged windows, followed by FP32 matrix multiplication. Dilation 1 remains native. All 64 positions, SiLU, both passes, batch 256, objective, Adam, dropout and RNG order are preserved. Removed/fusion controls retain their original execution. No command caching or execution/recovery code was added.

Nine CPU regression tests pass, including two Adam updates across both feature widths and seeds 17/29/43, gradient comparisons, warm optimiser restoration, causality, timing instrumentation parity and exact candidate replay. The maximum graph parameter difference after Adam was 9.19e-7. Native/candidate CPU trajectories are not bitwise identical. An initial broader fusion change exceeded the fixed parameter gate and was rejected; the tolerance was not relaxed. The real 256×65×55 FIT batch also matches frozen default collation byte-for-byte on CPU.

The notebook embeds the benchmark bundle. Copy its code cells into the paused E02 notebook to use the current runtime. Its one 300-second invocation times every convolution, command/telemetry encoding, graph/reference, discriminator and backward region; records native backend selection; compares two full-batch updates and exact candidate replay; and measures allocated/reserved memory. It uses the existing checkpoint as a read-only baseline when available. It never runs an epoch or changes scientific artifacts.

**No A100 speedup has been measured here.** Read BENCHMARK.json before any further epoch. A numerical pass is separate from exact old-trajectory compatibility; the source diff has not been activated in your frozen E02 run. Full instructions and fixed tolerances are in README.md.
