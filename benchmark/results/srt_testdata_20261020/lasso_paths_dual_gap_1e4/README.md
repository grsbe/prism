# SRT SNR30 LASSO path with Spasmi's dual-gap default

Reproduced by [the benchmark script](../../../srt_testdata_lasso_path.jl), with outputs in `cpu_dense/`, `gpu_blackwell_dense/`, and `gpu_blackwell_csr/`. Each contains `path_diagnostics.csv` and `coefficients.jld2`. The earlier gradient-mapping path remains in [the sibling folder](../lasso_paths/README.md).

## Configuration

- SRT `testdata/A.jld2` (5,475 × 6,000) and `y_noisy_SNR30.jld2`; nonnegative LASSO, zero prior, objective $\tfrac12\|Ax-b\|_2^2+\lambda\|x\|_1$.
- Spasmi FISTA, common power-estimated step `127.87471066504979`, warm starts, and default `DualGapStop(tol=1e-4, check_every=1, maxiter=10000)`. Convergence requires both $\|x_k-x_{k-1}\|_\infty\leq10^{-4}\|x_k\|_\infty$ and primal–dual gap $\leq10^{-4}\|b\|_2^2$.
- CPU computes 30 log-spaced points from the exact zero-solution $\lambda_{\max}=7.258878902438494\times10^{-5}$ to $10^{-4}\lambda_{\max}$. It continues at the same log spacing until three successive relative RMSE gains are below 1%, or $\lambda/\lambda_{\max}=10^{-8}$. Both GPU variants replay the CPU grid. This fit-plateau rule is a benchmark-path endpoint, not a claim that the emission field is stable.
- Julia one thread, OpenBLAS 32 threads, CUDA device 1 = RTX PRO 6000 Blackwell Max-Q in CUDA's ordering. Summed `solve!` times below include first-call compilation but exclude initialization, transfer, diagnostic CSV/JLD2 writing, and post-solve gap recomputation. Spasmi is loaded from the local checkout through `/home/tobias-grasberger/.cache/prism-bench-RoR2tH`.

| Variant | Points | Summed solve time | Final RMSE | Final KKT/λ | Final gap / threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU dense | 31 | 77.44 s | 2.91687566e-5 | 0.02572 | 0.99574 |
| Blackwell GPU dense | 31 | 13.13 s | 2.91687566e-5 | 0.02572 | 0.99574 |
| Blackwell GPU CSR | 31 | 5.78 s | 2.91687566e-5 | 0.02572 | 0.99574 |

All 31 points in all variants passed the dual-gap stop; the final point is $\lambda/\lambda_{\max}=7.27895384\times10^{-5}$. The maximum gap/threshold ratio over each path was 0.99574. GPU predictions differ from CPU by at most $4.31\times10^{-15}\|b\|_2$. The final KKT/λ is much lower than the old path's 0.611, but dual-gap convergence does not impose a fixed KKT/λ threshold.

At the final matched λ, the new and old coefficient fields differ by 21.97% relative to their larger norm; the new solution has 131 nonzeros versus 193 previously. The synthetic target is known: relative field error rises from 0.4221 on the old, prematurely stopped path to 0.4676 on this more accurate solve. Thus numerical convergence improves, but the smaller-λ fitted field is **not** a better reconstruction of the truth. The RMSE plateau alone is not a safe λ-selection principle.

Reproduce from `/home/tobias-grasberger/prism`:

```bash
PRISM_PATH_OUT_DIR=/home/tobias-grasberger/prism/benchmark/results/srt_testdata_20261020/lasso_paths_dual_gap_1e4 \
PRISM_SRT_GPU_DEVICE=1 PRISM_SRT_BLAS_THREADS=32 \
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=32 \
julia --project=/home/tobias-grasberger/.cache/prism-bench-RoR2tH benchmark/srt_testdata_lasso_path.jl
```
