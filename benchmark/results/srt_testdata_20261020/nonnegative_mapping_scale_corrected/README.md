# Nonnegative LASSO: corrected gradient-mapping scale

[Reproduction script](../../../srt_testdata_nonnegative_mapping_scale_path.jl). The script replays the 31 saved λ values from the [old nonnegative path](../lasso_paths/README.md) and compares with the [dual-gap path](../lasso_paths_dual_gap_1e4/README.md). Per-point diagnostics and coefficients are in this directory. [Path comparison plot](stopping_rule_path_comparison.pdf) (also [PNG](stopping_rule_path_comparison.png)) uses the same λ grid and known-target reference for all three rules. Blue crosses mark points where the corrected rule reached `maxiter`, rather than converging. The plot is reproduced by [field-error preparation](../../../prepare_nonnegative_stop_comparison.jl) and [plotting](../../../plot_nonnegative_stop_comparison.py).

## Setup

- SRT SNR30 data, 5,475 observations × 6,000 coefficients; nonnegative LASSO, zero prior. The known target and noise norm are used only for evaluation.
- Blackwell GPU, CSR matrix, common FISTA step `127.87471066504979`, warm starts, at most 10,000 iterations per λ.
- `GradientMappingStop(atol=0, rtol=1e-3)`. The mapping numerator is unchanged. The scale is now the maximum of λ, active-coordinate absolute gradients, and feasible negative gradients at zero coefficients; positive gradients at zero cannot enlarge the tolerance. The old path used the largest absolute gradient over *all* coefficients. Spasmi's default remains `DualGapStop`.
- Same λ grid, objective, and input as the saved paths. Solve times include first-use compilation, but exclude data transfer, diagnostic recomputation, and file writing; they are not matched-accuracy speed comparisons.

| Rule | Converged / 31 | Total iterations | Summed solve time | Final KKT/λ | Final field error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old gradient-mapping scale | 31 | 11,716 | 2.15 s | 0.611 | 0.4221 |
| Corrected gradient-mapping scale | 28 | 87,834 | 17.36 s | 0.00892 (not converged) | 0.4836 |
| Dual gap (`tol=1e-4`) | 31 | 25,993 | 5.78 s | 0.0257 | 0.4676 |

The correction first changes the result at path point 14 (`λ/λmax≈0.0161`). At point 20, near the best *known-target* field (relative error 0.4201), it needs 1,725 iterations rather than 404 under the old rule, and the coefficient vectors differ by 2.54%. At the last point, they differ by 26.7% despite nearly identical RMSEs. The corrected rule has not met its own tolerance at points 28, 30, and 31; these saved vectors are **not** certified solutions. At point 31 the old path's KKT/λ was 0.611, whereas the corrected run reduced it to 0.00892 before reaching `maxiter`.

At point 25 the relative known-target field errors are 0.414 (old), 0.452 (corrected), and 0.446 (dual gap); at point 31 they are 0.422, 0.484, and 0.468. Thus the path comparison separates **optimization accuracy** from **reconstruction accuracy**: the older early stop happens to regularize this synthetic example.

The time–accuracy trade-off depends strongly on λ. At point 20 (`λ/λmax=0.002395`), cumulative GPU solve time is 1.74 s (old), 3.64 s (corrected), or 2.25 s (dual gap). Corrected versus old coefficients differ by 2.54% relative to the larger coefficient-vector norm, while their predictions differ by only 0.136% of the known noise norm. At point 25 (`λ/λmax=0.0004894`), cumulative times are 1.94, 7.19, and 3.57 s; corrected versus old coefficients differ by 17.6%, yet predictions differ by only 0.692% of the noise norm. At the last point, times are 2.15, 17.36, and 5.78 s; the coefficients differ by 26.7% and predictions by 0.700% of the noise norm. These times include first-use compilation, so their ratios are indicative for this run, not isolated steady-state solver benchmarks.

The old rule's small RMSE and sometimes lower known-target field error in the low-λ tail reflect *early stopping*, not accurate optimization. Stricter convergence need not improve reconstruction under overfitting. The corrected mapping stop is substantially stricter here; the dual-gap default is a different certificate with a different scale and can stop earlier or later depending on λ.

Reproduce from the Spasmi repository:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=32 PRISM_SRT_GPU_DEVICE=1 \
julia --project=/home/tobias-grasberger/.cache/prism-bench-RoR2tH \
  benchmark/srt_testdata_nonnegative_mapping_scale_path.jl
```
