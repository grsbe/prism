# Signed SNR30 path: dual gap versus earlier gradient mapping

[Reproduction script](../../../srt_testdata_signed_stop_path.jl). Results are in [`precise_dual_gap/`](precise_dual_gap/) and [`earlier_gradient_mapping/`](earlier_gradient_mapping/), each with `path_diagnostics.csv` and `coefficients.jld2`.

## Exact setup

- SRT `testdata/A.jld2`, `y_noisy_SNR30.jld2`, `target.jld2`, and `L2_norm_of_noise_SNR30.jld2`; 5,475 observations × 6,000 coefficients. **Signed** LASSO, zero prior: $\min_x\tfrac12\|Ax-b\|_2^2+\lambda\|x\|_1$. The known target is used only for evaluation, not fitting.
- Blackwell GPU only, CSR storage for both paths; CUDA device 1 in CUDA.jl's ordering, one Julia thread, `OPENBLAS_NUM_THREADS=32`. Shared power-estimated FISTA step $127.87471066504985$, warm starts, 10,000 maximum iterations per point.
- New rule: `DualGapStop(tol=1e-4, check_every=1, maxiter=10000)`, requiring both $\|x_k-x_{k-1}\|_\infty\leq10^{-4}\|x_k\|_\infty$ and dual gap $\leq10^{-4}\|b\|_2^2$. Earlier rule: `GradientMappingStop(atol=0, rtol=1e-3, check_every=1, maxiter=10000)`. Both paths also report returned-point KKT/λ and dual gap divided by the *new* threshold.
- For signed LASSO, $\lambda_{\max}=\|A^\top b\|_\infty=7.258878902438508\times10^{-5}$. The new-rule path defines the common grid: 30 log-spaced points from $\lambda_{\max}$ to $10^{-4}\lambda_{\max}$, then the same spacing until three successive relative RMSE gains below 1%, $10^{-8}\lambda_{\max}$, or a failed solve. The earlier rule replays exactly this grid, including any nonconverged tail points.

| Stop | Points | Met own stop | Total iterations | Sum of solve times | Best sampled field error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dual gap | 32 | 31 | 61,509 | 16.92 s | 0.42378 at point 20 |
| Earlier gradient mapping | 32 | 26 | 100,726 | 18.15 s | 0.42333 at point 20 |

Both paths' closest sampled match to the known noise norm is point 19, $\lambda=2.3884\times10^{-7}$; the best known-target field is point 20, $\lambda=1.7385\times10^{-7}$. At point 20, residual/noise is 0.989996 (dual gap) versus 0.990003 (earlier); the fields are nearly the same. Below that, accurate optimization of the *signed* problem does **not** stabilize the field: dual-gap field error reaches 0.646 at point 25, 2.258 at point 31, and 3.040 at the nonconverged point 32. These tail fields strongly overfit despite falling RMSE.

The signed case reverses the simple nonnegative-path timing story: the earlier gradient-mapping rule takes *more* iterations near and below the useful λ range. At point 20 it takes 1,455 iterations versus 757; at point 25, 7,225 versus 3,190. It hits the 10,000-iteration limit at points 27–32. The dual-gap rule passes through point 31 but hits the limit at point 32. A failed point is saved for diagnosis, **not** certified as converged. First-call compilation is included in solve times, and the rules have different certificates and tolerances, so the totals are not a matched-accuracy speed comparison.

Important caveat: dual-gap convergence does not ensure small KKT/λ at tiny λ. At point 25 its dual gap is 0.38 of the permitted threshold, yet KKT/λ is 1.99; the earlier rule has KKT/λ about 0.001 there. The gap threshold is normalized by $\|b\|^2$, not by λ, so neither stop is uniformly “more precise” for every coefficient diagnostic. The earlier rule's failure at points 27–31 also does not imply those fields fail the *dual-gap* criterion.

Run from `/home/tobias-grasberger/prism`:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=32 PRISM_SRT_GPU_DEVICE=1 \
julia --project=/home/tobias-grasberger/.cache/prism-bench-RoR2tH \
     benchmark/srt_testdata_signed_stop_path.jl
```
