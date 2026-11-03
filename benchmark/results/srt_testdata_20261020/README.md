# Spasmi benchmark on SRT testdata — 2026-10-20

The [script](../../srt_testdata_prism.jl) imports **Spasmi only** as its
solver package. It reads SRT's existing testdata but does not call SRT code.
All benchmark files and the Julia dependency environment are under
`/home/tobias-grasberger`.

## Exact setup

- Data: `sparse-reconstruction-toolkit/testdata/A.jld2` and
  `y_noisy_SNR30.jld2`; $A$ is 5,475 × 6,000 Float64 with 5,662,385
  stored nonzeros (17.237% density).
- Problem: nonnegative LASSO with zero prior,
  $\tfrac12\|Ax-b\|_2^2+\lambda\|x\|_1$ and
  $\lambda=2.84\times10^{-7}$.
- FISTA: identical power-estimated step $t=127.87471066504979$ across
  CPU dense, CPU sparse, GPU dense, and GPU CSR. Gradient-mapping stop is
  set to $a=r=0$ with a check at the final iteration, to force a common
  iteration count. Each result still gets a final KKT evaluation.
- ADMM: CPU dense only, $\rho=1/t=0.007820154546580851$; zero KKT
  tolerance and a check at the final iteration.
- Timing: 40 and 200 iterations, three repeated solves per case; the
  median is reported. A full untimed solve warms compilation before
  timing. Each solve resets to zero. GPU timings synchronize. Julia used
  one thread; CPU BLAS used 32 threads. Solver initialization and host/device
  conversion are excluded from the solve times below (the script reports
  them separately).
- CUDA ordinals on this host: 0 = NVIDIA RTX 6000 Ada Generation;
  1 = NVIDIA RTX PRO 6000 Blackwell Max-Q. CUDA was installed only in
  `~/.cache/prism-bench-RoR2tH`, not added to Spasmi's dependencies.

## Median solve time

| Spasmi solver / storage | 40 iterations | 200 iterations |
| --- | ---: | ---: |
| CPU dense FISTA (Ada run) | 130.1 ms | 486.4 ms |
| CPU sparse FISTA (Ada run) | 315.2 ms | 1,534.4 ms |
| GPU dense FISTA, Ada | 26.57 ms | 126.83 ms |
| GPU CSR FISTA, Ada | 9.35 ms | 41.16 ms |
| CPU dense ADMM (Ada run) | 810.8 ms | 3,921.2 ms |
| CPU dense FISTA (Blackwell run) | 124.1 ms | 509.6 ms |
| CPU sparse FISTA (Blackwell run) | 312.1 ms | 1,675.5 ms |
| GPU dense FISTA, Blackwell | 12.20 ms | 57.77 ms |
| GPU CSR FISTA, Blackwell | 5.11 ms | 23.46 ms |
| CPU dense ADMM (Blackwell run) | 803.2 ms | 4,046.5 ms |

At 200 iterations, GPU CSR is about 11.8× faster than CPU dense FISTA on
Ada and 21.7× faster on Blackwell, using the respective CPU measurements.
CSR is about 3.1× faster than dense GPU on Ada and 2.5× on Blackwell.

The FISTA variants give the same predicted observations to relative
error about $10^{-15}$ on both GPUs. At 200 iterations, their final
KKT/$\lambda$ is 0.20966 and residual norm is 0.00219241. CPU ADMM's
KKT/$\lambda$ is 3.07073 and residual norm is 0.00230907; its prediction
differs from CPU dense FISTA by 0.128 relative to $\|b\|_2$.
**None of these fixed-iteration runs converged**, and equal iteration
counts are not equal accuracy across FISTA and ADMM. This is a throughput
comparison, not time-to-solution or a claim that ADMM is intrinsically
slower at a matched error. GPU ADMM is not available in Spasmi.

For these runs, initial dense CPU→GPU transfer took about 0.13 s and
CPU sparse→GPU CSR conversion about 0.8 s. These one-time costs matter
for a single short solve but can be amortized over a regularization path.
The CUDA runtime emitted warnings about loading cuSPARSE and nvJitLink
from the system CUDA installation; the solves and CPU/GPU parity tests
completed successfully.

## Historical time to the gradient-mapping FISTA stop

The archived run of the [convergence script](../../srt_testdata_convergence.jl)
used `GradientMappingStop(atol=0, rtol=0.001, check_every=1, maxiter=10000)`
on the same SNR30 nonnegative LASSO, with the same $\lambda$, matrix, and
power-estimated step as above. The only solver variants are CPU dense FISTA,
GPU dense FISTA, and GPU CSR FISTA. Three zero-started solves per variant
follow a one-iteration compilation warmup; median wall times exclude data
transfer and initialization, and GPU runs are synchronized.

| Device | CPU dense | GPU dense | GPU CSR |
| --- | ---: | ---: | ---: |
| Ada run | 5.366 s | 1.496 s | **0.569 s** |
| Blackwell run | 5.231 s | 0.777 s | **0.391 s** |

These archived runs used the then-default gradient-mapping criterion, not
Spasmi's current `DualGapStop` default. All variants stopped after
**2,118 iterations**. CPU and GPU predictions agree to approximately
$4\times10^{-15}$ relative to $\|b\|_2$. Their final objective is
$2.92796736\times10^{-6}$, residual norm is 0.00218756, and
returned-point KKT/$\lambda$ is **0.003785**. The last number is a
diagnostic: gradient-mapping convergence does **not** imply
KKT/$\lambda\leq0.001$. It is below Spasmi's 0.1 warning threshold.
GPU CSR was about 9.4× faster than CPU dense FISTA on Ada and 13.4×
faster on Blackwell. This is time to the chosen *mapping* stop, not to
a stricter returned-point KKT stop.

## Reproduce

```bash
cd /home/tobias-grasberger/prism
PRISM_SRT_ITERATIONS=200 PRISM_SRT_GPU_DEVICE=1 \
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=32 \
julia --project=/home/tobias-grasberger/.cache/prism-bench-RoR2tH \
     benchmark/srt_testdata_prism.jl
```

Use `PRISM_SRT_GPU_DEVICE=0` for Ada and
`PRISM_SRT_ITERATIONS=40` for the short fixed-iteration run. For the
historical gradient-mapping setup, replace the script name in the
command with `benchmark/srt_testdata_convergence.jl` (the iteration override
is ignored). That script now inherits the package default; explicitly pass
`GradientMappingStop(atol=0, rtol=0.001, check_every=1, maxiter=10000)`
to select the archived stopping-policy parameters. The corrected current
nonnegative mapping scale can still make results differ from this old run.
