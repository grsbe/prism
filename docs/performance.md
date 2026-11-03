# Performance: defaults, costs, and trade-offs

Prism should do the usual **safe, reusable** optimizations automatically.
Performance still depends on matrix shape, conditioning, sparsity, device,
and the chosen numerical accuracy. This page separates existing defaults
from choices that carry a real cost.

## What is already done by default

| Behavior | Benefit | Qualification |
| --- | --- | --- |
| Reusable FISTA workspaces and in-place gradient and prox buffers | Avoid most per-iteration allocations | Construct `initialize(problem, method)` once for repeated solves |
| Warm starts on an automatic or supplied $\lambda$ path | Often saves iterations | Best used for a descending path; a poor previous solution can also hurt |
| FISTA update gate, then returned-point dual gap | Skips expensive certificate checks while updates are large | A check needs extra products; older gradient mapping remains opt-in |
| FISTA `step=:auto` | Avoids a full SVD for large matrices | Power estimate with a 1.2 safety factor is **not certified** |
| Cached ADMM factorization; smaller $AA^\top$ system when wide | Avoids repeated factorizations and huge $n\times n$ factors | Factorization still uses memory; changing $\rho$ needs a new solver |
| Owned result snapshots and optional path streaming | Safe warm starts; bounded retained path memory with `store=false` | Owning each emitted result still copies its coefficients |

For small dense CPU matrices (at most 250,000 elements),
`step=:auto` uses the exact spectral norm. For larger or device
matrices it uses 50 power iterations and a 1.2 safety factor. Use
`:frobenius` when a provably conservative step is more important
than speed; it may be much smaller than necessary. A known safe
positive numeric step can be passed directly.

## Why a returned-point KKT check needs more products

One FISTA iteration forms

$$
g(y_k)=A^\top(Ay_k-b),\qquad
x_{k+1}=\operatorname{prox}(y_k-tg(y_k)).
$$

Thus the gradient at **$y_k$** is already available. SRT's older
optimality test uses it via the proximal-gradient mapping

$$
G_t(y_k)=\frac{y_k-x_{k+1}}{t}.
$$

`GradientMappingStop` reuses the just-computed gradient and prox.
It requires no extra $A$ or $A^\top$ product at a check.

Opt-in `KKTStop` and the default dual-gap check verify coefficients being
returned, **$x_{k+1}$**, during iteration. Their gradient is

$$
g(x_{k+1})=g(y_k)+A^\top A(x_{k+1}-y_k).
$$

Unless $x_{k+1}=y_k$, this is not the gradient already in hand.
Computing $A^\top A(x_{k+1}-y_k)$ still needs a forward and adjoint
product; storing $A^\top A$ instead can cost $O(n^2)$ memory. Prism does not build that Gram matrix. KKT checks therefore cost
roughly one additional forward/adjoint pair each time they run.
Every-ten-iteration checking amortizes this cost with `KKTStop`, but a strict KKT
criterion can also require **more iterations**, often the larger
effect in highly correlated inversions.

The default `DualGapStop` first tests the cheap coefficient update.
Only when it is small does it compute a returned-point feasible dual gap,
usually requiring another forward/adjoint product. The older mapping rule
remains opt-in and is cheaper, but mapping-stop convergence and
dual-gap convergence are not numerically equivalent. See
[optimality](optimality.md).

## CPU parallelism

Julia threads and BLAS threads are different. Dense matrix-vector
work is often BLAS-bound; setting both to a large count can oversubscribe
the machine. Prism does **not** silently set BLAS to 32 threads:
that would be an unwanted resource decision on a shared server.

Benchmark your actual matrix, including initialization and path
solves, with a few thread counts; for example:

```julia
using LinearAlgebra
BLAS.set_num_threads(8)  # try 1, 8, 16, 32 on an approved machine
```

On this workstation 32 BLAS threads can be appropriate, but do not
assume a 32x speedup. For small matrices, launch and scheduling
overhead can dominate. For large solves, check both wall time and
memory bandwidth. Set thread counts before solver initialization
when measuring end-to-end time.

## GPU use

CUDA is an optional dependency. Put both $A$ and $b$ on the device
before creating `LassoProblem`; FISTA uses generic matrix-vector
and broadcast operations. Keeping a long $\lambda$ path on the GPU
avoids repeated host/device transfers. Moving each result back to
the CPU or checking diagnostics too frequently can synchronize
the device and erode the speedup. The final KKT check also contains
reductions and may synchronize.

`solve_path!` keeps the same GPU-resident solver, matrix, observations, and
workspaces throughout the path; it does not transfer $A$ at each lambda.
With `store=true`, every path point retains its own GPU coefficient snapshot,
so GPU memory grows with path length. With `store=false`, the callback still
receives one owned result snapshot per point, but Prism does not retain it
after the callback. Transfer or persist only the points you need.

Block LASSO–ridge FISTA also supports GPU dense and CSR inputs, with the
L1 and ridge blocks independently signed or nonnegative. The block operator
keeps both matrices and all workspaces on the device, including across a
warm-started path. It avoids materializing a combined GPU matrix, saving
matrix storage at the cost of two matrix-vector products and a few extra
device kernels per iteration. Benchmark for the actual matrix dimensions.

Group-LASSO CUDA FISTA supports dense and CSR designs and noncontiguous
partitions without scalar indexing. Its current group proximal step loops
over groups and performs a reduction per group, so **many tiny groups may be
slow on GPU** despite a fast matrix multiply. Benchmark with your actual
group count before choosing CPU versus GPU; a segmented GPU kernel would be
a future optimization.

The current optional CUDA tests include the NVIDIA RTX PRO 6000
Blackwell. GPU speedup is workload-dependent; the included
`benchmark/gpu_fista.jl` compares fixed-iteration CPU/GPU runs and
reports the achieved KKT ratio. It is a benchmark, not a claim that
either fixed-iteration solution has converged. The real-data
[`srt_testdata_prism.jl`](../benchmark/srt_testdata_prism.jl)
benchmark includes GPU CSR and GPU dense FISTA, CPU FISTA, and CPU ADMM.
See its [results](../benchmark/results/srt_testdata_20261020/README.md).
GPU ADMM is not implemented.

## Measure the right thing

Benchmark at least these separately: initialization, one cold
$\lambda$ solve, and a warm-started path. Warm Julia compilation
before timing repeated calls. Hold the matrix, step, tolerance,
thread/device settings, and $\lambda$ sequence fixed when comparing
solvers. Report both wall time and numerical quality
(`converged`, iterations, KKT/λ, prediction and objective).
A faster solve that stopped at a different accuracy is not a
like-for-like comparison.

Memory can dominate: a dense $m\times n$ `Float64` matrix alone
uses about $8mn$ bytes; copying it or materializing a large
dictionary can cost more than solver workspace. Use
`solve_path!(...; store=false, on_result=...)` when only selected
path points or checkpoint files are needed.

Future work includes reducing the extra final-result diagnostic pass,
systematic Blackwell/CPU benchmarks on representative ill-conditioned
designs, and optional on-disk path resume. These are **not** claimed as
existing defaults.
