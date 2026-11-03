# Prism.jl

Sparse and smooth reconstruction for linear inverse problems in Julia.
Given observations $b$ and a forward operator $A$, Prism estimates sparse
coefficients $x$ with optional nonnegativity and spatial regularization.
The core choices are explicit: **problem**, **solver**, **stopping rule**,
and **regularization path**.

Prism is a new package under development, not a drop-in replacement for
SparseReconstructionToolkit (SRT). It retains a small, separate compatibility
surface; SRT remains unchanged.

## Install

Prism requires Julia 1.11 or newer. It is not registered yet. Until the
renamed source is pushed, use `Pkg.develop(path="/home/tobias-grasberger/prism")`.
After pushing it, install directly from the renamed GitHub repository:

```julia
using Pkg
Pkg.add(url="https://github.com/grsbe/prism.git")
using Prism
```

In the Julia package REPL, the equivalent is
`pkg> add https://github.com/grsbe/prism.git`. To work on the source
instead, use `Pkg.develop(path="/home/tobias-grasberger/prism")` in your
project environment. From the checkout itself, run
`julia --project=. -e 'using Pkg; Pkg.instantiate(); using Prism'`
to verify it loads.

CPU solvers need no optional packages. For GPU FISTA, add `CUDA` to the
*calling project's* environment (`Pkg.add("CUDA")`), then check
`CUDA.functional()` before moving the matrix and observations to the GPU.
The optional Gurobi backend requires `JuMP`, `Gurobi`, and a working Gurobi
license in that environment.

## What is provided

| Reconstruction problem | Solver | Current support |
| --- | --- | --- |
| Signed or nonnegative penalized LASSO | FISTA, ADMM; optional Gurobi QP | FISTA: dense/sparse CPU and dense/CSR CUDA; ADMM: dense CPU |
| Disjoint group LASSO, signed or nonnegative | FISTA | Dense/sparse CPU and dense/CSR CUDA |
| Signed or nonnegative least squares in an L1 ball | FISTA, ADMM | Dense CPU |
| Sparse block plus smooth ridge block; independent nonnegativity on each block | FISTA | Dense CPU, GPU dense/CSR |
| L1 plus squared graph-Laplacian smoothing | FISTA | Dense CPU |
| L1 plus 2D isotropic or anisotropic total variation | PDHG | Dense CPU |

For group sparsity, provide a partition of coefficient indices; every column
must belong to exactly one nonempty group. For example:

```julia
problem = GroupLassoProblem(A, b, [[1, 2], [3]];
                            constraint=:nonnegative)
result = solve(problem, FISTA(), 0.2)
```

The default group weight is `sqrt(length(group))`; pass `weights` to override
it. This selects whole groups, not individual coefficients within a selected
group. Groups need not be contiguous. Overlapping groups are not supported.
A group path uses the same `lambda_grid` and `solve_path!` workflow as LASSO,
but grouped cross-validation is not yet implemented.

The central problem is

$$
\min_{x\in C}\ \frac12\|Ax-b\|_2^2+\lambda\|x\|_1,
$$

where $C$ is the full space or the nonnegative orthant. The other problem
types replace or augment the regularizer; they do not hide all choices behind
a single constructor. For a block problem, use `BlockLassoRidgeProblem(A, B, b;
x_constraint=:nonnegative, z_constraint=:signed)` (or reverse the two)
to control the L1 and L2 blocks independently. The `constraint` keyword
still sets both blocks together.

A LASSO solver can be reused across a descending
$\lambda$ path with warm starts. Results own their coefficient arrays and
report numerical diagnostics rather than only a vector.

Prism does **not** select a scientifically appropriate $\lambda$, estimate
transport/model error, build footprints, or turn coefficient positivity into
physical field positivity. Form the design, subtract backgrounds or priors,
and apply measurement weighting before creating a problem.

## Start here

```julia
using Prism, LinearAlgebra

A = Matrix{Float64}(I, 3, 3)
b = [2.0, -1.0, 0.5]
problem = LassoProblem(A, b; constraint=:nonnegative)
method = FISTA(step=:auto)  # iterate-change gate, then dual-gap certificate
result = solve(problem, method, 0.2)

@show result.x result.converged result.kkt_ratio

# Reuse workspaces and warm-start a descending path.
solver = initialize(problem, method)
results = solve_path!(solver)  # 30 log-spaced λ, from λmax to 1e-4 λmax
# Or inspect/customize the grid; explicit paths still work:
lambdas = lambda_grid(problem; min_ratio=1e-2, n_lambdas=50)
results = solve_path!(solver, lambdas)
```

The automatic LASSO path computes its full grid by default; a flat RMSE
does not stop it or imply stable coefficients. Choose a λ using a separate
criterion (for example, an uncertainty-based discrepancy), not simply the
last path point. The `1e-4` endpoint is a coverage default, not a
scientific optimum.

`result.converged` means the selected **numerical** stop passed. It does
not validate $A$, $b$, the background, uncertainty model, or choice of
$\lambda$. The default `DualGapStop(tol=1e-4)` first checks whether the largest
coefficient update is at most `tol` times the largest coefficient; only
then does it check a returned-point primal-dual gap against
`tol * norm(b)^2`. The latter matches scikit-learn's normalization for
Prism's unnormalized objective. `KKTStop` and the older
`GradientMappingStop` remain explicit alternatives. For a zero L1 penalty (or zero ridge penalty in the block model), the
FISTA default uses a scaled KKT fallback; ADMM uses an absolute KKT
fallback at zero L1 penalty. No dual gap is claimed in these cases.

For example, choose the older gradient-mapping rule explicitly:

```julia
method = FISTA(stop=GradientMappingStop(rtol=1e-4))
result = solve(problem, method, 0.2)
@show result.stop_reason result.kkt_ratio
```

The optional gradient-mapping rule uses the already-computed gradient at
FISTA's extrapolated point.
It is cheaper to check, but is **not** the exact KKT test at the returned
coefficients. FISTA and ADMM use dual-gap stopping by default where a
certificate is implemented; ADMM also requires primal and dual feasibility.
TV-PDHG keeps its existing primal–dual/KKT checks. Penalized results always
warn when the final returned-point KKT residual exceeds `0.1 * λ`, even if
a dual-gap stop passed. A warning is a diagnostic, not a reversal of
`converged=true`. See [optimality](docs/optimality.md).

For a path too large to retain in memory:

```julia
solve_path!(solver, [2.0, 0.5, 0.2]; store=false,
            on_result=(i, r) -> println(i, " ", r.kkt_ratio))
```

A callback can write each result to your own checkpoint. On-disk resume is
not yet integrated. For a GPU path, the solver keeps its matrix, observations,
and workspaces on the GPU across lambda values; it does not re-upload them at
each point. `store=true` retains a separate coefficient snapshot per point on
the GPU. With `store=false`, `on_result` receives an owned snapshot but the path
does not retain it. Explicit CPU copies still incur a transfer.

For grouped model selection, `ordered_group_folds` keeps whole days (or
other supplied groups) together, and `cross_validate` evaluates shared
`lambda/lambda_max` ratios with a fresh training-only `lambda_max` in each
fold. See [grouped cross-validation](docs/cross_validation.md). CV suggests path
ratios but does not automatically refit on all observations or establish
physical identifiability.

## Choosing a method

Start with FISTA for a large or GPU-resident penalized LASSO problem. ADMM
is a CPU alternative that caches a factorization; for a wide design it
factors the smaller $AA^\top+\rho I$ system. Compare both on your actual
conditioning and memory budget. Use the other problem types only when their
penalties match the scientific model. GPU acceleration is important here,
but CUDA is optional: the caller explicitly places $A$ and $b$ on device.

The initial FISTA step is selected once when the solver is initialized.
`step=:auto` uses an exact spectral norm for small dense CPU matrices and
a power estimate otherwise. The power estimate is fast, **not a certified
upper bound**. For a guaranteed conservative step use `:frobenius`, or
pass a justified positive step. See [performance](docs/performance.md).

## Documentation

- [Overview and reading guide](docs/index.md)
- [Mathematical models and algorithms](docs/mathematics.md)
- [Optimality and stopping](docs/optimality.md)
- [Grouped cross-validation](docs/cross_validation.md)
- [Performance: defaults, costs, and trade-offs](docs/performance.md)
- [Architecture and migration status](docs/architecture.md)

The pages are plain Markdown ready to be used by Documenter.jl later;
a published documentation site has not been configured.

## Status and tests

Prism has tests for analytical solutions, KKT and primal-dual stopping,
correlated columns, warm-start ownership, paths, and the SRT ADMM
false-convergence example. Optional tests compare migrated objectives and
the legacy FISTA stop with SRT, test CUDA on an NVIDIA RTX PRO 6000
Blackwell, and compare against Gurobi. Run core tests from the repository:

```bash
julia --project=. test/runtests.jl
```

Soft-min robust LASSO, SRT's remaining hyperparameter/uncertainty workflows,
and integrated disk checkpoints have not been migrated. Grouped LASSO CV is
available. The Gurobi extension
covers basic penalized LASSO only. No publication license has been chosen.
