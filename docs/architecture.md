# Architecture and migration boundary

The package separates the mathematical problem, numerical method, and public
interface. `src/api.jl` is the new entry point. `src/compatibility_api.jl` is
the only place that exposes SRT-style function signatures; it calls the new
implementations and does not contain duplicate iteration loops.

```text
src/
  Prism.jl                includes and exports
  loss.jl                  data losses (least squares, Huber)
  penalty.jl               coefficient penalties (L1, Huber) for LASSO/block
  problem.jl               validated mathematical problems
  result.jl                owned results and stopping policies
  optimality.jl            independent diagnostics at returned solutions
  solvers/fista.jl          reusable FISTA, CPU/device-compatible arrays
  solvers/group_fista.jl    disjoint group-LASSO FISTA, CPU and GPU
  solvers/admm.jl           reusable CPU ADMM
  solvers/constrained.jl    CPU L1-ball FISTA/ADMM and projected-gradient check
  solvers/block_fista.jl    sparse-plus-ridge block FISTA, CPU and GPU
  solvers/laplacian_fista.jl CPU L1 + squared-Laplacian FISTA
  solvers/tv_pdhg.jl        CPU L1 + TV PDHG, primal-dual checks
  workflows/paths.jl       lambda paths and per-point callback
  workflows/cross_validation.jl ordered-group LASSO cross-validation
  api.jl                   new problem → method → result interface
  compatibility_api.jl     selected old SRT names
  sampling/gibbs_posterior.jl Gibbs posterior sampler for signed/nonnegative LASSO
```

## Where the remaining SRT methods will go

Each additional objective gets a **problem type** and a compatible method;
there will not be a universal constructor with dozens of optional keywords.
All methods will share the outer result contract, but their optimality
diagnostics may differ.

| Problem | Implementation location | Status |
| --- | --- | --- |
| L1-ball-constrained least squares | `solvers/constrained.jl` | CPU FISTA and ADMM implemented; SRT parity checked |
| Sparse block + smooth ridge block | `solvers/block_fista.jl` | CPU/GPU FISTA and λ path implemented; GPU dense/CSR parity checked |
| L1 + isotropic/anisotropic total variation | `solvers/tv_pdhg.jl` | CPU PDHG implemented with primal-dual checks; SRT parity checked |
| L1 + squared graph-Laplacian smoothing | `solvers/laplacian_fista.jl` | CPU FISTA implemented; SRT parity checked |
| Soft-min robust objective | `solvers/softmin_fista.jl` | Not yet ported |
| Basic LASSO QP via Gurobi | `ext/PrismGurobiExt.jl` | Optional extension implemented |
| Other Gurobi formulations | `ext/PrismGurobiExt.jl` | Not yet ported |

Ordered-group cross-validation for ordinary LASSO is implemented in
`workflows/`;
resampling and other hyperparameter workflows are not. These
workflows do not belong inside the solvers. Posterior sampling at one fixed
λ lives in `sampling/`; it reuses the problem and result types but none of
the solver loops. GPU FISTA is a first-class
implementation of the same
penalized-LASSO problem; CUDA remains optional. Real-device Float32/Float64
parity passed on Blackwell. An optional fixed-iteration benchmark exists, but
larger ill-conditioned workloads, sparse device matrices, and systematic
performance characterization still need validation.

## Migration rules

1. Freeze SRT reference behavior with representative test problems before
   moving each solver. Do not alter the old SRT repository during migration.
2. Port one problem family at a time; compare objective, constraints, and
   solutions. Where a legacy convergence flag was permissive, document the
   deliberate change rather than pretending numerical identity.
3. Add an independent KKT/primal-dual test at the **returned** variable before
   adding each method to the public API.
4. Keep compatibility wrappers small and separately tested. A similar name
   does not promise identical defaults, stopping, or package identity.
5. Extend grouped validation beyond LASSO where needed; add atomic path
   checkpointing and device CI before claiming full SRT replacement.

This is a development package. The package name, UUID, and public interface
should not be advertised as stable until the remaining methods and migration
tests are complete.
