# Implementation plan: one shared FISTA loop and six fixes in Prism

This plan has two parts. Part A replaces five copies of the FISTA iteration
with one shared loop, without changing any numerical result. Part B fixes six
defects found in a code review. Do Part A first, then Part B.

Every code change in this plan already exists as a finished, tested file or
patch in `docs/plans/prism_fixes_reference/`. Your job is to apply them in the
right order and prove, with the checks given here, that the result is correct.
Do not rewrite the reference code in your own words. If a check fails, stop and
report the output; do not "fix" it by changing a tolerance or a test.

## 1. Rules

1. Work only inside `/home/tobias-grasberger/prism`. Temporary files go in
   `/home/tobias-grasberger/tmp`, never in `/tmp`.
2. Never delete a file. When this plan says "replace a file", first copy the
   current version to `/home/tobias-grasberger/tmp/prism_backup_<date>/` with
   the same relative path, then overwrite it.
3. Do not commit or push.
4. Do not add package dependencies.
5. This plan is independent of the posterior-sampling plans in the same
   folder. If `src/Prism.jl` already contains an `include("sampling/...")`
   line, keep it.

## 2. Before you change anything

Run these three commands from `/home/tobias-grasberger/prism` and keep the
output.

```
mkdir -p /home/tobias-grasberger/tmp/prism_golden
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. docs/plans/prism_fixes_reference/fista_golden.jl /home/tobias-grasberger/tmp/prism_golden/before.bin
```

The test suite must pass. The second command prints
`recorded 2368 cases to ...`. That file is the exact behaviour (coefficients,
iteration counts, stop reasons, diagnostics) of every FISTA-based solver on
2,368 fixed cases. It is the reference for Part A. Record it on the same
machine you will use for the rest of the plan, because bit-exact results
depend on the machine's linear-algebra library.

## Part A. One shared FISTA loop

### A.1 Why

The same loop (gradient step, proximal step, momentum, three stopping rules) is
written out separately in `src/solvers/fista.jl`, `group_fista.jl`,
`block_fista.jl`, `laplacian_fista.jl` and `constrained.jl`. The copies have
already drifted apart in small ways, and every change to the iteration has to
be made five times.

### A.2 The design (for your understanding; the code is already written)

A new file `src/solvers/fista_core.jl` contains the loop once, as
`_fista_solve!(solver, lambda; warm_start)`, plus the three stopping checks.
Each solver type supplies only what is specific to it, through small functions
dispatched on the solver type:

| Function | Meaning |
| --- | --- |
| `_fista_workspace(solver)` | the `FISTASolver` that holds the work arrays and the stop rule |
| `_fista_step(solver)` | step size (default: the workspace step) |
| `_fista_gradient!(solver, y)` | gradient of the smooth part at `y` |
| `_fista_prox!(solver, step, lambda)` | proximal or projection step |
| `_fista_kkt(solver, lambda)` | optimality residual at the current point |
| `_fista_gap(solver, lambda)` | primal-dual gap at the current point |
| `_fista_gap_tolerance(solver, stop)` | tolerance for the gap |
| `_fista_kkt_scale(solver)` | scale for the zero-penalty fallback test |
| `_fista_gap_degenerate(solver, lambda)` | whether the gap test is unusable (default: `lambda == 0`) |
| `_fista_degenerate_reason(solver)` | stop reason reported in that case |
| `_fista_mapping_scale(solver, lambda, step)` | scale for the gradient-mapping stop |
| `_fista_result(solver, lambda, converged, reason, iterations, diagnostics)` | build the result object |

The public `solve!` methods keep their names, arguments and results. The only
structural change visible outside is that `ConstrainedFISTASolver` gains one
field, `gradient_scale`, computed once in its constructor.

### A.3 Steps

1. Confirm that the files you are about to overwrite are unchanged since this
   plan was written, so that you do not discard someone else's work:
   `git status --short src/` must print nothing, and
   `git log -1 --format=%h` must print `21edc42`. If either differs, stop and
   report.
2. Back up `src/Prism.jl` and the five solver files named in A.1 (rule 2).
3. Copy these six files from
   `docs/plans/prism_fixes_reference/refactored_src/solvers/` to
   `src/solvers/`, overwriting the five existing ones and adding the new one:
   `fista_core.jl`, `fista.jl`, `group_fista.jl`, `block_fista.jl`,
   `laplacian_fista.jl`, `constrained.jl`.
4. In `src/Prism.jl`, add the line `include("solvers/fista_core.jl")`
   immediately **before** the existing line `include("solvers/fista.jl")`.
   Change nothing else in that file. (Do not copy
   `refactored_src/Prism.jl` over it; that copy is only there to show the
   result.)

### A.4 Checks for Part A (all three are required)

```
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. docs/plans/prism_fixes_reference/fista_golden.jl /home/tobias-grasberger/tmp/prism_golden/after_refactor.bin
julia docs/plans/prism_fixes_reference/fista_golden_compare.jl /home/tobias-grasberger/tmp/prism_golden/before.bin /home/tobias-grasberger/tmp/prism_golden/after_refactor.bin
```

- The test suite passes.
- The compare script prints
  `IDENTICAL: all 2368 cases match bit for bit`. Anything else means the
  refactor changed behaviour: stop and report the mismatching case names it
  prints.
- `grep -c "next_t = " src/solvers/*.jl` shows the momentum update in
  `fista_core.jl` only (count 1 there, 0 in the other FISTA files).

If a functional CUDA device is available, also run the optional GPU tests the
way their header comments describe (`test/optional_cuda.jl`,
`optional_cuda_group.jl`, `optional_cuda_block.jl`, `optional_cuda_huber.jl`).
They must pass. If no device is available, say so in your report; do not skip
this silently.

## Part B. Six fixes

All six are in one patch, `docs/plans/prism_fixes_reference/fixes.diff`, made
against the Part A result. Apply it from the repository root:

```
patch -p1 --dry-run < docs/plans/prism_fixes_reference/fixes.diff
patch -p1 < docs/plans/prism_fixes_reference/fixes.diff
```

The dry run must report six files patched with no rejects before you run the
real command. Back up the six files first (rule 2); the dry run lists them.

What the patch changes, and why:

| # | File | Defect | Fix |
| --- | --- | --- | --- |
| 1 | `src/solvers/tv_pdhg.jl` (`_tv_converged`) | With L1 penalty `lambda == 0` (the default of `solve(::TVProblem, ::PDHG)`), the stationarity tolerance was an absolute `1e-10` and the solver never reported convergence. | Use the TV weight as the scale when `lambda == 0`, and the data gradient at zero when both are zero. Behaviour for `lambda > 0` is unchanged. |
| 2 | `src/solvers/admm.jl` (`_admm_optimality`) | With `DualGapStop` and `lambda == 0`, ADMM used an absolute tolerance of `1e-10`: never satisfied for large-scale data, satisfied too early for tiny-scale data. | Tolerance is `stop.tol * norm(A'b, Inf)`, relative to the data. |
| 3 | `src/solvers/block_fista.jl` | `BlockDesign` is a matrix type without `getindex`, so printing a block solver built from sparse or GPU matrices threw `CanonicalIndexError`. | Add `Base.show` methods that print a one-line summary. |
| 4 | `src/solvers/constrained.jl` (`_project_l1_ball!`) | For long vectors, rounding could leave the projection slightly outside the L1 ball, and the solver's own feasibility check then threw an `ArgumentError` mid-solve. | If the result exceeds the radius by more than a few rounding units, rescale it onto the ball. |
| 5 | `src/workflows/cross_validation.jl` | `cross_validate` with the Huber penalty failed deep inside with a message about `lambda_grid`. | Reject it at the top with a message that explains why. |
| 6 | `src/solvers/fista.jl`, `tv_pdhg.jl`, `constrained.jl` | The TV solver and the constrained ADMM solver always computed a full singular value decomposition for the step size, which is slow for large matrices. | New helper `_squared_norm_bound` applies the rule FISTA already uses: exact for dense matrices up to 250,000 elements, power estimate above. `PDHG(step=:exact)` still forces the exact value. |

Then add the tests:

1. Copy `docs/plans/prism_fixes_reference/review_fixes_test.jl` to
   `test/review_fixes.jl`.
2. Add `include("review_fixes.jl")` as the last line of `test/runtests.jl`.
3. In the docstring of `cross_validate` in `src/workflows/cross_validation.jl`,
   change the first sentence after the signature from
   "Grouped K-fold CV for a signed or nonnegative LASSO." to
   "Grouped K-fold CV for a signed or nonnegative LASSO with the L1 penalty
   (the Huber penalty is rejected)."
4. In `README.md`, in the "Huber penalty" section, append one sentence:
   "`cross_validate` also rejects it, because its lambda ratios are relative to
   `lambda_max`."

### Checks for Part B

```
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. docs/plans/prism_fixes_reference/fista_golden.jl /home/tobias-grasberger/tmp/prism_golden/after_fixes.bin
julia docs/plans/prism_fixes_reference/fista_golden_compare.jl /home/tobias-grasberger/tmp/prism_golden/before.bin /home/tobias-grasberger/tmp/prism_golden/after_fixes.bin
```

- The suite passes and includes a line `review fixes | 61 61`.
- The compare script again prints
  `IDENTICAL: all 2368 cases match bit for bit`. None of the six fixes is
  supposed to change any FISTA result on those cases.

Useful to know: on the code before Part B, the new test file fails in five of
its six groups (everything except the last, which only measures speed-related
behaviour). If you want to convince yourself the tests are meaningful, run
`julia --project=. test/review_fixes.jl` once before applying the patch.

## 3. Not part of this plan

Do not attempt these; they need a decision from the maintainer first.

- **`converged == true` with a large KKT ratio at the small-λ end of a path.**
  The default `DualGapStop` tolerance is relative to `‖b‖²`, so at very small
  λ a point can pass while its KKT residual is several times λ, and the
  existing KKT warning fires on a "converged" result. Whether to tighten the
  default, add a KKT condition to the stop, or only document it is a design
  choice.
- **Group LASSO on GPU with many groups.** The proximal step and the KKT and
  gap functions loop over groups one at a time, which launches one GPU
  operation per group. This needs profiling on a real device before any
  redesign.
- **Scale floors of the form `max(..., 1)`** in the zero-penalty test of the
  FISTA family and in the constrained and TV tolerances. They assume data of
  order one or larger (true for whitened problems). No failure was reproduced,
  so nothing is changed here.

## 4. Definition of done

- [ ] `git status --short` shows modified: `README.md`, `src/Prism.jl`,
      `src/solvers/{admm,block_fista,constrained,fista,group_fista,laplacian_fista,tv_pdhg}.jl`,
      `src/workflows/cross_validation.jl`, `test/runtests.jl`; added:
      `src/solvers/fista_core.jl`, `test/review_fixes.jl`; plus the
      `docs/plans/` folder. Nothing else.
- [ ] Backups of every overwritten file exist under
      `/home/tobias-grasberger/tmp/prism_backup_<date>/`.
- [ ] The full test suite passes.
- [ ] Both golden comparisons printed `IDENTICAL`.
- [ ] Report back: the three test summaries (before, after Part A, after
      Part B), both compare-script lines, and whether the GPU tests were run.
