# Implementation plan: posterior sampling for the nonnegative LASSO in Prism

> **SUPERSEDED (2026-11-11). Do not implement this plan.** It is replaced by
> `gibbs_posterior_sampling.md` in this folder, which covers signed and
> nonnegative problems with one Gibbs sampler. This file is kept only as a
> record.

Do this plan **after** `laplace_posterior_sampling.md` (in the same folder) has
been implemented and its tests pass. That first plan adds
`src/sampling/laplace_posterior.jl` with `MALA`, `PosteriorSamples`,
`sample_posterior`, the summary functions and the diagnostics
`_split_rhat` / `_effective_sample_size`. This plan adds a second sampler for
problems with `constraint=:nonnegative` and reuses all of those.

Follow the plan literally. Where it gives code, copy it; it has been run and
checked. If the repository contradicts the plan, stop and report.

## 1. What you are building

For a nonnegative LASSO problem and one fixed λ, draw coefficient vectors from

    p(x | b)  ∝  exp( −( 0.5 * ‖A x − b‖² + λ * sum(x) ) / s2 )     for x ≥ 0,
                 0 otherwise

where `s2` is the noise variance of `b` (default 1, correct for whitened
data). The most probable point of this distribution is the nonnegative LASSO
solution at λ.

The signed sampler from the first plan cannot be used here; its mathematics is
only valid without the constraint. This plan uses a different, simpler method:
a **Gibbs sampler** that updates one coefficient at a time.

## 2. The mathematics

Hold every coefficient fixed except `x_i`. As a function of `x_i` alone, the
density above is a normal distribution cut off at zero:

    x_i | everything else  ~  Normal(mu_i, sigma_i²) restricted to x_i ≥ 0

    d_i     = ‖a_i‖²                                (a_i is column i of A)
    mu_i    = x_i + ( a_i' * (b − A x) − λ ) / d_i
    sigma_i = sqrt( s2 / d_i )

One **sweep** updates `i = 1, 2, …, n` in order, each time drawing a new `x_i`
from that distribution. After each update the vector `b − A x` changes by
`−(new − old) * a_i`, so it is kept up to date instead of being recomputed.
If column `i` is entirely zero (`d_i == 0`), the data say nothing about `x_i`
and it is drawn from the prior: an exponential distribution with mean `s2 / λ`.

There is nothing to tune: no step size, no acceptance rate. One sweep costs
about as much as two matrix-vector products.

Two equivalent bookkeeping forms are used:

- **Residual form** (`m <= n`): keep `r = b − A x` (length m).
- **Gram form** (`m > n`): precompute `K = A'A` once and keep `A'(b − A x)`
  (length n). Cheaper when there are many more observations than coefficients.

## 3. Rules

1. Work only inside `/home/tobias-grasberger/prism`; temporary files in
   `/home/tobias-grasberger/tmp`, never `/tmp`.
2. Never delete files. Only add the new files and make the edits listed below.
   Do not touch anything under `src/solvers/` or `src/workflows/`.
3. No new package dependencies.
4. Do not commit or push.
5. Same code style as the rest of Prism (see rule 5 of the first plan).

## 4. Scope

In scope: `LassoProblem` with `constraint == :nonnegative`,
`loss == LeastSquares()`, `penalty == L1()`, `Matrix{Float64}` and
`Vector{Float64}`, one `λ > 0`.

Throw `ArgumentError` with a clear message for: `constraint == :signed` used
with the Gibbs sampler (message must say to use `MALA()`); Huber loss or
penalty; sparse, GPU or `Float32` data; `λ == 0` (the distribution is not
well defined without the penalty when there are more coefficients than
observations); `noise_variance <= 0`.

## 5. Files

| Action | File |
| --- | --- |
| add | `src/sampling/nonnegative_posterior.jl` |
| add | `test/nonnegative_posterior.jl` |
| edit | `src/sampling/laplace_posterior.jl` (three small edits, section 7.4) |
| edit | `src/Prism.jl` (one `include` line) |
| edit | `test/runtests.jl` (one `include` line) |
| edit | `docs/posterior_sampling.md`, `README.md` |

- `src/Prism.jl`: add `include("sampling/nonnegative_posterior.jl")`
  immediately after `include("sampling/laplace_posterior.jl")`.
- `test/runtests.jl`: add `include("nonnegative_posterior.jl")` as the last
  line.

## 6. Reference code (tested; copy it)

Copy the whole content of
`docs/plans/nonnegative_sampling_reference/kernels.jl` to the top of
`src/sampling/nonnegative_posterior.jl`, unchanged. It defines:

- `_rand_truncated_normal(rng, mu, sigma)`: one draw from a normal
  distribution restricted to `x ≥ 0`. It is exact also far in the tail.
- `_NonnegativeWork`: what is computed once (column norms, optionally `A'A`).
- `_nonnegative_state(p, x)`: the kept vector for the chosen form.
- `_nonnegative_sweep!(rng, p, x, state)`: one sweep.
- `_nonnegative_chain(rng, p, start; warmup, draws, thin)`: one whole chain,
  returning an `n × draws` matrix.

`docs/plans/nonnegative_sampling_reference/kernels_check.jl` is the script that
verified this code against numerical integration; run it with
`julia <file>` to see the expected accuracy (all printed errors are below
0.006).

## 7. Public interface

### 7.1 `Gibbs` configuration

```julia
struct Gibbs
    warmup::Int
    draws::Int
    chains::Int
    thin::Int
end

Gibbs(; warmup=1_000, draws=2_000, chains=4, thin=5)
```

Validate like `MALA`: `warmup >= 0`, `draws >= 4`, `chains >= 1`, `thin >= 1`,
all integers; otherwise `ArgumentError`. `draws` is the number of stored
draws per chain; each chain runs `warmup + draws*thin` sweeps.

### 7.2 `sample_posterior` for `Gibbs`

```julia
sample_posterior(problem::LassoProblem, method::Gibbs, lambda;
                 noise_variance=1.0, x0=nothing,
                 rng=Random.default_rng()) -> PosteriorSamples
```

Body, in order:

1. Validate (section 4); if `x0` is given, check its length and that all
   entries are finite. Negative entries of `x0` are allowed; the chain clips
   them to zero.
2. `work = _NonnegativeWork(problem.A, problem.b, Float64(lambda), Float64(noise_variance))`.
3. `start = isnothing(x0) ? zeros(n) : Vector{Float64}(x0)`.
4. `seeds = rand(rng, UInt64, method.chains)`; chain `k` uses
   `Xoshiro(seeds[k])`. (Same scheme as `MALA`.)
5. For each chain `k`, in a plain `for` loop:
   `x[:, :, k] = _nonnegative_chain(Xoshiro(seeds[k]), work, start; warmup=method.warmup, draws=method.draws, thin=method.thin)`.
6. `rhat` and `ess` per coefficient with `_split_rhat` and
   `_effective_sample_size`, exactly as in the `MALA` method; same warning.
7. Return `PosteriorSamples` with:
   - `x` as above,
   - `log_density = fill(NaN, draws, chains)` (not computed for Gibbs),
   - `lambda`, `noise_variance`,
   - `acceptance_rate = fill(1.0, chains)` (every Gibbs update is accepted),
   - `step_size = fill(NaN, chains)` (there is no step size),
   - `rhat`, `ess`.

### 7.3 Choosing the sampler automatically

Add these in `src/sampling/nonnegative_posterior.jl`:

```julia
_default_sampler(problem::LassoProblem) =
    problem.constraint == :nonnegative ? Gibbs() : MALA()

sample_posterior(problem::LassoProblem, lambda::Real; kwargs...) =
    sample_posterior(problem, _default_sampler(problem), lambda; kwargs...)

sample_posterior(problem::LassoProblem, result::SolveResult; kwargs...) =
    sample_posterior(problem, _default_sampler(problem), result; kwargs...)
```

so that the common call is simply

```julia
problem = LassoProblem(A, b; constraint=:nonnegative)   # whitened A and b
path = solve_path(problem, FISTA())
samples = sample_posterior(problem, path[12])
credible_interval(samples; level=0.9)
```

End the file with `export Gibbs`.

### 7.4 Edits to `src/sampling/laplace_posterior.jl`

1. The convenience method that takes a `SolveResult` is currently restricted
   to `method::MALA`. Change its signature to
   `method::Union{MALA,Gibbs}`. Because `Gibbs` is defined in a file included
   later, move this one method out of `laplace_posterior.jl` and into
   `nonnegative_posterior.jl` (below the definition of `Gibbs`) instead of
   editing it in place.
2. In the `MALA` method, change the error message for a nonnegative problem
   to: `"MALA supports only constraint=:signed; use Gibbs() for nonnegative
   coefficients"`.
3. In `Base.show` for `PosteriorSamples`, print the acceptance-rate range only
   when the step sizes are not all `NaN`.

## 8. Tests (`test/nonnegative_posterior.jl`)

Start with `using Prism, Test, Random, LinearAlgebra, Statistics`, one
`@testset "Nonnegative posterior sampling" begin ... end`, and always pass a
seeded `rng`. The reference values come from numerical integration on a grid,
which needs no extra package. Total run time should be a few seconds.

1. **Validation.** `sample_posterior(problem, Gibbs(), 0.5)` throws
   `ArgumentError` for a `:signed` problem, for `loss=Huber(1.0)`, for
   `penalty=Huber(1.0)`, for sparse `A`, for `Float32` data, for `lambda = 0`
   and for `noise_variance = 0`. `Gibbs(draws=0)` and `Gibbs(thin=0)` throw.
   `sample_posterior(signed_problem, MALA(), 0.5)` still works, and
   `sample_posterior(nonnegative_problem, MALA(), 0.5)` throws with a message
   containing `"Gibbs()"`.

2. **Truncated normal draws.** For each `(mu, sigma)` in
   `(0.7, 0.5), (-0.2, 1.0), (-3.0, 0.4), (-20.0, 1.0), (5.0, 1.0)`: compute
   the exact mean and standard deviation by summing on
   `g = range(0, max(mu, 0) + 12sigma; length=2_000_001)` with weights
   `exp(-(g - mu)^2 / (2sigma^2))` (normalize the weights to sum to one), draw
   200,000 values with `Prism._rand_truncated_normal`, and require the sample
   mean and standard deviation to match within `rtol=0.03`. Also require every
   draw to be `>= 0`.

3. **Two-coefficient posterior against a grid, wide and tall.** For `m = 1`
   and `m = 6`: build `A = randn(rng, m, 2)`, then make the columns correlated
   with `A[:, 2] = 0.9 .* A[:, 1] .+ 0.3 .* randn(rng, m)`;
   `b = A * [1.0, 0.0] .+ 0.4 .* randn(rng, m)`; `lambda = 0.5`,
   `noise_variance = 0.3`. Exact moments: on `g = range(0, 12; length=1201)`
   evaluate `logd[u, v] = -(sum(abs2, A*[u, v] - b)/2 + lambda*(u + v)) / noise_variance`,
   subtract the maximum, exponentiate, normalize, and compute the mean and
   standard deviation of each coordinate. Run
   `Gibbs(warmup=1_000, draws=25_000, chains=2, thin=1)`. Require
   `posterior_mean` and `posterior_std` to match the grid values within
   `atol=0.03`, and `maximum(samples.rhat) < 1.05`. (`m = 1` uses the residual
   form, `m = 6` the Gram form.)

4. **Both forms agree.** `A = randn(rng, 30, 8)`,
   `b = A * max.(randn(rng, 8), 0) + 0.5 .* randn(rng, 30)`, `lambda = 1.0`,
   `noise_variance = 0.25`. Build the Gram-form work object with
   `Prism._NonnegativeWork(A, b, 1.0, 0.25)` and a residual-form one with the
   same fields but `gram_form = false` and `K = zeros(0, 0)`. Run
   `Prism._nonnegative_chain` on each with different seeds,
   `warmup=1_000, draws=50_000, thin=1`. Means and standard deviations of the
   eight coefficients agree within `atol=0.03`.

5. **Zero column.** `A = randn(rng, 5, 3); A[:, 2] .= 0`, `lambda = 0.8`,
   `noise_variance = 0.4`. The posterior mean of coefficient 2 is
   `noise_variance / lambda = 0.5` within `rtol=0.05`
   (`Gibbs(warmup=500, draws=20_000, chains=2, thin=1)`).

6. **Nonnegativity and shapes.** All entries of `samples.x` are `>= 0`;
   `size(samples.x) == (n, draws, chains)`; `acceptance_rate` is all ones;
   `step_size` and `log_density` are all `NaN`.

7. **Reproducibility.** Same seed gives identical `x`; a different seed gives
   a different `x`.

8. **Automatic choice and path convenience.** For a nonnegative problem,
   `sample_posterior(problem, solve(problem, FISTA(), 0.5); rng=...)` returns
   a result whose `step_size` is all `NaN` (so Gibbs was used) and whose
   `lambda == 0.5`. For a signed problem the same call returns finite step
   sizes (so MALA was used). Keep these runs short by passing the method
   explicitly where you need a small `draws`.

9. **Mode consistency (sanity, loose).** For a well-determined problem
   (`m = 200, n = 5`, small noise, true coefficients `[2, 0, 1, 0, 3]`,
   `lambda = 0.5`, `noise_variance = 0.01`), the posterior mean of every
   coefficient is within `0.1` of the nonnegative LASSO solution from
   `solve(problem, FISTA(stop=KKTStop(rtol=1e-6)), 0.5)`.

Tests 2, 3 and 4 prove the sampler is correct. They are not optional, and
their tolerances must not be loosened: in the reference runs the largest error
was 0.006.

## 9. Mistakes to avoid

- Draw `x_i` from the normal restricted to `x_i ≥ 0`. Do not draw from the
  unrestricted normal and clip negative values to zero; that gives a wrong
  distribution with a spike at zero.
- Use the **current** `x` and the **current** kept vector for every coordinate
  in the sweep (each update must see the updates before it). Do not compute
  all `mu_i` first and then update all coefficients.
- `mu_i` contains `− λ / d_i`, not `− λ / s2`. `s2` enters only through
  `sigma_i = sqrt(s2 / d_i)`.
- Keep the periodic recomputation of the kept vector (every 100 sweeps) that
  is in the reference code; it stops rounding errors from accumulating.
- Do not add an acceptance step. Gibbs updates are always accepted.
- Posterior draws are strictly positive, never exactly zero. Do not describe
  the posterior mean as sparse.

## 10. Documentation

In `docs/posterior_sampling.md` add a section "Nonnegative coefficients" that
states: the density from section 1; that `Gibbs()` is chosen automatically for
`constraint=:nonnegative`; that there is nothing to tune; the cost (measured on
the development machine: about 0.004 s per sweep for m = 2000, n = 6000 and
0.02 s for m = 5000, n = 10000, so the default 11,000 sweeps per chain take
under a minute and about four minutes respectively); that strongly correlated
columns slow mixing, so `rhat` and `ess` must be checked and `thin`/`draws`
increased if they are poor; and that, as in the signed case, the intervals are
conditional on the chosen λ and on `noise_variance`.

Update the limits paragraph of that file (it currently says signed only), and
in `README.md` change the table row added by the first plan to
`| Posterior samples for signed or nonnegative LASSO at one λ | MALA (signed), Gibbs (nonnegative) | Dense CPU, Float64 |`
and use the nonnegative example from 7.3 in the "Posterior uncertainty"
section.

## 11. Definition of done

- [ ] The files in section 5 are added or edited and nothing else changed.
- [ ] `julia --project=. -e 'using Pkg; Pkg.test()'` passes, including both
      sampling test files.
- [ ] `sample_posterior(problem, path_result)` works for a nonnegative problem
      without naming a method.
- [ ] No new dependencies.
- [ ] Report back: the test summary, the run time of the new test file, and
      the largest `rhat` and smallest `ess` seen in test 3.
