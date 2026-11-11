# Implementation plan: posterior sampling for the LASSO in Prism (Gibbs sampler)

> **IMPLEMENTED (2026-11-11).** The sampler is in
> `src/sampling/gibbs_posterior.jl`, tested in `test/gibbs_posterior.jl` and
> documented in `docs/posterior_sampling.md`. This file is kept as a record.
> One deviation: `Pkg.add` writes its own `[compat]` entries, which were then
> set to `Random = "1"` and `SpecialFunctions = "2"`.

This plan replaces two older plans in this folder,
`laplace_posterior_sampling.md` and `nonnegative_posterior_sampling.md`. Do not
implement those. They are kept only as a record.

This plan tells you exactly what to build, in what order, and how to check it.
Follow it literally. Where it gives code, copy the code; it has been run and
checked. Where it gives a rule, do not replace it with your own idea. If
something in the repository contradicts this plan, stop and report the
contradiction instead of guessing.

## 1. What you are building and why

Prism (`/home/tobias-grasberger/prism`, a Julia package) solves

    minimize over x:   0.5 * ‖A x − b‖²  +  λ * ‖x‖₁      (optionally with x ≥ 0)

and returns one coefficient vector per λ. That vector is a point estimate with
no uncertainty. You will add a sampler that, for **one fixed λ**, draws
coefficient vectors from the Bayesian posterior whose most probable point is
that same LASSO solution. From the draws the user gets a posterior mean,
standard deviation and credible interval for every coefficient.

The idea of sampling this posterior comes from a colleague's research code in
`/home/tobias-grasberger/sparse-inversion-sampling`. **Do not copy that code,
its method or its dependencies.** It uses a different algorithm (MALA on a
scale mixture) that was measured to be far slower than the method in this plan
and that cannot handle `x ≥ 0`. This plan is self-contained.

## 2. Rules that apply to every step

1. Work only inside `/home/tobias-grasberger/prism`. Put temporary files in
   `/home/tobias-grasberger/tmp`, never in `/tmp`.
2. Never delete files. Only add the new files and make the edits listed in
   section 6. Do not modify any file under `src/solvers/` or `src/workflows/`.
3. The only new package dependencies are the standard library `Random` and the
   package `SpecialFunctions`. Nothing else: no Distributions, no AdvancedMH,
   no MCMCChains, no Turing.
4. Naming: never use the word "Laplacian" in a new name, docstring or file
   name. Prism already uses it for something unrelated (graph-Laplacian
   smoothing in `src/solvers/laplacian_fista.jl`).
5. Match the existing code style: four-space indent, docstrings in triple
   quotes above definitions, internal helpers prefixed with `_`, argument
   validation that throws `ArgumentError` or `DimensionMismatch` with a clear
   message, `export` statements at the bottom of the file that defines the
   names.
6. Do not commit or push. Leave the changes in the working tree. The working
   tree already contains other uncommitted changes (a FISTA refactor and
   fixes); leave them exactly as they are.
7. Run the full test suite before you start and after you finish:
   `cd /home/tobias-grasberger/prism && julia --project=. -e 'using Pkg; Pkg.test()'`.
   It passes today. It must still pass at the end.

## 3. The mathematics (everything you need)

### 3.1 The posterior

Inputs: matrix `A` (m rows, n columns), vector `b` (length m), penalty `λ > 0`,
and the noise variance `s2 > 0` of each entry of `b` (default `1`, which is
correct when the user has whitened `A` and `b`, as Prism's documentation tells
them to).

    signed:        p(x | b)  ∝  exp( −( 0.5 * ‖A x − b‖² + λ * ‖x‖₁ ) / s2 )
    nonnegative:   the same for x ≥ 0, and 0 if any coefficient is negative

Its maximum is the LASSO solution at λ (with the same constraint) for every
value of `s2`; `s2` only controls how wide the distribution is.

### 3.2 One coefficient at a time (Gibbs sampling)

Hold every coefficient fixed except `x_i`. With `a_i` = column `i` of `A`:

    d_i     = ‖a_i‖²
    center  = x_i + a_i' * (b − A x) / d_i
    shift   = λ / d_i
    sigma   = sqrt( s2 / d_i )

- **Nonnegative:** `x_i` given the rest is `Normal(center − shift, sigma²)`
  restricted to `x_i ≥ 0`.
- **Signed:** `x_i` given the rest has density proportional to
  `exp(−(x − center)² / (2 sigma²) − shift * |x| / sigma²)`. That is two
  pieces glued at zero: `Normal(center − shift, sigma²)` restricted to
  `x ≥ 0`, and `Normal(center + shift, sigma²)` restricted to `x ≤ 0`. First
  pick the piece with the correct probability, then draw from it. The
  probability needs the logarithm of the normal cumulative distribution
  function, which is why `SpecialFunctions` is needed.

One **sweep** updates `i = 1, 2, …, n` in order, each time drawing a new `x_i`
from that distribution. After each update the vector `b − A x` changes by
`−(new − old) * a_i`, so it is kept up to date instead of being recomputed.
If column `i` is entirely zero (`d_i == 0`), the data say nothing about `x_i`
and it is drawn from the prior: an exponential distribution with mean `s2 / λ`,
with a random sign in the signed case.

There is nothing to tune: no step size, no acceptance rate.

Two equivalent bookkeeping forms are used:

- **Residual form:** keep `r = b − A x` (length m). Used for every sparse `A`
  and for a dense `A` with `m <= n`.
- **Gram form:** precompute `K = A'A` once and keep `A'(b − A x)` (length n).
  Used for a dense `A` with `m > n`.

## 4. Scope

In scope: `LassoProblem` with `constraint == :signed` or `:nonnegative`,
`loss == LeastSquares()`, `penalty == L1()`, `A` a `Matrix{Float64}` or a
`SparseMatrixCSC{Float64}`, `b` a `Vector{Float64}`, a single `λ > 0`.

Out of scope, and the code must throw a clear `ArgumentError` for each:

- Huber loss or Huber penalty.
- GPU arrays, `Float32` data, any other matrix type. The message must say to
  pass a `Matrix{Float64}` or `SparseMatrixCSC{Float64}`.
- `λ == 0` (without the penalty the distribution is not well defined when
  there are more coefficients than observations).
- `noise_variance <= 0` or not finite.
- Group, block, constrained, graph-Laplacian and TV problems (no method is
  defined for them, so there is nothing to do).

Do not implement: selection of λ, estimation of `s2`, per-coefficient λ,
correlated noise, multi-threaded chains, GPU support.

## 5. Reference code (tested; copy it)

`docs/plans/gibbs_sampling_reference/kernels.jl` contains the complete, tested
numerical core. Copy its whole content, unchanged, to the top of
`src/sampling/gibbs_posterior.jl`. It defines:

| Name | Meaning |
| --- | --- |
| `_log_normal_cdf(t)` | logarithm of the normal cumulative distribution function |
| `_rand_truncated_normal(rng, mu, sigma)` | one draw from a normal restricted to `x ≥ 0`, exact also far in the tail |
| `_rand_two_sided_normal(rng, center, shift, sigma)` | one draw from the signed one-coefficient distribution of 3.2 |
| `_GibbsWork` | what is computed once (column norms, optionally `A'A`); field `signed` selects the constraint |
| `_gibbs_state(p, x)` | the kept vector for the chosen form |
| `_gibbs_sweep!(rng, p, x, state)` | one sweep |
| `_gibbs_chain(rng, p, start; warmup, draws, thin)` | one whole chain, returning an `n × draws` matrix |

`docs/plans/gibbs_sampling_reference/kernels_check.jl` is the script that
verified this code against numerical integration and against an independent
sampler. After step 1 of section 6 you can run it with
`julia --project=. docs/plans/gibbs_sampling_reference/kernels_check.jl`
(under a minute). Every number it prints must be below 0.02; in the
reference run the largest was 0.011.

Diagnostics. Add these two functions below the copied kernels. The argument is
a `draws × chains` matrix holding one scalar quantity (one coefficient).

```julia
function _split_rhat(x::AbstractMatrix)
    half = size(x, 1) ÷ 2
    half >= 2 || return NaN
    parts = hcat(x[1:half, :], x[half+1:2half, :])
    within = mean(var(parts; dims=1))
    between = var(vec(mean(parts; dims=1)))
    within > 0 || return NaN
    return sqrt(((half - 1) / half * within + between) / within)
end

function _effective_sample_size(x::AbstractMatrix)
    N, M = size(x)
    N >= 4 || return NaN
    centered = x .- mean(x; dims=1)
    within = mean(var(x; dims=1))
    between = M > 1 ? var(vec(mean(x; dims=1))) : zero(within)
    total = (N - 1) / N * within + between
    total > 0 || return NaN
    rho(t) = 1 - (within - mean(sum(view(centered, 1:N-t, j) .*
                                    view(centered, t+1:N, j)) / N
                                for j in 1:M)) / total
    tau = -1.0
    t = 0
    while t + 1 < N
        pair = rho(t) + rho(t + 1)
        pair > 0 || break
        tau += 2 * pair
        t += 2
    end
    return N * M / max(tau, 1 / log10(N * M))
end
```

Expected behaviour of the diagnostics: on 4000×4 independent normal draws,
`_split_rhat` is 1.00 and `_effective_sample_size` is about 14000; if one chain
is shifted by 3 standard deviations, `_split_rhat` is about 1.7.

## 6. Files to add and edit

| Action | File |
| --- | --- |
| edit | `Project.toml` (and `Manifest.toml`, changed by the package manager) |
| add | `src/sampling/gibbs_posterior.jl` |
| add | `test/gibbs_posterior.jl` |
| add | `docs/posterior_sampling.md` |
| edit | `src/Prism.jl` |
| edit | `test/runtests.jl` |
| edit | `README.md` |
| edit | `docs/index.md` |

Do them in this order:

1. Dependencies. Before changing anything, copy `Project.toml` and
   `Manifest.toml` to `/home/tobias-grasberger/tmp/prism_backup_<date>/`. Then
   run, from the repository root,
   `julia --project=. -e 'using Pkg; Pkg.add(["Random", "SpecialFunctions"])'`.
   Afterwards edit `Project.toml` by hand: under `[compat]` add the line
   `SpecialFunctions = "2"`; remove the `Random` line from `[extras]`; and make
   `[targets]` read `test = ["Test"]`. `[deps]` must now list exactly
   `LinearAlgebra`, `Random`, `SparseArrays`, `SpecialFunctions`, `Statistics`.
2. `src/Prism.jl`: add the two lines `using Random` and
   `using SpecialFunctions: logerfc` after `using LinearAlgebra`. Add
   `include("sampling/gibbs_posterior.jl")` as the last `include`, after
   `include("compatibility_api.jl")`. Do not edit the existing `export` lines;
   the new file exports its own names.
3. `src/sampling/gibbs_posterior.jl`: sections 5 and 7.
4. `test/gibbs_posterior.jl`: section 8. Add
   `include("gibbs_posterior.jl")` as the last line of `test/runtests.jl`.
5. `docs/posterior_sampling.md`: section 10. In `docs/index.md` add an item 6
   to the list "Read in this order":
   `6. [Posterior sampling](posterior_sampling.md): uncertainty of the LASSO
   coefficients at one chosen λ.`
6. `README.md`: add one row to the table under "What is provided":
   `| Posterior samples for signed or nonnegative LASSO at one λ | Gibbs | Dense/sparse CPU, Float64 |`
   and add a short section "Posterior uncertainty" containing the usage example
   from section 7.3 and a link to `docs/posterior_sampling.md`.

## 7. Public interface to implement

All in `src/sampling/gibbs_posterior.jl`, below the code from section 5. End
the file with `export Gibbs, PosteriorSamples, sample_posterior,
posterior_mean, posterior_std, posterior_quantile, credible_interval`.

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

Validate in the keyword constructor: `warmup >= 0`, `draws >= 4`,
`chains >= 1`, `thin >= 1`, all integers; otherwise `ArgumentError`. `draws` is
the number of **stored** draws per chain. Each chain runs
`warmup + draws*thin` sweeps and stores every `thin`-th sweep after warm-up.

### 7.2 `PosteriorSamples` result

```julia
struct PosteriorSamples
    x::Array{Float64,3}       # n × draws × chains, coefficient draws
    lambda::Float64
    noise_variance::Float64
    constraint::Symbol        # :signed or :nonnegative, copied from the problem
    rhat::Vector{Float64}     # per coefficient, split R-hat
    ess::Vector{Float64}      # per coefficient, effective sample size
end
```

### 7.3 `sample_posterior`

```julia
sample_posterior(problem::LassoProblem, method::Gibbs, lambda::Real;
                 noise_variance=1.0, x0=nothing,
                 rng=Random.default_rng()) -> PosteriorSamples

sample_posterior(problem::LassoProblem, method::Gibbs, result::SolveResult;
                 kwargs...) =
    sample_posterior(problem, method, result.lambda; x0=result.x, kwargs...)

sample_posterior(problem::LassoProblem, lambda_or_result; kwargs...) =
    sample_posterior(problem, Gibbs(), lambda_or_result; kwargs...)
```

The second and third methods are what make this usable on top of a path:

```julia
problem = LassoProblem(A, b; constraint=:nonnegative)   # whitened A and b
path = solve_path(problem, FISTA())
chosen = path[12]                                        # the user picks one λ
samples = sample_posterior(problem, chosen)
posterior_mean(samples); credible_interval(samples; level=0.9)
```

Body of the first method, in this order:

1. Validate (section 4). Also: `lambda` finite and `> 0`; if `x0` is given,
   `length(x0) == size(problem.A, 2)` (else `DimensionMismatch`) and all
   entries finite. Negative entries of `x0` are allowed for a nonnegative
   problem; the chain clips them to zero.
2. `work = _GibbsWork(problem.A, problem.b, Float64(lambda), Float64(noise_variance), problem.constraint == :signed)`.
3. `start = isnothing(x0) ? zeros(n) : Vector{Float64}(x0)`.
4. Chain seeds: `seeds = rand(rng, UInt64, method.chains)`. Chain `k` uses its
   own `Xoshiro(seeds[k])`. This makes the result reproducible for a seeded
   `rng`.
5. For each chain `k`, in a plain `for` loop (no threads):
   `x[:, :, k] = _gibbs_chain(Xoshiro(seeds[k]), work, start; warmup=method.warmup, draws=method.draws, thin=method.thin)`.
6. Diagnostics: for each coefficient `i`,
   `rhat[i] = _split_rhat(x[i, :, :])`, `ess[i] = _effective_sample_size(x[i, :, :])`.
7. Warn, in the same style Prism already uses for poor solver convergence:
   `@warn "Posterior chains have not mixed; increase warmup, draws or thin" max_rhat min_ess`
   if `maximum(filter(isfinite, rhat)) > 1.05` or
   `minimum(filter(isfinite, ess)) < 100` (skip the check when the filtered
   list is empty).
8. Return the `PosteriorSamples`.

### 7.4 Summaries

All pool the draws of all chains.

- `posterior_mean(s::PosteriorSamples)` → `Vector{Float64}` of length n.
- `posterior_std(s)` → `Vector{Float64}` of length n.
- `posterior_quantile(s, p::Real)` → `Vector{Float64}`; require `0 <= p <= 1`.
  Use `Statistics.quantile` on `vec(s.x[i, :, :])` for each `i`.
- `credible_interval(s; level=0.9)` → named tuple `(lower, upper)` with
  `lower = posterior_quantile(s, (1 - level)/2)` and
  `upper = posterior_quantile(s, (1 + level)/2)`; require `0 < level < 1`.

Also define a short `Base.show(io::IO, s::PosteriorSamples)` that prints the
dimensions, the constraint, λ, the largest R-hat and the smallest ESS on one
or two lines, so printing a result in the REPL does not dump the whole array.

## 8. Tests (`test/gibbs_posterior.jl`)

Start the file with
`using Prism, Test, Random, LinearAlgebra, SparseArrays, Statistics` and one
`@testset "Posterior sampling" begin ... end`. Always pass an explicitly
seeded generator such as `MersenneTwister(1)`; never rely on the global RNG.
The reference values come from numerical integration on a grid, which needs no
extra package. The whole file should run in well under one minute.

1. **Validation.** `sample_posterior(problem, Gibbs(), 0.5)` throws
   `ArgumentError` for `loss=Huber(1.0)`, for `penalty=Huber(1.0)`, for
   `Float32` data, for `lambda = 0` and for `noise_variance = 0`; a wrong
   length `x0` throws `DimensionMismatch`. `Gibbs(draws=0)` and
   `Gibbs(thin=0)` throw `ArgumentError`.

2. **One-coefficient draws.** For each `(center, shift, sigma)` in
   `(0.7, 0.3, 0.5), (-0.2, 1.0, 1.0), (0.0, 2.0, 0.4), (30.0, 1.0, 1.0),
   (-4.0, 6.0, 0.5), (0.5, 40.0, 1.0)`: compute the exact mean and standard
   deviation by summing on
   `g = range(-abs(center) - 12sigma, abs(center) + 12sigma; length=2_000_001)`
   with weights `exp(l - maximum(l))`, where
   `l = -(g - center)^2 / (2sigma^2) - shift * abs(g) / sigma^2` (normalize
   the weights to sum to one). Draw 200,000 values with
   `Prism._rand_two_sided_normal`. Require
   `abs(sample mean − exact mean) < 0.03 * exact sd` and the sample standard
   deviation to match with `rtol=0.03`.
   Then for each `(mu, sigma)` in
   `(0.7, 0.5), (-0.2, 1.0), (-3.0, 0.4), (-20.0, 1.0), (5.0, 1.0)` do the
   same for `Prism._rand_truncated_normal` on
   `g = range(0, max(mu, 0) + 12sigma; length=2_000_001)` with weights
   `exp(-(g - mu)^2 / (2sigma^2))`; require mean and standard deviation to
   match with `rtol=0.03` and every draw to be `>= 0`.

3. **Two-coefficient posterior against a grid.** For each of
   `constraint in (:signed, :nonnegative)` and `m in (1, 6)`: build
   `A = randn(rng, m, 2)`, then make the columns correlated with
   `A[:, 2] = 0.9 .* A[:, 1] .+ 0.3 .* randn(rng, m)`;
   `b = A * [1.0, 0.0] .+ 0.4 .* randn(rng, m)`; `lambda = 0.5`,
   `noise_variance = 0.3`. Exact moments: on
   `g = range(constraint == :signed ? -12 : 0, 12; length=1201)` evaluate
   `logd[u, v] = -(sum(abs2, A*[u, v] - b)/2 + lambda*(abs(u) + abs(v))) / noise_variance`,
   subtract the maximum, exponentiate, normalize, and compute the mean and
   standard deviation of each coordinate. Run
   `Gibbs(warmup=1_000, draws=25_000, chains=2, thin=1)`. Require
   `posterior_mean` and `posterior_std` to match the grid values within
   `atol=0.03`, and `maximum(samples.rhat) < 1.05`. (`m = 1` uses the residual
   form, `m = 6` the Gram form.)

4. **Residual form, Gram form and sparse storage agree.** `A = randn(rng, 30, 8)`,
   `b = A * randn(rng, 8) + 0.5 .* randn(rng, 30)`, `lambda = 1.0`,
   `noise_variance = 0.25`. For `signed in (true, false)`: build
   `pg = Prism._GibbsWork(A, b, 1.0, 0.25, signed)` (Gram form),
   `pr = Prism._GibbsWork(A, b, 1.0, 0.25, signed, pg.column_squares, false, zeros(0, 0))`
   (residual form) and `ps = Prism._GibbsWork(sparse(A), b, 1.0, 0.25, signed)`.
   Run `Prism._gibbs_chain` on each with different seeds,
   `warmup=1_000, draws=50_000, thin=1`. Means and standard deviations of the
   eight coefficients agree between all three within `atol=0.03`.

5. **Zero column.** `A = randn(rng, 5, 3); A[:, 2] .= 0`, `lambda = 0.8`,
   `noise_variance = 0.4`, `Gibbs(warmup=500, draws=20_000, chains=2, thin=1)`.
   Nonnegative: the posterior mean of coefficient 2 is
   `noise_variance / lambda = 0.5` within `rtol=0.05`. Signed: the mean of
   coefficient 2 is `0` within `atol=0.03` and the mean of its absolute value
   is `0.5` within `rtol=0.05`.

6. **Nonnegativity and shapes.** For a nonnegative problem all entries of
   `samples.x` are `>= 0`. For a signed problem some entries are negative.
   `size(samples.x) == (n, draws, chains)`; `length(samples.rhat) == n`;
   `length(samples.ess) == n`; `samples.constraint` equals the problem's
   constraint. With `ci = credible_interval(samples)`:
   `all(ci.lower .<= posterior_quantile(samples, 0.5) .<= ci.upper)`.

7. **Reproducibility.** Two calls with `rng=MersenneTwister(5)` give identical
   `x` arrays (`==`). A call with a different seed gives a different array.

8. **Path convenience.**
   `sample_posterior(problem, solve(problem, FISTA(), 0.8); rng=MersenneTwister(1))`
   and `sample_posterior(problem, 0.8; rng=MersenneTwister(1))` both return a
   result with `lambda == 0.8`. Keep other runs short by passing
   `Gibbs(warmup=200, draws=50, chains=2, thin=1)` explicitly.

9. **Mode consistency (sanity, loose).** For a well-determined problem
   (`m = 200, n = 5`, `A = randn(rng, m, n)`, true coefficients
   `[2, 0, 1, 0, 3]`, noise standard deviation 0.1, `lambda = 0.5`,
   `noise_variance = 0.01`), for both constraints, the posterior mean of every
   coefficient is within `0.1` of the LASSO solution from
   `solve(problem, FISTA(stop=KKTStop(rtol=1e-6)), 0.5)`.

10. **Diagnostics.** With `MersenneTwister(3)`: for `randn(rng, 4000, 4)`,
    `Prism._split_rhat` is within 0.01 of 1 and
    `Prism._effective_sample_size` is between 10_000 and 17_000; after adding
    3 to the fourth column, `_split_rhat > 1.3`.

Tests 2, 3 and 4 prove the sampler is correct. They are not optional, and
their tolerances must not be loosened: in the reference runs the largest error
was 0.009. If one fails by a wide margin, the implementation is wrong.

## 9. Mistakes to avoid

- Draw from the restricted normal. Do not draw from the unrestricted normal
  and clip negative values to zero; that gives a wrong distribution with a
  spike at zero.
- Use the **current** `x` and the **current** kept vector for every coordinate
  in the sweep (each update must see the updates before it). Do not compute
  all centers first and then update all coefficients.
- `shift = λ / d_i`, not `λ / s2`. `s2` enters only through
  `sigma = sqrt(s2 / d_i)`.
- Keep the periodic recomputation of the kept vector (every 100 sweeps) that
  is in the reference code; it stops rounding errors from accumulating.
- Do not add an acceptance step. Gibbs updates are always accepted.
- Do not replace `_log_normal_cdf` with `log(cdf)` or `log(0.5 * erfc(...))`.
  Far in the tail those give `-Inf` and the signed sampler picks the wrong
  piece.
- Do not call the result sparse. Posterior draws and the posterior mean are
  never exactly zero. Only the posterior mode (the LASSO solution) is sparse.

## 10. Documentation (`docs/posterior_sampling.md`)

Write about one page, in the plain style of the other files in `docs/`. It
must state, in this order:

1. What the function returns and the usage example from 7.3.
2. The two posterior formulas from 3.1 and the statement that the mode is the
   LASSO solution at the same λ and constraint.
3. That `noise_variance` must be the variance of the noise in `b` in the units
   of `b` as passed in; `1` is correct only for whitened data; too small a
   value gives intervals that are too narrow.
4. That the posterior mean is denser and differently shrunk than the LASSO
   solution, and that intervals are conditional on the chosen λ (uncertainty
   about λ is not included).
5. Limits: least squares and L1 only, CPU `Float64` only, dense or sparse.
6. Cost and mixing: copy the paragraph in section 11 of this plan.
7. How to read the diagnostics: `rhat` should be below 1.05 and `ess` above a
   few hundred for every coefficient that matters; otherwise increase `thin`
   or `draws`.

## 11. Measured cost (text for the documentation)

These numbers were measured on the development machine, on one CPU core, for
n = 6000 coefficients. Use them in the documentation as written; do not
re-measure or round them differently.

- **Cost of one sweep** (signed; nonnegative is about 10 % cheaper):

  | m | n | Nonzero fraction of `A` | Dense `Matrix` | `SparseMatrixCSC` |
  | --- | --- | --- | --- | --- |
  | 1500 | 6000 | 33 % | 3.8 ms | 4.3 ms |
  | 1500 | 6000 | 5 % | 4.0 ms | 1.1 ms |
  | 6000 | 6000 | 33 % | 16 ms | 16 ms |
  | 6000 | 6000 | 5 % | 16 ms | 2.8 ms |

  The default `Gibbs()` runs 11,000 sweeps per chain and four chains, so about
  3 minutes in total for m = 1500 and about 12 minutes for m = 6000 with a
  dense matrix. Sparse storage only pays off when clearly fewer than a third
  of the entries are nonzero.
- **Mixing.** The sampler updates one coefficient at a time, so coefficients
  whose columns of `A` overlap strongly move slowly together. On a synthetic
  test with overlapping plume-shaped rows (n = 6000), half of the coefficients
  reached an effective sample size equal to about 97 % of the stored draws,
  but the worst coefficient reached only 12 % (m = 1500) and 3 to 13 %
  (m = 6000). `rhat` stayed below 1.03 in all cases. Always check `ess`; if
  the smallest values are too low for the coefficients that matter, increase
  `thin` or `draws`.

## 12. Definition of done

- [ ] The file changes in section 6 exist and nothing else changed compared
      with the state before you started (`git status --short` shows the same
      entries as before plus `Project.toml`, `Manifest.toml`, `docs/index.md`,
      `docs/posterior_sampling.md`, `src/sampling/`, `test/gibbs_posterior.jl`).
- [ ] `julia --project=. -e 'using Pkg; Pkg.test()'` passes, including the new
      test file.
- [ ] `sample_posterior(problem, path_result)` works for a signed and for a
      nonnegative problem without naming a method.
- [ ] No new entry in `[deps]` other than `Random` and `SpecialFunctions`.
- [ ] The word "Laplacian" does not appear in any file you added
      (`grep -ri laplacian src/sampling test/gibbs_posterior.jl docs/posterior_sampling.md`
      prints nothing).
- [ ] Report back: the test summary, the run time of the new test file, and
      the largest `rhat` and smallest `ess` seen in test 3.
