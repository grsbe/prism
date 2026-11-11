# Implementation plan: posterior sampling for the signed LASSO in Prism

> **SUPERSEDED (2026-11-11). Do not implement this plan.** It is replaced by
> `gibbs_posterior_sampling.md` in this folder, which covers signed and
> nonnegative problems with one Gibbs sampler. This file is kept only as a
> record of the MALA approach.

This plan tells you exactly what to build, in what order, and how to check it.
Follow it literally. Where it gives code, copy the code; it has been run and
checked. Where it gives a rule, do not replace it with your own idea. If
something in the repository contradicts this plan, stop and report the
contradiction instead of guessing.

Nonnegative problems are handled by a separate follow-on plan,
`nonnegative_posterior_sampling.md`; do this plan first.

## 1. What you are building and why

Prism (`/home/tobias-grasberger/prism`, a Julia package) solves

    minimize over x:   0.5 * ‖A x − b‖²  +  λ * ‖x‖₁

and returns one coefficient vector per λ. That vector is a point estimate with
no uncertainty. You will add a sampler that, for **one fixed λ**, draws
coefficient vectors from the Bayesian posterior whose most probable point is
that same LASSO solution. From the draws the user gets a posterior mean,
standard deviation and credible interval for every coefficient.

The method comes from a colleague's research code in
`/home/tobias-grasberger/sparse-inversion-sampling` (files
`src/mtc_laplacian_prob.jl`, `src/mala_helper.jl`, derivation in
`documentation/sampling_Laplacian.tex`). **Do not copy that code and do not
add its dependencies.** It is script-quality, depends on the Turing/AdvancedMH
stack, and has defects (listed in section 9). You will reimplement the same
mathematics inside Prism using only Julia standard libraries. Read the
colleague's files only if you want background; this plan is sufficient.

## 2. Rules that apply to every step

1. Work only inside `/home/tobias-grasberger/prism`. Put temporary files in
   `/home/tobias-grasberger/tmp`, never in `/tmp`.
2. Never delete or overwrite existing files. Only add new files and make the
   small edits listed in section 5. Do not modify any file under
   `src/solvers/` or `src/workflows/`.
3. Do not add any package dependency other than the standard library `Random`.
   No Distributions, no AdvancedMH, no MCMCChains, no Turing.
4. Naming: the new feature is called **Laplace** posterior sampling. Prism
   already uses the word "Laplacian" for something unrelated (graph-Laplacian
   smoothing in `src/solvers/laplacian_fista.jl`). Never use "Laplacian" in a
   new name, docstring or file name.
5. Match the existing code style: four-space indent, docstrings in triple
   quotes above definitions, internal helpers prefixed with `_`, argument
   validation that throws `ArgumentError` or `DimensionMismatch` with a clear
   message, `export` statements at the bottom of the file that defines the
   names.
6. Do not commit or push. Leave the changes in the working tree.
7. Run the full test suite before you start and after you finish:
   `cd /home/tobias-grasberger/prism && julia --project=. -e 'using Pkg; Pkg.test()'`.
   It passes today. It must still pass at the end.

## 3. The mathematics (everything you need)

### 3.1 The posterior

Inputs: matrix `A` (m rows, n columns), vector `b` (length m), penalty `λ > 0`,
and the noise variance `s2 > 0` of each entry of `b` (default `1`, which is
correct when the user has whitened `A` and `b`, as Prism's documentation tells
them to).

The posterior density of the coefficients is

    p(x | b)  ∝  exp( −( 0.5 * ‖A x − b‖² + λ * ‖x‖₁ ) / s2 )

Its maximum is the LASSO solution at λ for every value of `s2`; `s2` only
controls how wide the distribution is. Define

    delta = λ / s2          (the Laplace prior is  p(x_i) ∝ exp(−delta * |x_i|))
    rate  = delta² / 2

### 3.2 The trick: one hidden variance per coefficient

A Laplace prior equals a Gaussian prior whose variance is itself random:

    x_i | w_i ~ Normal(0, variance w_i),      w_i ~ Exponential with rate `rate`

Given all the variances `w`, everything is Gaussian, so `x` can be integrated
out exactly. The sampler therefore works in two stages:

1. Sample the variances `w` from `p(w | b)` with a Markov chain (stage 1).
2. For each sampled `w`, draw one `x` from the Gaussian `p(x | w, b)` (stage 2).

The resulting `x` draws are draws from the posterior in 3.1.

### 3.3 Stage 1 density, in the variable θ = log(w)

The chain runs on `θ = log.(w)` (one unconstrained real number per
coefficient). With `W = Diagonal(w)` and `C = s2*I + A*W*A'` (an m×m matrix):

    logp(θ) = −0.5*logdet(C) − 0.5 * b' * C⁻¹ * b − rate*sum(w) + sum(θ)

    gradient_i = w_i * ( −0.5*q_i + 0.5*z_i² − rate ) + 1
        where  q_i = a_i' * C⁻¹ * a_i     (a_i is column i of A)
               z_i = a_i' * C⁻¹ * b

This is the "observation form": cost per evaluation is dominated by an m×m
Cholesky and one triangular solve with n right-hand sides. Use it when
`m <= n`.

When `m > n` use the equivalent "coefficient form", which needs only n×n work.
With `K = A'A / s2`, `c = A'b / s2`, `s = sqrt.(w)`,
`B = I + Diagonal(s)*K*Diagonal(s)`:

    logp(θ) = −0.5*logdet(B) + 0.5 * (s.*c)' * B⁻¹ * (s.*c) − rate*sum(w) + sum(θ)

The two forms differ only by a constant that does not depend on θ, and their
gradients are identical. Both are given as code in section 6.

### 3.4 Stage 2 draw

Observation form: draw `x0 = sqrt.(w) .* randn(n)` and
`e = sqrt(s2) .* randn(m)`, then

    x = x0 + w .* ( A' * ( C⁻¹ * (b − A*x0 − e) ) )

Coefficient form: with the Cholesky factor `F` of `B` and `z = randn(n)`,

    x = s .* (F \ (s .* c))  +  s .* (F.U \ z)

### 3.5 The Markov chain (MALA)

MALA = Metropolis-adjusted Langevin algorithm. One step from state θ with step
size `h`:

    forward_mean  = θ + (h/2) * gradient(θ)
    proposal      = forward_mean + sqrt(h) * randn(n)
    backward_mean = proposal + (h/2) * gradient(proposal)
    log_ratio     = logp(proposal) − logp(θ)
                    − sum(abs2, θ − backward_mean) / (2h)
                    + sum(abs2, proposal − forward_mean) / (2h)
    accept with probability min(1, exp(log_ratio)); otherwise stay at θ

If `log_ratio` is not finite, reject. During warm-up only, after every step
update `log(h) += (acceptance_probability − target) / sqrt(step_number)`. After
warm-up `h` is frozen. Never adapt `h` after warm-up; that would make the
samples invalid.

## 4. Scope

In scope: `LassoProblem` with `constraint == :signed`, `loss` equal to
`LeastSquares()`, `penalty` equal to `L1()`, CPU dense `Matrix{Float64}` and
`Vector{Float64}`, a single `λ > 0`.

Out of scope, and the code must throw a clear `ArgumentError` for each:

- `constraint == :nonnegative`. The variance trick in 3.2 is only valid for
  signed coefficients. Message: `"posterior sampling supports only
  constraint=:signed; the Laplace scale-mixture sampler is not valid for
  nonnegative coefficients"`.
- Huber loss or Huber penalty.
- Sparse matrices, GPU arrays, `Float32`. Message should say to pass a
  `Matrix{Float64}`.
- `λ == 0`.
- Group, block, constrained, graph-Laplacian and TV problems (no method is
  defined for them, so nothing to do).

Do not implement: selection of λ, estimation of `s2`, per-coefficient λ,
correlated noise, multi-threaded chains, GPU support.

## 5. Files to add and edit

| Action | File |
| --- | --- |
| add | `src/sampling/laplace_posterior.jl` |
| add | `test/laplace_posterior.jl` |
| add | `docs/posterior_sampling.md` |
| edit | `Project.toml` |
| edit | `src/Prism.jl` |
| edit | `test/runtests.jl` |
| edit | `README.md` |
| edit | `docs/index.md` |

Edits, exactly:

- `Project.toml`: under `[deps]` add the line
  `Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"` (keep the section
  alphabetical). Remove the `Random` line from `[extras]` and remove
  `"Random"` from the `test` target list, so that `[targets]` reads
  `test = ["Test"]`. Change nothing else.
- `src/Prism.jl`: add `using Random` after `using LinearAlgebra`. Add
  `include("sampling/laplace_posterior.jl")` as the last `include`, after
  `include("compatibility_api.jl")`. Do not edit the existing `export` lines;
  the new file exports its own names.
- `test/runtests.jl`: add `include("laplace_posterior.jl")` after the last
  existing `include` line.
- `README.md`: add one row to the table under "What is provided":
  `| Posterior samples for signed LASSO at one λ | MALA on Laplace scale mixture | Dense CPU, Float64 |`
  and add a short section "Posterior uncertainty" containing the usage example
  from section 7 and a link to `docs/posterior_sampling.md`.
- `docs/index.md`: add a link to `posterior_sampling.md` in the same style as
  the existing links.

## 6. Reference code (tested; copy it)

The functions below were run and verified: the two density forms agree to
1e-14, the gradient matches finite differences to 1e-8, the stage 2 draws have
exactly the right mean and covariance, and the complete sampler reproduces an
exact one-dimensional posterior and an independent Gibbs sampler. The scripts
that did this are in `docs/plans/laplace_sampling_reference/`; you can run them
with plain `julia <file>` to see the expected output.

Put this in `src/sampling/laplace_posterior.jl`. `_LaplaceWork` holds
everything that is computed once.

```julia
struct _LaplaceWork
    A::Matrix{Float64}
    b::Vector{Float64}
    rate::Float64             # delta^2 / 2 with delta = lambda / noise_variance
    noise_variance::Float64
    observation_form::Bool    # true when m <= n
    K::Matrix{Float64}        # A'A / noise_variance (0x0 in observation form)
    c::Vector{Float64}        # A'b / noise_variance (empty in observation form)
end

function _LaplaceWork(A, b, lambda, noise_variance)
    m, n = size(A)
    delta = lambda / noise_variance
    observation_form = m <= n
    K = observation_form ? zeros(0, 0) : (A' * A) ./ noise_variance
    c = observation_form ? zeros(0) : (A' * b) ./ noise_variance
    return _LaplaceWork(A, b, delta^2 / 2, noise_variance, observation_form, K, c)
end

# Returns (logp, gradient). Returns (-Inf, zeros) if the factorization fails.
function _laplace_logp_gradient(p::_LaplaceWork, theta)
    w = exp.(theta)
    all(isfinite, w) || return -Inf, zero(theta)
    if p.observation_form
        F = cholesky(Symmetric((p.A .* w') * p.A' + p.noise_variance * I);
                     check=false)
        issuccess(F) || return -Inf, zero(theta)
        alpha = F \ p.b
        z = p.A' * alpha
        q = vec(sum(abs2, F.L \ p.A; dims=1))
        logp = -sum(log, diag(F.L)) - dot(p.b, alpha) / 2 -
               p.rate * sum(w) + sum(theta)
    else
        s = sqrt.(w)
        F = cholesky(Symmetric(s .* p.K .* s' + I); check=false)
        issuccess(F) || return -Inf, zero(theta)
        t = F \ (s .* p.c)
        logp = -sum(log, diag(F.L)) + dot(t, s .* p.c) / 2 -
               p.rate * sum(w) + sum(theta)
        X = s .* (F \ (s .* p.K))
        q = diag(p.K) .- vec(sum(p.K .* X'; dims=2))
        z = p.c .- p.K * (s .* t)
    end
    gradient = w .* (-q ./ 2 .+ z .^ 2 ./ 2 .- p.rate) .+ 1
    return logp, gradient
end

function _laplace_draw_coefficients(rng, p::_LaplaceWork, theta)
    w = exp.(theta)
    if p.observation_form
        x0 = sqrt.(w) .* randn(rng, length(w))
        e = sqrt(p.noise_variance) .* randn(rng, length(p.b))
        F = cholesky(Symmetric((p.A .* w') * p.A' + p.noise_variance * I))
        return x0 .+ w .* (p.A' * (F \ (p.b .- p.A * x0 .- e)))
    else
        s = sqrt.(w)
        F = cholesky(Symmetric(s .* p.K .* s' + I))
        return s .* (F \ (s .* p.c)) .+ s .* (F.U \ randn(rng, length(w)))
    end
end
```

Diagnostics. The argument is a `draws × chains` matrix holding one scalar
quantity (for example one coefficient).

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

## 7. Public interface to implement

All in `src/sampling/laplace_posterior.jl`. End the file with
`export MALA, PosteriorSamples, sample_posterior, posterior_mean,
posterior_std, posterior_quantile, credible_interval`.

### 7.1 `MALA` configuration

```julia
struct MALA
    warmup::Int
    draws::Int
    chains::Int
    thin::Int
    initial_step::Float64
    target_acceptance::Float64
end

MALA(; warmup=5_000, draws=2_000, chains=4, thin=1, initial_step=1e-2,
     target_acceptance=0.574)
```

Validate in the keyword constructor: `warmup >= 0`, `draws >= 4`, `chains >= 1`,
`thin >= 1` (all integers), `initial_step` finite and positive,
`0 < target_acceptance < 1`. Throw `ArgumentError` otherwise.

`draws` is the number of **stored** draws per chain. The chain runs
`warmup + draws*thin` steps and stores every `thin`-th step after warm-up.

### 7.2 `PosteriorSamples` result

```julia
struct PosteriorSamples
    x::Array{Float64,3}              # n × draws × chains, coefficient draws
    log_density::Matrix{Float64}     # draws × chains, logp(θ) of each stored state
    lambda::Float64
    noise_variance::Float64
    acceptance_rate::Vector{Float64} # per chain, measured after warm-up
    step_size::Vector{Float64}       # per chain, the frozen step size
    rhat::Vector{Float64}            # per coefficient, split R-hat (NaN if chains*draws too small)
    ess::Vector{Float64}             # per coefficient, effective sample size
end
```

Do not store the θ chain; it doubles memory and the user does not need it.

### 7.3 `sample_posterior`

```julia
sample_posterior(problem::LassoProblem, method::MALA, lambda;
                 noise_variance=1.0, x0=nothing,
                 rng=Random.default_rng()) -> PosteriorSamples

sample_posterior(problem::LassoProblem, method::MALA, result::SolveResult;
                 kwargs...) =
    sample_posterior(problem, method, result.lambda; x0=result.x, kwargs...)
```

The second method is what makes this usable on top of a path:

```julia
problem = LassoProblem(A, b)                       # whitened A and b
path = solve_path(problem, FISTA())
chosen = path[12]                                  # the user picks one λ
samples = sample_posterior(problem, MALA(), chosen)
posterior_mean(samples); credible_interval(samples; level=0.9)
```

Body of the three-argument method, in this order:

1. Validate (section 4). Also: `lambda` finite and `> 0`; `noise_variance`
   finite and `> 0`; if `x0` is given, `length(x0) == size(problem.A, 2)` and
   all entries finite.
2. `work = _LaplaceWork(problem.A, problem.b, Float64(lambda), Float64(noise_variance))`
   and `delta = lambda / noise_variance`.
3. Starting point, the same for all chains before jitter:
   `base = isnothing(x0) ? fill(log(2 / delta^2), n) : log.(abs.(x0) ./ delta .+ 1 / delta^2)`.
4. Chain seeds: `seeds = rand(rng, UInt64, method.chains)`. Chain `k` uses its
   own `chain_rng = Xoshiro(seeds[k])`. This makes the result reproducible for
   a seeded `rng` and independent of the order chains run in.
5. For each chain `k` (plain `for` loop, no threads; BLAS is already
   multi-threaded):
   - `theta = base .+ 0.5 .* randn(chain_rng, n)` (the jitter makes the R-hat
     check meaningful).
   - Run the loop from section 3.5 for `warmup + draws*thin` steps. Keep the
     current `logp` and `gradient` of the state so each step costs exactly one
     call to `_laplace_logp_gradient` (for the proposal). If the starting
     `logp` is not finite, throw an `ErrorException` saying the initial state
     has zero density.
   - After warm-up, at every `thin`-th step: store `logp` in
     `log_density[j, k]` and store
     `_laplace_draw_coefficients(chain_rng, work, theta)` in `x[:, j, k]`.
   - Count accepted steps after warm-up; `acceptance_rate[k] = accepted / (draws*thin)`.
   - `step_size[k] = exp(log_step)` at the end of warm-up.
6. Diagnostics: for each coefficient `i`,
   `rhat[i] = _split_rhat(x[i, :, :])`, `ess[i] = _effective_sample_size(x[i, :, :])`.
7. Warn, in the same style Prism already uses for poor solver convergence:
   `@warn "Posterior chains have not mixed; increase warmup/draws" max_rhat min_ess`
   if `maximum(filter(isfinite, rhat)) > 1.05` or
   `minimum(filter(isfinite, ess)) < 100` (skip the check when the filtered
   list is empty).
8. Return the `PosteriorSamples`.

The MALA loop for one chain, written out (this exact loop was tested):

```julia
logp, gradient = _laplace_logp_gradient(work, theta)
log_step = log(method.initial_step)
accepted = 0
total = method.warmup + method.draws * method.thin
for it in 1:total
    h = exp(log_step)
    forward_mean = theta .+ (h / 2) .* gradient
    proposal = forward_mean .+ sqrt(h) .* randn(chain_rng, n)
    proposal_logp, proposal_gradient = _laplace_logp_gradient(work, proposal)
    backward_mean = proposal .+ (h / 2) .* proposal_gradient
    log_ratio = proposal_logp - logp -
                sum(abs2, theta .- backward_mean) / (2h) +
                sum(abs2, proposal .- forward_mean) / (2h)
    probability = isfinite(log_ratio) ? min(1.0, exp(log_ratio)) : 0.0
    if rand(chain_rng) < probability
        theta, logp, gradient = proposal, proposal_logp, proposal_gradient
        it > method.warmup && (accepted += 1)
    end
    if it <= method.warmup
        log_step += (probability - method.target_acceptance) / sqrt(it)
    elseif (it - method.warmup) % method.thin == 0
        j = (it - method.warmup) ÷ method.thin
        # store logp and one coefficient draw in column j of chain k
    end
end
```

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
dimensions, λ, the range of acceptance rates, the largest R-hat and the
smallest ESS on one or two lines, so printing a result in the REPL does not
dump the whole array.

## 8. Tests (`test/laplace_posterior.jl`)

Start the file with `using Prism, Test, Random, LinearAlgebra, Statistics` and
one `@testset "Laplace posterior sampling" begin ... end`. Always pass an
explicitly seeded generator such as `MersenneTwister(1)`; never rely on the
global RNG. The whole file should run in well under one minute.

1. **Validation.** `sample_posterior` throws `ArgumentError` for: a
   `:nonnegative` problem; `loss=Huber(1.0)`; `penalty=Huber(1.0)`; a sparse
   `A`; `Float32` data; `lambda = 0`; `noise_variance = 0`; an `x0` of the
   wrong length (`DimensionMismatch` is also acceptable for that one). `MALA`
   throws for `draws=0`, `thin=0`, `initial_step=-1`, `target_acceptance=1`.

2. **Density and gradient.** Build `m, n = 12, 20` random data with
   `noise_variance = 0.3`, `lambda = 0.8`. Build a second work object with the
   same data but `observation_form` forced to `false` (construct
   `Prism._LaplaceWork` field by field with `K = A'A/0.3`, `c = A'b/0.3`). At
   two random θ vectors: the difference of `logp` between the two θ agrees
   between the two forms to `1e-10`; the gradients agree to `1e-10`; the
   gradient agrees with central finite differences (step `1e-6`) to `1e-6`.

3. **Exact one-dimensional posterior.** `A = reshape([1.3], 1, 1)`,
   `b = [0.9]`, `noise_variance = 0.5`, `lambda = 0.6`. The exact posterior
   density is proportional to
   `exp(-(1.3x - 0.9)^2 / (2*0.5) - (0.6/0.5)*abs(x))`; integrate it on
   `range(-10, 10; length=400_001)` to get mean `0.4530` and variance `0.2194`.
   Run `MALA(warmup=2_000, draws=40_000, chains=1)`. Check
   `posterior_mean ≈ 0.4530` and `posterior_std^2 ≈ 0.2194`, both with
   `atol=0.03`.

4. **Agreement with an independent Gibbs sampler (wide case, m < n).** Use
   `m, n = 12, 20`, two nonzero true coefficients, `noise_variance = 0.3`,
   `lambda = 0.8`. Implement this reference sampler inside the test file only:

   ```julia
   function rinvgauss(rng, mu, shape)
       y = randn(rng)^2
       x = mu + mu^2 * y / (2shape) -
           mu / (2shape) * sqrt(4mu * shape * y + mu^2 * y^2)
       return rand(rng) <= mu / (mu + x) ? x : mu^2 / x
   end

   function gibbs_reference(rng, A, b, lambda, noise_variance; warmup, draws)
       n = size(A, 2)
       delta = lambda / noise_variance
       K = (A' * A) ./ noise_variance
       c = (A' * b) ./ noise_variance
       w = ones(n)
       X = zeros(n, draws)
       for it in 1:(warmup + draws)
           F = cholesky(Symmetric(K + Diagonal(1 ./ w)))
           x = F \ c .+ F.U \ randn(rng, n)
           for i in 1:n
               w[i] = 1 / rinvgauss(rng, delta / max(abs(x[i]), 1e-300), delta^2)
           end
           it > warmup && (X[:, it - warmup] = x)
       end
       return X
   end
   ```

   Run it with `warmup=2_000, draws=40_000`, and run
   `sample_posterior` with `MALA(warmup=5_000, draws=10_000, chains=4)`.
   Require `maximum(abs, posterior_mean - mean(X; dims=2)) < 0.05` and
   `maximum(abs, posterior_std - std(X; dims=2)) < 0.05`. Also require
   `maximum(samples.rhat) < 1.05` and all acceptance rates between 0.3 and 0.9.
   (With much longer runs the two samplers agree to 0.007; 0.05 leaves room
   for Monte Carlo noise. If this test fails by a wide margin, the
   implementation is wrong; do not loosen the tolerance.)

5. **Tall case (m > n)**, which exercises the coefficient form: `m, n = 40, 8`,
   same comparison against `gibbs_reference`, same tolerances.

6. **Reproducibility.** Two calls with `rng=MersenneTwister(5)` give identical
   `x` arrays (`==`). A call with a different seed gives a different array.

7. **Path convenience.** `sample_posterior(problem, MALA(warmup=200, draws=50,
   chains=2), solve(problem, FISTA(), 0.8))` returns a result with
   `lambda == 0.8` and `size(x) == (n, 50, 2)`.

8. **Shapes and summaries.** `size(samples.x)`, `size(samples.log_density)`,
   lengths of `rhat`, `ess`, `acceptance_rate`, `step_size`;
   `credible_interval(samples).lower .<= posterior_mean(samples) .<=
   credible_interval(samples).upper` need not hold in general, so instead check
   `lower .<= posterior_quantile(samples, 0.5) .<= upper`.

9. **Diagnostics.** With `MersenneTwister(3)`: for `randn(rng, 4000, 4)`,
   `Prism._split_rhat` is within 0.01 of 1 and
   `Prism._effective_sample_size` is between 10_000 and 17_000; after adding 3
   to the fourth column, `_split_rhat > 1.3`.

Tests 3, 4 and 5 are the ones that prove the mathematics is right. They are
not optional.

## 9. Mistakes to avoid (each of these is a real defect in the code being ported or an easy slip)

- Do not adapt the step size after warm-up.
- Do not restart the chain in batches. The colleague's helper restarts
  AdvancedMH every batch; each restart re-emits the previous state as a new
  sample (marked "not accepted"), which duplicates one sample per batch and
  biases the measured acceptance rate downward.
- Do not `push!` results into shared arrays from several threads (the
  colleague's multi-chain wrapper does; it is a data race). Use the plain loop
  with preallocated arrays indexed by chain.
- Do not skip the `+ sum(theta)` term in `logp` or the `+ 1` in the gradient.
  They come from the change of variable `θ = log(w)`.
- The sign in front of the quadratic term matters. In the colleague's write-up
  the "Implementation" formulas show `−½‖x*‖²`; that is a typo there. The code
  in section 6 has the correct signs for both forms.
- Do not introduce a threshold that treats small `w` as exactly zero (the
  colleague's `atol`). In the θ variable every `w` is positive, so the
  threshold never triggers and only adds an inconsistent special case.
- `delta = lambda / noise_variance`, not `lambda / sqrt(noise_variance)`, and
  `rate = delta^2 / 2`.
- Do not call the result sparse. Posterior draws and the posterior mean are
  never exactly zero. Only the posterior mode (the LASSO solution) is sparse.

## 10. Documentation (`docs/posterior_sampling.md`)

Write about one page, in the plain style of the other files in `docs/`. It
must state, in this order:

1. What the function returns and the three-line usage example from 7.3.
2. The posterior formula from 3.1 and the statement that its mode is the LASSO
   solution at the same λ.
3. That `noise_variance` must be the variance of the noise in `b` in the units
   of `b` as passed in; `1` is correct only for whitened data; too small a
   value gives intervals that are too narrow.
4. That the posterior mean is denser and differently shrunk than the LASSO
   solution, and that intervals are conditional on the chosen λ (uncertainty
   about λ is not included).
5. Limits: signed coefficients only, least squares and L1 only, dense CPU
   `Float64` only.
6. Cost: every step factors a `min(m, n)`-sized dense matrix. Measured on the
   development machine (32 BLAS threads): about 0.26 s per step for
   m = 2000, n = 6000 and about 1.2 s per step for m = 5000, n = 10000, so the
   default 7000 steps per chain take about 30 minutes and 2.3 hours per chain
   respectively.
7. How to read the diagnostics: `rhat` should be below 1.05 and `ess` above a
   few hundred for every coefficient that matters; otherwise increase `warmup`
   and `draws`.

## 11. Definition of done

- [ ] The eight file changes in section 5 exist and nothing else changed
      (`git status` shows only those files plus this plan folder).
- [ ] `julia --project=. -e 'using Pkg; Pkg.test()'` passes, including the new
      test file.
- [ ] `sample_posterior` on a `:nonnegative` problem throws the message in
      section 4.
- [ ] No new entry in `[deps]` other than `Random`.
- [ ] The word "Laplacian" does not appear in any file you added
      (`grep -ri laplacian src/sampling test/laplace_posterior.jl docs/posterior_sampling.md`
      prints nothing).
- [ ] Report back: the test summary, the wall time of the new test file, and
      the measured acceptance rates and largest R-hat from test 4.
