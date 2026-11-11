# Log of the standard normal cumulative distribution function, accurate in
# both tails (logerfc comes from SpecialFunctions).
_log_normal_cdf(t) = logerfc(-t / sqrt(2)) - log(2)

# One draw from Normal(mu, sigma^2) restricted to x >= 0.
function _rand_truncated_normal(rng, mu, sigma)
    alpha = -mu / sigma                  # lower bound in standard units
    if alpha <= 0                        # at least half the mass is allowed
        while true
            z = randn(rng)
            z >= alpha && return mu + sigma * z
        end
    else                                 # tail: shifted-exponential rejection
        a = (alpha + sqrt(alpha^2 + 4)) / 2
        while true
            z = alpha + randexp(rng) / a
            log(rand(rng)) <= -(z - a)^2 / 2 && return max(mu + sigma * z, 0.0)
        end
    end
end

# One draw from the density proportional to
#     exp(-(x - center)^2 / (2 sigma^2) - shift * |x| / sigma^2)
# on the whole real line. It is a normal centred at `center - shift` cut to
# x >= 0, glued to a normal centred at `center + shift` cut to x <= 0.
function _rand_two_sided_normal(rng, center, shift, sigma)
    log_positive = -center * shift / sigma^2 +
                   _log_normal_cdf((center - shift) / sigma)
    log_negative = center * shift / sigma^2 +
                   _log_normal_cdf(-(center + shift) / sigma)
    if rand(rng) < 1 / (1 + exp(log_negative - log_positive))
        return _rand_truncated_normal(rng, center - shift, sigma)
    else
        return -_rand_truncated_normal(rng, -(center + shift), sigma)
    end
end

struct _GibbsWork{M<:AbstractMatrix{Float64}}
    A::M                                 # Matrix{Float64} or SparseMatrixCSC
    b::Vector{Float64}
    lambda::Float64
    noise_variance::Float64
    signed::Bool                         # false means x >= 0
    column_squares::Vector{Float64}      # squared 2-norm of every column of A
    gram_form::Bool                      # true for a dense A with m > n
    K::Matrix{Float64}                   # A'A in Gram form, 0x0 otherwise
end

function _GibbsWork(A, b, lambda, noise_variance, signed)
    gram_form = A isa Matrix && size(A, 1) > size(A, 2)
    return _GibbsWork(A, b, lambda, noise_variance, signed,
                      vec(sum(abs2, A; dims=1)), gram_form,
                      gram_form ? A' * A : zeros(0, 0))
end

# dot(A[:, i], r) and r .-= change .* A[:, i], for dense and sparse columns.
_column_dot(A::Matrix, i, r) = dot(view(A, :, i), r)
function _column_dot(A::SparseMatrixCSC, i, r)
    rows, values = rowvals(A), nonzeros(A)
    total = 0.0
    @inbounds for k in nzrange(A, i)
        total += values[k] * r[rows[k]]
    end
    return total
end

_column_subtract!(r, change, A::Matrix, i) = (@views r .-= change .* A[:, i]; r)
function _column_subtract!(r, change, A::SparseMatrixCSC, i)
    rows, values = rowvals(A), nonzeros(A)
    @inbounds for k in nzrange(A, i)
        r[rows[k]] -= change * values[k]
    end
    return r
end

# Residual form keeps r = b - A*x (length m); Gram form keeps A'(b - A*x)
# (length n). Either lets one coordinate update cost one column operation.
_gibbs_state(p::_GibbsWork, x) =
    p.gram_form ? p.A' * (p.b - p.A * x) : p.b - p.A * x

# One systematic Gibbs sweep over all coefficients; updates x and state.
function _gibbs_sweep!(rng, p::_GibbsWork, x, state)
    for i in eachindex(x)
        d = p.column_squares[i]
        if d == 0                        # zero column: only the prior acts
            magnitude = randexp(rng) * p.noise_variance / p.lambda
            x[i] = p.signed && rand(rng, Bool) ? -magnitude : magnitude
            continue
        end
        correlation = p.gram_form ? state[i] : _column_dot(p.A, i, state)
        center = x[i] + correlation / d
        sigma = sqrt(p.noise_variance / d)
        new = p.signed ?
              _rand_two_sided_normal(rng, center, p.lambda / d, sigma) :
              _rand_truncated_normal(rng, center - p.lambda / d, sigma)
        change = new - x[i]
        if change != 0
            if p.gram_form
                @views state .-= change .* p.K[:, i]
            else
                _column_subtract!(state, change, p.A, i)
            end
            x[i] = new
        end
    end
    return x
end

# One chain: returns an n × draws matrix of stored coefficient vectors.
function _gibbs_chain(rng, p::_GibbsWork, start; warmup, draws, thin)
    n = length(start)
    scale = 0.1 * p.noise_variance / p.lambda
    x = p.signed ? start .+ scale .* randn(rng, n) :
                   max.(start, 0.0) .+ scale .* randexp(rng, n)
    state = _gibbs_state(p, x)
    stored = zeros(n, draws)
    for it in 1:(warmup + draws * thin)
        if it % 100 == 0                 # remove accumulated rounding
            state = _gibbs_state(p, x)
        end
        _gibbs_sweep!(rng, p, x, state)
        if it > warmup && (it - warmup) % thin == 0
            stored[:, (it - warmup) ÷ thin] = x
        end
    end
    return stored
end

# Split R-hat of one scalar quantity; `x` is a draws × chains matrix.
function _split_rhat(x::AbstractMatrix)
    half = size(x, 1) ÷ 2
    half >= 2 || return NaN
    parts = hcat(x[1:half, :], x[half+1:2half, :])
    within = mean(var(parts; dims=1))
    between = var(vec(mean(parts; dims=1)))
    within > 0 || return NaN
    return sqrt(((half - 1) / half * within + between) / within)
end

# Effective sample size of one scalar quantity (Geyer's initial positive
# sequence on the chain-averaged autocorrelation); `x` is draws × chains.
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

"""
    Gibbs(; warmup=1_000, draws=2_000, chains=4, thin=5)

Coordinate Gibbs sampler for [`sample_posterior`](@ref). Each chain runs
`warmup + draws*thin` sweeps over all coefficients and stores every `thin`-th
sweep after warm-up, so `draws` is the number of stored draws per chain. There
is no step size and nothing to tune; if the reported effective sample sizes
are too small, increase `thin` or `draws`.
"""
struct Gibbs
    warmup::Int
    draws::Int
    chains::Int
    thin::Int
end

function Gibbs(; warmup=1_000, draws=2_000, chains=4, thin=5)
    all(v -> v isa Integer, (warmup, draws, chains, thin)) ||
        throw(ArgumentError("warmup, draws, chains and thin must be integers"))
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative"))
    draws >= 4 || throw(ArgumentError("draws must be at least 4"))
    chains >= 1 || throw(ArgumentError("chains must be at least 1"))
    thin >= 1 || throw(ArgumentError("thin must be at least 1"))
    return Gibbs(Int(warmup), Int(draws), Int(chains), Int(thin))
end

"""
Posterior draws returned by [`sample_posterior`](@ref).

- `x`: `n × draws × chains` array of coefficient draws.
- `lambda`, `noise_variance`, `constraint`: what the draws are conditional on.
- `rhat`: split R-hat per coefficient; values near 1 mean the chains agree.
- `ess`: effective sample size per coefficient, out of `draws*chains`.

Summarize with [`posterior_mean`](@ref), [`posterior_std`](@ref),
[`posterior_quantile`](@ref) and [`credible_interval`](@ref).
"""
struct PosteriorSamples
    x::Array{Float64,3}
    lambda::Float64
    noise_variance::Float64
    constraint::Symbol
    rhat::Vector{Float64}
    ess::Vector{Float64}
end

function Base.show(io::IO, s::PosteriorSamples)
    n, draws, chains = size(s.x)
    rhat = filter(isfinite, s.rhat)
    ess = filter(isfinite, s.ess)
    print(io, "PosteriorSamples(", n, " coefficients, ", draws, " draws × ",
          chains, " chains, ", s.constraint, ", lambda=", s.lambda,
          ", noise_variance=", s.noise_variance,
          ", max rhat=", isempty(rhat) ? NaN : round(maximum(rhat); digits=3),
          ", min ess=", isempty(ess) ? NaN : round(minimum(ess); digits=1),
          ")")
end

Base.show(io::IO, ::MIME"text/plain", s::PosteriorSamples) = show(io, s)

"""
    sample_posterior(problem::LassoProblem, [Gibbs()], lambda;
                     noise_variance=1.0, x0=nothing, rng=Random.default_rng())
    sample_posterior(problem::LassoProblem, [Gibbs()], result::SolveResult; ...)

Draw coefficient vectors from the posterior

    p(x | b) ∝ exp(-(0.5‖Ax - b‖² + λ‖x‖₁) / noise_variance)

restricted to `x ≥ 0` when `problem.constraint == :nonnegative`. Its most
probable point is the LASSO solution at the same `lambda` and constraint, so
the draws describe the uncertainty around one point of a LASSO path. Passing a
`SolveResult` uses its `lambda` and starts the chains at its solution.

`noise_variance` is the variance of the noise in each entry of `b`, in the
units of `b` as passed in; the default `1` is correct for whitened data. The
draws are conditional on `lambda` and `noise_variance`: uncertainty about
either is not included. Posterior draws are never exactly zero.

Supported: least-squares loss, L1 penalty, `lambda > 0`, and a
`Matrix{Float64}` or `SparseMatrixCSC{Float64}` on the CPU. A seeded `rng`
makes the result reproducible. A warning is issued when the chains have not
mixed (`rhat > 1.05` or `ess < 100` for some coefficient).
"""
function sample_posterior(problem::LassoProblem, method::Gibbs, lambda::Real;
                          noise_variance=1.0, x0=nothing,
                          rng=Random.default_rng())
    problem.loss isa LeastSquares ||
        throw(ArgumentError("sample_posterior supports only the least-squares loss"))
    problem.penalty isa L1 ||
        throw(ArgumentError("sample_posterior supports only the L1 penalty"))
    A, b = problem.A, problem.b
    (A isa Matrix{Float64} || A isa SparseMatrixCSC{Float64}) &&
        b isa Vector{Float64} ||
        throw(ArgumentError("sample_posterior needs a Matrix{Float64} or " *
                            "SparseMatrixCSC{Float64} and a Vector{Float64} " *
                            "on the CPU"))
    isfinite(lambda) && lambda > 0 ||
        throw(ArgumentError("lambda must be finite and positive; the " *
                            "posterior is not defined without the penalty"))
    noise_variance isa Real && isfinite(noise_variance) && noise_variance > 0 ||
        throw(ArgumentError("noise_variance must be finite and positive"))
    n = size(A, 2)
    if !isnothing(x0)
        length(x0) == n ||
            throw(DimensionMismatch("x0 must have one entry per column of A"))
        all(isfinite, x0) || throw(ArgumentError("x0 must be finite"))
    end

    work = _GibbsWork(A, b, Float64(lambda), Float64(noise_variance),
                      problem.constraint == :signed)
    start = isnothing(x0) ? zeros(n) : Vector{Float64}(x0)
    seeds = rand(rng, UInt64, method.chains)
    x = zeros(n, method.draws, method.chains)
    for k in 1:method.chains
        x[:, :, k] = _gibbs_chain(Xoshiro(seeds[k]), work, start;
                                  warmup=method.warmup, draws=method.draws,
                                  thin=method.thin)
    end

    rhat = [_split_rhat(view(x, i, :, :)) for i in 1:n]
    ess = [_effective_sample_size(view(x, i, :, :)) for i in 1:n]
    finite_rhat = filter(isfinite, rhat)
    finite_ess = filter(isfinite, ess)
    if !isempty(finite_rhat) && !isempty(finite_ess)
        max_rhat, min_ess = maximum(finite_rhat), minimum(finite_ess)
        if max_rhat > 1.05 || min_ess < 100
            @warn "Posterior chains have not mixed; increase warmup, draws or thin" max_rhat min_ess
        end
    end
    return PosteriorSamples(x, Float64(lambda), Float64(noise_variance),
                            problem.constraint, rhat, ess)
end

sample_posterior(problem::LassoProblem, method::Gibbs, result::SolveResult;
                 kwargs...) =
    sample_posterior(problem, method, result.lambda; x0=result.x, kwargs...)

sample_posterior(problem::LassoProblem, lambda_or_result; kwargs...) =
    sample_posterior(problem, Gibbs(), lambda_or_result; kwargs...)

"""Posterior mean of every coefficient, pooled over all chains."""
posterior_mean(s::PosteriorSamples) = vec(mean(s.x; dims=(2, 3)))

"""Posterior standard deviation of every coefficient, pooled over all chains."""
posterior_std(s::PosteriorSamples) = vec(std(s.x; dims=(2, 3)))

"""
    posterior_quantile(samples, p)

The `p`-quantile (`0 ≤ p ≤ 1`) of every coefficient, pooled over all chains.
"""
function posterior_quantile(s::PosteriorSamples, p::Real)
    0 <= p <= 1 || throw(ArgumentError("p must be between 0 and 1"))
    return [quantile(vec(view(s.x, i, :, :)), p) for i in axes(s.x, 1)]
end

"""
    credible_interval(samples; level=0.9)

Central credible interval of every coefficient as `(lower=..., upper=...)`:
the `(1 - level)/2` and `(1 + level)/2` posterior quantiles.
"""
function credible_interval(s::PosteriorSamples; level=0.9)
    0 < level < 1 || throw(ArgumentError("level must be between 0 and 1"))
    return (lower=posterior_quantile(s, (1 - level) / 2),
            upper=posterior_quantile(s, (1 + level) / 2))
end

export Gibbs, PosteriorSamples, sample_posterior
export posterior_mean, posterior_std, posterior_quantile, credible_interval
