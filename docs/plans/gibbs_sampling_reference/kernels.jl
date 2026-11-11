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
