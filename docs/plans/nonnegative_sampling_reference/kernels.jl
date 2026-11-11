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

struct _NonnegativeWork
    A::Matrix{Float64}
    b::Vector{Float64}
    lambda::Float64
    noise_variance::Float64
    column_squares::Vector{Float64}      # squared 2-norm of every column of A
    gram_form::Bool                      # true when m > n
    K::Matrix{Float64}                   # A'A in Gram form, 0x0 otherwise
end

function _NonnegativeWork(A, b, lambda, noise_variance)
    gram_form = size(A, 1) > size(A, 2)
    return _NonnegativeWork(A, b, lambda, noise_variance,
                            vec(sum(abs2, A; dims=1)), gram_form,
                            gram_form ? A' * A : zeros(0, 0))
end

# Residual form keeps r = b - A*x (length m); Gram form keeps A'(b - A*x)
# (length n). Either lets one coordinate update cost one column operation.
_nonnegative_state(p::_NonnegativeWork, x) =
    p.gram_form ? p.A' * (p.b - p.A * x) : p.b - p.A * x

# One systematic Gibbs sweep over all coefficients; updates x and state.
function _nonnegative_sweep!(rng, p::_NonnegativeWork, x, state)
    for i in eachindex(x)
        d = p.column_squares[i]
        if d == 0                        # zero column: only the prior acts
            x[i] = randexp(rng) * p.noise_variance / p.lambda
            continue
        end
        correlation = p.gram_form ? state[i] : dot(view(p.A, :, i), state)
        mu = x[i] + (correlation - p.lambda) / d
        new = _rand_truncated_normal(rng, mu, sqrt(p.noise_variance / d))
        change = new - x[i]
        if change != 0
            if p.gram_form
                @views state .-= change .* p.K[:, i]
            else
                @views state .-= change .* p.A[:, i]
            end
            x[i] = new
        end
    end
    return x
end

# One chain: returns an n × draws matrix of stored coefficient vectors.
function _nonnegative_chain(rng, p::_NonnegativeWork, start; warmup, draws, thin)
    n = length(start)
    x = max.(start, 0.0) .+ randexp(rng, n) .* (0.1 * p.noise_variance / p.lambda)
    state = _nonnegative_state(p, x)
    stored = zeros(n, draws)
    for it in 1:(warmup + draws * thin)
        if it % 100 == 0                 # remove accumulated rounding
            state = _nonnegative_state(p, x)
        end
        _nonnegative_sweep!(rng, p, x, state)
        if it > warmup && (it - warmup) % thin == 0
            stored[:, (it - warmup) ÷ thin] = x
        end
    end
    return stored
end
