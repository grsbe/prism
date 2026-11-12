"""
`L1()`: the default coefficient penalty `λ‖x‖₁`, which produces exact zeros.
Select a penalty with the `penalty` keyword of `LassoProblem` (and for the
sparse block of `BlockLassoRidgeProblem`). The alternative is
`penalty=Huber(delta)`, i.e. `λ∑ρ_δ(xⱼ)`: quadratic (ridge-like) for
`|xⱼ| ≤ delta` and linear with slope `λ*delta` beyond. It does **not**
produce sparsity; zeros arise only from the nonnegativity constraint.
"""
struct L1 end

const CoefficientPenalty = Union{L1,Huber}

# Pass a penalty to broadcast kernels as a scalar (it is isbits, so GPU-safe).
Base.broadcastable(p::CoefficientPenalty) = Ref(p)

_convert_penalty(p::L1, ::Type) = p
_convert_penalty(p::Huber, ::Type{T}) where {T} = Huber{T}(p.delta)
_convert_penalty(p, ::Type) =
    throw(ArgumentError("penalty must be L1() or Huber(delta)"))

"""Penalty value per unit λ: `‖x‖₁` or `∑ρ_δ(xⱼ)`."""
_penalty_value(::L1, x) = sum(abs, x)
_penalty_value(p::Huber, x) = _loss_value(p, x)

# Largest pull per unit λ; KKT residuals are reported relative to λ times this.
_penalty_slope(::L1) = 1
_penalty_slope(p::Huber) = p.delta

"""Write `prox_{threshold*penalty + I_C}(v)` into `x` (elementwise)."""
function _penalty_prox!(::L1, x, v, threshold, constraint)
    if constraint == :nonnegative
        @. x = max(v - threshold, zero(eltype(x)))
    else
        @. x = sign(v) * max(abs(v) - threshold, zero(eltype(x)))
    end
    return x
end

function _penalty_prox!(p::Huber, x, v, threshold, constraint)
    delta = p.delta
    shrink = 1 + threshold
    # Quadratic region if the result stays within delta, otherwise linear.
    @. x = ifelse(abs(v) <= delta * shrink, v / shrink,
                  v - threshold * delta * sign(v))
    if constraint == :nonnegative  # separable 1D: clipping the prox is exact
        @. x = max(x, zero(eltype(x)))
    end
    return x
end

"""KKT violation of one coefficient `x` with smooth-part gradient `g`."""
function _kkt_violation(::L1, x, g, lambda, nonnegative)
    nonnegative && return ifelse(x > 0, abs(g + lambda), max(-g - lambda, 0))
    return ifelse(x > 0, abs(g + lambda),
                  ifelse(x < 0, abs(g - lambda), max(abs(g) - lambda, 0)))
end

function _kkt_violation(p::Huber, x, g, lambda, nonnegative)
    s = g + lambda * clamp(x, -p.delta, p.delta)  # Huber is differentiable
    return nonnegative ? ifelse(x > 0, abs(s), max(-s, 0)) : abs(s)
end

"""
Dual feasibility scale and conjugate term for the coefficient penalty, given
`g = A'u` at an unscaled dual point `u`. The gap adds `term / scale^2`.
L1: `‖g‖∞ ≤ λ` (or `-g ≤ λ`), conjugate zero. Huber: the conjugate of
`λρ_δ` at `w = -g` is `w²/(2λ)` for `|w| ≤ λδ` (only `w₊` for `x ≥ 0`).
"""
function _penalty_dual(::L1, g, lambda, constraint)
    bound = constraint == :signed ? maximum(abs, g) :
            max(zero(lambda), -minimum(g))
    return max(one(float(lambda)), bound / lambda), zero(float(lambda))
end

# `∑ min(gⱼ, 0)²` as one reduction, without a temporary array.
_negative_part_energy(g) = sum(v -> abs2(min(v, zero(v))), g)

function _penalty_dual(p::Huber, g, lambda, constraint)
    if constraint == :signed
        bound, energy = maximum(abs, g), sum(abs2, g)
    else
        bound = max(zero(lambda), -minimum(g))
        energy = _negative_part_energy(g)
    end
    scale = max(one(float(lambda)), bound / (lambda * p.delta))
    return scale, energy / (2 * lambda)
end
