"""
Data-fit loss `ℓ(r)` applied to the residual `r = Ax - b` (summed over rows).
Select it with the `loss` keyword of a problem constructor. Each loss provides
its value, its derivative `ψ(r)`, and a dual term; all are elementwise so the
same code runs on CPU and device arrays.
"""
abstract type DataLoss end

"""`LeastSquares()`: the default loss `½‖r‖²`."""
struct LeastSquares <: DataLoss end

"""
    Huber(delta)

Huber loss: `ρ(r) = r²/2` for `|r| ≤ delta`, otherwise
`delta*(|r| - delta/2)`. Its derivative is `clamp(r, -delta, delta)`, so the
gradient Lipschitz constant is `‖A‖²`, as for least squares. `delta` is in
residual units: weight (whiten) the observations first, so that a value such
as `1.345` has its usual meaning. As `delta → ∞` the loss becomes
least squares.
"""
struct Huber{T<:AbstractFloat} <: DataLoss
    delta::T
    function Huber{T}(delta) where {T<:AbstractFloat}
        isfinite(delta) && delta > 0 ||
            throw(ArgumentError("Huber delta must be finite and positive"))
        return new{T}(T(delta))
    end
end

Huber(delta::Real) = Huber{float(typeof(delta))}(delta)

Huber(; delta) = Huber(delta)

# Match the loss parameter to the problem's element type (e.g. Float32 on GPU).
_convert_loss(loss::LeastSquares, ::Type) = loss
_convert_loss(loss::Huber, ::Type{T}) where {T} = Huber{T}(loss.delta)
_convert_loss(loss, ::Type) =
    throw(ArgumentError("loss must be LeastSquares() or Huber(delta)"))

_huber_rho(r, delta) = ifelse(abs(r) <= delta, r^2 / 2,
                              delta * (abs(r) - delta / 2))

"""Loss value `∑ℓ(rᵢ)` at a residual vector."""
_loss_value(::LeastSquares, r) = sum(abs2, r) / 2
function _loss_value(loss::Huber, r)
    delta = loss.delta
    return sum(v -> _huber_rho(v, delta), r)
end

"""Overwrite a residual with the loss derivative `ψ(r)` and return it."""
_loss_derivative!(::LeastSquares, r) = r
function _loss_derivative!(loss::Huber, r)
    delta = loss.delta
    @. r = clamp(r, -delta, delta)
    return r
end

# The Fenchel conjugate of both losses is u²/2 on its domain (|u| ≤ delta for
# Huber). The dual points used by the gap functions are ψ(r)/scale with
# scale ≥ 1, which always lies in that domain.
_loss_conjugate(::DataLoss, u) = sum(abs2, u) / 2

"""Correlations `A'ψ(b)`, the negative gradient at zero coefficients."""
_loss_correlations(A, b, ::LeastSquares) = adjoint(A) * b
_loss_correlations(A, b, loss::Huber) =
    adjoint(A) * _loss_derivative!(loss, copy(b))

# Dual-gap tolerance scale `2ℓ(b)`; equals ‖b‖² for least squares. For Huber
# this avoids letting large outliers loosen the stop as ‖b‖² would.
_dual_gap_tolerance(b, loss::DataLoss, stop) = stop.tol * 2 * _loss_value(loss, b)
