"""
    lasso_kkt_residual(problem, x, lambda)

Largest coordinatewise violation of the exact KKT conditions at `x`,
using the gradient `g = A'ψ(Ax-b)` of the problem's loss. For the Huber
penalty the condition is `g + λψ_δ(x) = 0` (or `≥ 0` at `x = 0` when
nonnegative).
This is evaluated at the **returned coefficients**, not a FISTA extrapolate.
For `constraint=:nonnegative`, the feasible-point condition at zero is
`gradient[j] + lambda ≥ 0`.
"""
function lasso_kkt_residual(problem::LassoProblem, x::AbstractVector, lambda)
    _validate_lambda(lambda)
    length(x) == size(problem.A, 2) ||
        throw(DimensionMismatch("x must have one entry per column of A"))
    g = adjoint(problem.A) *
        _loss_derivative!(problem.loss, problem.A * x - problem.b)
    if problem.constraint == :nonnegative
        minimum(x) >= 0 || throw(ArgumentError("x violates nonnegativity"))
    end
    nonnegative = problem.constraint == :nonnegative
    return maximum(_kkt_violation.(problem.penalty, x, g, lambda, nonnegative))
end

# Internal no-allocation check using the FISTA workspace. The gradient at y has
# already served its purpose after the prox step and may be overwritten here.
function _lasso_kkt_residual!(problem::LassoProblem, x, lambda,
                              residual, gradient, violations)
    mul!(residual, problem.A, x)
    @. residual = residual - problem.b
    _loss_derivative!(problem.loss, residual)
    mul!(gradient, adjoint(problem.A), residual)
    nonnegative = problem.constraint == :nonnegative
    violations .= _kkt_violation.(problem.penalty, x, gradient, lambda,
                                  nonnegative)
    return maximum(violations)
end

"""Evaluate the penalized objective `ℓ(Ax-b) + λR(x)` at a coefficient vector."""
function objective_value(problem::LassoProblem, x::AbstractVector, lambda)
    _validate_lambda(lambda)
    length(x) == size(problem.A, 2) ||
        throw(DimensionMismatch("x must have one entry per column of A"))
    problem.constraint == :nonnegative && minimum(x) < 0 &&
        return Inf
    return _loss_value(problem.loss, problem.A * x - problem.b) +
           lambda * _penalty_value(problem.penalty, x)
end

"""
    lasso_dual_gap(problem::LassoProblem, x, lambda)

Nonnegative primal-dual gap for signed or nonnegative penalized LASSO.
For `lambda > 0`, scale the loss derivative `u = ψ(Ax-b)` (the residual for
least squares, the clipped residual for Huber) until the dual constraint is
feasible: `norm(A'u, Inf) <= lambda` for signed coefficients,
or `A'u >= -lambda` for nonnegative coefficients. For the Huber penalty the
bound is `lambda*delta` and its quadratic conjugate is added to the gap.
At `lambda == 0`,
the zero dual vector gives a valid but usually loose bound; the default
stop uses a KKT fallback in that case.
"""
function lasso_dual_gap(problem::LassoProblem, x::AbstractVector, lambda)
    _validate_lambda(lambda)
    length(x) == size(problem.A, 2) ||
        throw(DimensionMismatch("x must have one entry per column of A"))
    problem.constraint == :nonnegative && minimum(x) < 0 &&
        throw(ArgumentError("x violates nonnegativity"))
    residual = problem.A * x - problem.b
    return _lasso_gap_from_residual!(problem, x, lambda, residual,
                                     similar(residual, size(problem.A, 2)))
end

# Unvalidated gap using caller-owned buffers; `residual` and `gradient` are
# overwritten. FISTA passes its workspace, so a gap check allocates no array.
function _lasso_dual_gap!(problem::LassoProblem, x, lambda, residual, gradient)
    mul!(residual, problem.A, x)
    @. residual = residual - problem.b
    return _lasso_gap_from_residual!(problem, x, lambda, residual, gradient)
end

# Gap given `residual = Ax - b`; overwrites `residual` and `gradient`.
function _lasso_gap_from_residual!(problem::LassoProblem, x, lambda, residual,
                                   gradient)
    primal = _loss_value(problem.loss, residual) +
             lambda * _penalty_value(problem.penalty, x)
    iszero(lambda) && return primal
    u = _loss_derivative!(problem.loss, residual)
    mul!(gradient, adjoint(problem.A), u)
    scale, penalty_dual = _penalty_dual(problem.penalty, gradient, lambda,
                                        problem.constraint)
    gap = primal + (_loss_conjugate(problem.loss, u) + penalty_dual) / scale^2 +
          dot(problem.b, u) / scale
    return max(zero(gap), gap)
end

_dual_gap_tolerance(b, stop::DualGapStop) = stop.tol * sum(abs2, b)
