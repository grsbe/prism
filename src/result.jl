"""
    DualGapStop(; tol=1e-4, check_every=1, maxiter=10_000)

FISTA stopping for an unnormalized least-squares objective. First require
`norm(x_new - x_old, Inf) <= tol * norm(x_new, Inf)`, then certify the
returned iterate with a primal-dual gap no larger than `tol * norm(b)^2`.
The latter is equivalent to scikit-learn's `tol * norm(b)^2/n` for its
objective divided by `n`. Models without a usable dual certificate use a
documented optimality-residual fallback.
"""
struct DualGapStop{T<:AbstractFloat}
    tol::T
    check_every::Int
    maxiter::Int
end

function DualGapStop(; tol=1e-4, check_every=1, maxiter=10_000)
    _validate_stop_number(tol, "tol")
    check_every isa Integer && check_every > 0 ||
        throw(ArgumentError("check_every must be a positive integer"))
    maxiter isa Integer && maxiter > 0 ||
        throw(ArgumentError("maxiter must be a positive integer"))
    T = float(typeof(tol))
    return DualGapStop{T}(T(tol), Int(check_every), Int(maxiter))
end

"""KKT-based stopping; both `atol` and `rtol*lambda` are in gradient units."""
struct KKTStop{T<:AbstractFloat}
    atol::T
    rtol::T
    check_every::Int
    maxiter::Int
end

function KKTStop(; atol=1e-10, rtol=1e-3, check_every=10, maxiter=10_000)
    _validate_stop_number(atol, "atol")
    _validate_stop_number(rtol, "rtol")
    check_every isa Integer && check_every > 0 ||
        throw(ArgumentError("check_every must be a positive integer"))
    maxiter isa Integer && maxiter > 0 ||
        throw(ArgumentError("maxiter must be a positive integer"))
    T = float(promote_type(typeof(atol), typeof(rtol)))
    return KKTStop{T}(T(atol), T(rtol), Int(check_every), Int(maxiter))
end

"""
    GradientMappingStop(; atol=0, rtol=1e-3, check_every=1,
                        maxiter=10_000, warn_optimality_ratio=0.1)

Optional FISTA stopping rule. At the extrapolated iterate `y`, test
`norm((y - prox(y - step*gradient(y)))/step, Inf)` against a gradient-scaled
tolerance. Signed LASSO and the other FISTA families use
`atol + rtol*max(norm(gradient(y), Inf), eps(step))`. For nonnegative LASSO
with positive lambda, the scale ignores positive gradients at coefficients
that remain zero and is at least lambda. This is **not** the KKT
residual at the returned coefficients; that residual is still reported.
`warn_optimality_ratio` controls only the constrained FISTA
projected-gradient warning. For penalized models, a final KKT/λ warning at
0.1 is issued regardless of the selected stop rule.
"""
struct GradientMappingStop{T<:AbstractFloat}
    atol::T
    rtol::T
    check_every::Int
    maxiter::Int
    warn_optimality_ratio::T
end

function GradientMappingStop(; atol=0, rtol=1e-3, check_every=1,
                             maxiter=10_000, warn_optimality_ratio=0.1)
    _validate_stop_number(atol, "atol")
    _validate_stop_number(rtol, "rtol")
    check_every isa Integer && check_every > 0 ||
        throw(ArgumentError("check_every must be a positive integer"))
    maxiter isa Integer && maxiter > 0 ||
        throw(ArgumentError("maxiter must be a positive integer"))
    _validate_stop_number(warn_optimality_ratio, "warn_optimality_ratio")
    T = float(promote_type(typeof(atol), typeof(rtol),
                           typeof(warn_optimality_ratio)))
    return GradientMappingStop{T}(T(atol), T(rtol), Int(check_every),
                                  Int(maxiter), T(warn_optimality_ratio))
end

"""
Owned snapshot of a solve. `x` is copied out of the reusable solver workspace.
`converged` means that the stated stopping conditions were met, not that the
physical model or selected lambda is scientifically correct.
"""
struct SolveResult{V,T,D}
    x::V
    lambda::T
    objective::T
    residual_norm::T
    kkt_residual::T
    kkt_ratio::T
    converged::Bool
    stop_reason::Symbol
    iterations::Int
    diagnostics::D
end

_kkt_tolerance(stop::KKTStop, lambda) = stop.atol + stop.rtol * lambda

function _warn_fista_optimality(ratio, stop::GradientMappingStop, label)
    if ratio > stop.warn_optimality_ratio
        @warn "FISTA returned with a large final optimality residual" quantity=label ratio=ratio threshold=stop.warn_optimality_ratio
    end
    return nothing
end

_warn_fista_optimality(ratio, stop::KKTStop, label) = nothing

function _warn_kkt_ratio(ratio, lambda, reason)
    if !iszero(lambda) && ratio > 0.1
        @warn "Solver returned with a large final optimality residual" quantity=:kkt_over_lambda ratio threshold=0.1 stop_reason=reason
    end
    return nothing
end

function _make_result(problem, x, lambda, converged, reason, iterations, diagnostics)
    owned_x = copy(x)
    residual = problem.A * owned_x - problem.b
    residual_norm = norm(residual)
    kkt = lasso_kkt_residual(problem, owned_x, lambda)
    ratio = iszero(lambda) ? (iszero(kkt) ? zero(kkt) : oftype(kkt, Inf)) :
            kkt / (lambda * _penalty_slope(problem.penalty))
    objective = _loss_value(problem.loss, residual) +
                lambda * _penalty_value(problem.penalty, owned_x)
    T = promote_type(typeof(float(lambda)), typeof(objective))
    _warn_kkt_ratio(ratio, lambda, reason)
    return SolveResult(owned_x, T(lambda), T(objective), T(residual_norm),
                       T(kkt), T(ratio), converged, reason, iterations, diagnostics)
end
