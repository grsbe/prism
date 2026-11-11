"""
    TVProblem(A, b, image_size, tv_weight; constraint=:signed, isotropic=true)

Minimize `½‖Ax-b‖² + λ₁‖x‖₁ + tv_weight*TV(x)` on a column-major 2D
grid. `constraint=:nonnegative` imposes `x ≥ 0`.
"""
struct TVProblem{MA<:AbstractMatrix,VB<:AbstractVector,T}
    A::MA
    b::VB
    image_size::Tuple{Int,Int}
    tv_weight::T
    constraint::Symbol
    isotropic::Bool
end

function TVProblem(A::AbstractMatrix, b::AbstractVector, image_size::Tuple,
                   tv_weight; constraint=:signed, isotropic=true)
    LassoProblem(A, b; constraint)
    A isa StridedMatrix && b isa StridedVector ||
        throw(ArgumentError("TV PDHG currently requires CPU dense arrays"))
    length(image_size) == 2 && all(d -> d isa Integer && d > 0, image_size) ||
        throw(ArgumentError("image_size must contain two positive integers"))
    dims = (Int(image_size[1]), Int(image_size[2]))
    prod(dims) == size(A, 2) ||
        throw(DimensionMismatch("A must have one column per image pixel"))
    _validate_lambda(tv_weight)
    isotropic isa Bool || throw(ArgumentError("isotropic must be Bool"))
    return TVProblem(A, b, dims, eltype(b)(tv_weight), constraint, isotropic)
end

function _tv_forward!(gx, gy, x, dims)
    height, width = dims
    @inbounds for j in 1:width, i in 1:height
        k = i + (j - 1) * height
        gx[k] = i < height ? x[k + 1] - x[k] : zero(eltype(x))
        gy[k] = j < width ? x[k + height] - x[k] : zero(eltype(x))
    end
    return nothing
end

function _tv_adjoint_add!(out, px, py, dims)
    height, width = dims
    @inbounds for j in 1:width, i in 1:height
        k = i + (j - 1) * height
        i > 1 && (out[k] += px[k - 1])
        i < height && (out[k] -= px[k])
        j > 1 && (out[k] += py[k - height])
        j < width && (out[k] -= py[k])
    end
    return out
end

"""Forward-difference isotropic or anisotropic TV in column-major order."""
function total_variation2d(x::AbstractVector, dims::Tuple{Int,Int};
                            isotropic=true)
    length(x) == prod(dims) ||
        throw(DimensionMismatch("x must have prod(image_size) entries"))
    gx, gy = similar(x), similar(x)
    _tv_forward!(gx, gy, x, dims)
    return isotropic ? sum(hypot.(gx, gy)) : sum(abs, gx) + sum(abs, gy)
end

"""Configuration for primal-dual hybrid gradient on TV problems."""
struct PDHG{S}
    step::S
    stop::KKTStop
    theta::Float64
end

function PDHG(; step=:auto, stop=KKTStop(), theta=1.0)
    theta isa Real && isfinite(theta) && 0 <= theta <= 1 ||
        throw(ArgumentError("theta must lie in [0,1]"))
    return PDHG(step, stop, Float64(theta))
end

"""Owned TV solution and independently evaluated primal-dual conditions."""
struct TVResult{V,T,D}
    x::V
    lambda::T
    tv_weight::T
    objective::T
    residual_norm::T
    stationarity_residual::T
    data_dual_residual::T
    tv_complementarity_gap::T
    dual_feasibility_residual::T
    converged::Bool
    stop_reason::Symbol
    iterations::Int
    diagnostics::D
end

mutable struct TVPDHGSolver{P,V,R,T,S}
    problem::P
    x::V
    xbar::V
    previous::V
    y::R
    px::V
    py::V
    gx::V
    gy::V
    gradient::V
    residual::R
    step::T
    stop::S
    theta::T
end

function TVPDHGSolver(problem::TVProblem; step=:auto, stop=KKTStop(),
                      theta=1.0, x0=nothing)
    PDHG(; step, stop, theta) # validate configuration
    T = eltype(problem.b)
    x = zeros(T, size(problem.A, 2))
    if !isnothing(x0)
        length(x0) == length(x) || throw(DimensionMismatch("x0 has wrong length"))
        copyto!(x, x0)
    end
    all(isfinite, x) || throw(ArgumentError("x0 must be finite"))
    problem.constraint == :nonnegative && minimum(x) < 0 &&
        throw(ArgumentError("x0 must be nonnegative"))
    # ‖[A; ∇]‖² ≤ ‖A‖² + 8 on a nearest-neighbor 2D grid.
    t = if step === :auto
        T(0.99 / sqrt(_squared_norm_bound(problem.A, problem.b) + 8))
    elseif step === :exact
        T(0.99 / sqrt(opnorm(problem.A)^2 + 8))
    elseif step === :frobenius
        T(0.99 / sqrt(sum(abs2, problem.A) + 8))
    elseif step isa Real && isfinite(step) && step > 0
        T(step)
    else
        throw(ArgumentError("step must be :auto, :exact, :frobenius, or positive"))
    end
    t > 0 || throw(ArgumentError("step underflowed to zero"))
    return TVPDHGSolver(problem, x, copy(x), similar(x),
                        zeros(T, length(problem.b)), zeros(T, length(x)),
                        zeros(T, length(x)), similar(x), similar(x),
                        similar(x), similar(problem.b), t, stop, T(theta))
end

function _tv_project_dual!(solver::TVPDHGSolver)
    p, t, weight = solver.problem, solver.step, solver.problem.tv_weight
    _tv_forward!(solver.gx, solver.gy, solver.xbar, p.image_size)
    @inbounds for k in eachindex(solver.px)
        px = solver.px[k] + t * solver.gx[k]
        py = solver.py[k] + t * solver.gy[k]
        if p.isotropic
            scale = weight == 0 ? oftype(weight, Inf) :
                    max(one(weight), hypot(px, py) / weight)
            solver.px[k] = px / scale
            solver.py[k] = py / scale
        else
            solver.px[k] = clamp(px, -weight, weight)
            solver.py[k] = clamp(py, -weight, weight)
        end
    end
    return nothing
end

function _tv_conditions(solver::TVPDHGSolver, lambda)
    p, x = solver.problem, solver.x
    residual = p.A * x - p.b
    data_dual = norm(solver.y - residual, Inf)
    gradient = adjoint(p.A) * solver.y
    _tv_adjoint_add!(gradient, solver.px, solver.py, p.image_size)
    if p.constraint == :nonnegative
        violations = ifelse.(x .> 0, abs.(gradient .+ lambda),
                             max.(-gradient .- lambda, 0))
    else
        violations = ifelse.(x .> 0, abs.(gradient .+ lambda),
                      ifelse.(x .< 0, abs.(gradient .- lambda),
                              max.(abs.(gradient) .- lambda, 0)))
    end
    gx, gy = similar(x), similar(x)
    _tv_forward!(gx, gy, x, p.image_size)
    tv = p.isotropic ? sum(hypot.(gx, gy)) : sum(abs, gx) + sum(abs, gy)
    complementarity = max(p.tv_weight * tv - dot(solver.px, gx) -
                          dot(solver.py, gy), zero(eltype(x)))
    dual_feasibility = p.isotropic ?
        maximum(max.(hypot.(solver.px, solver.py) .- p.tv_weight, 0)) :
        max(maximum(max.(abs.(solver.px) .- p.tv_weight, 0)),
            maximum(max.(abs.(solver.py) .- p.tv_weight, 0)))
    return (stationarity=maximum(violations), data_dual=data_dual,
            complementarity=complementarity,
            dual_feasibility=dual_feasibility, tv=tv,
            residual_norm=norm(residual))
end

function _tv_converged(solver, lambda, conditions)
    stop, p = solver.stop, solver.problem
    # At lambda == 0 the L1 scale vanishes; use the TV weight (or, with no
    # regularizer at all, the data gradient at zero) as the gradient scale.
    stationarity_scale = !iszero(lambda) ? lambda :
                         !iszero(p.tv_weight) ? p.tv_weight :
                         max(norm(adjoint(p.A) * p.b, Inf), 1)
    stationarity_tol = stop.atol + stop.rtol * stationarity_scale
    data_tol = stop.atol + stop.rtol * max(norm(p.b, Inf), 1)
    gap_tol = stop.atol * max(norm(solver.x, 1), 1) +
              stop.rtol * max(p.tv_weight * conditions.tv, 1)
    dual_tol = stop.atol + stop.rtol * p.tv_weight
    return conditions.stationarity <= stationarity_tol &&
           conditions.data_dual <= data_tol &&
           conditions.complementarity <= gap_tol &&
           conditions.dual_feasibility <= dual_tol
end

function _tv_result(solver, lambda, converged, reason, iterations)
    p = solver.problem
    x = copy(solver.x)
    conditions = _tv_conditions(solver, lambda)
    objective = 0.5 * conditions.residual_norm^2 +
                lambda * sum(abs, x) + p.tv_weight * conditions.tv
    T = promote_type(typeof(objective), typeof(float(lambda)))
    return TVResult(x, T(lambda), T(p.tv_weight), T(objective),
                    T(conditions.residual_norm), T(conditions.stationarity),
                    T(conditions.data_dual), T(conditions.complementarity),
                    T(conditions.dual_feasibility), converged, reason,
                    iterations, (step=solver.step, theta=solver.theta))
end

function solve!(solver::TVPDHGSolver, lambda; warm_start=true)
    _validate_lambda(lambda)
    p, t = solver.problem, solver.step
    if !warm_start
        fill!(solver.x, zero(eltype(solver.x)))
        fill!(solver.y, zero(eltype(solver.y)))
        fill!(solver.px, zero(eltype(solver.px)))
        fill!(solver.py, zero(eltype(solver.py)))
    end
    copyto!(solver.xbar, solver.x)
    conditions = _tv_conditions(solver, lambda)
    if _tv_converged(solver, lambda, conditions)
        return _tv_result(solver, lambda, true, :primal_dual, 0)
    end
    for iteration in 1:solver.stop.maxiter
        copyto!(solver.previous, solver.x)
        mul!(solver.residual, p.A, solver.xbar)
        @. solver.y = (solver.y + t * (solver.residual - p.b)) / (1 + t)
        _tv_project_dual!(solver)
        mul!(solver.gradient, adjoint(p.A), solver.y)
        _tv_adjoint_add!(solver.gradient, solver.px, solver.py, p.image_size)
        if p.constraint == :nonnegative
            @. solver.x = max(solver.x - t * solver.gradient - t * lambda, 0)
        else
            @. solver.x = sign(solver.x - t * solver.gradient) *
                          max(abs(solver.x - t * solver.gradient) - t * lambda, 0)
        end
        if iteration % solver.stop.check_every == 0 ||
           iteration == solver.stop.maxiter
            conditions = _tv_conditions(solver, lambda)
            if _tv_converged(solver, lambda, conditions)
                return _tv_result(solver, lambda, true, :primal_dual,
                                   iteration)
            end
        end
        @. solver.xbar = solver.x + solver.theta *
                         (solver.x - solver.previous)
    end
    return _tv_result(solver, lambda, false, :maxiter, solver.stop.maxiter)
end

initialize(problem::TVProblem, method::PDHG; x0=nothing) =
    TVPDHGSolver(problem; step=method.step, stop=method.stop,
                 theta=method.theta, x0)

function solve(problem::TVProblem, method::PDHG, lambda=0; x0=nothing)
    return solve!(initialize(problem, method; x0), lambda)
end

export TVProblem, TVResult, PDHG, TVPDHGSolver, total_variation2d
