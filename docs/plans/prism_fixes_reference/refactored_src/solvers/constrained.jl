"""
    ConstrainedLassoProblem(A, b, radius; constraint=:signed)

Minimize ½‖Ax-b‖² subject to ‖x‖₁ ≤ radius. With
`constraint=:nonnegative`, also require x ≥ 0. This is an L1-ball
constraint, not the penalized problem with λ = radius.
"""
struct ConstrainedLassoProblem{MA<:AbstractMatrix,VB<:AbstractVector,T}
    A::MA
    b::VB
    radius::T
    constraint::Symbol
end

function ConstrainedLassoProblem(A::AbstractMatrix, b::AbstractVector,
                                 radius; constraint=:signed)
    LassoProblem(A, b; constraint)
    radius isa Real && isfinite(radius) && radius >= 0 ||
        throw(ArgumentError("radius must be finite and nonnegative"))
    return ConstrainedLassoProblem(A, b, eltype(b)(radius), constraint)
end

"""Owned constrained-solve result; optimality is the projected-gradient mapping."""
struct ConstrainedResult{V,T,D}
    x::V
    radius::T
    objective::T
    residual_norm::T
    optimality_residual::T
    converged::Bool
    stop_reason::Symbol
    iterations::Int
    diagnostics::D
end

function _project_l1_ball!(out::AbstractVector, v::AbstractVector,
                           radius, nonnegative::Bool)
    radius == 0 && return fill!(out, zero(eltype(out)))
    magnitudes = nonnegative ? max.(v, zero(eltype(v))) : abs.(v)
    if sum(magnitudes) <= radius
        copyto!(out, nonnegative ? magnitudes : v)
        return out
    end
    sorted = sort(collect(magnitudes); rev=true)
    cumulative = zero(eltype(sorted))
    active = 0
    threshold = zero(eltype(sorted))
    for j in eachindex(sorted)
        cumulative += sorted[j]
        candidate = (cumulative - radius) / j
        if sorted[j] > candidate
            active = j
            threshold = candidate
        end
    end
    active > 0 || error("L1 projection failed to find an active coordinate")
    if nonnegative
        @. out = max(v - threshold, zero(eltype(out)))
    else
        @. out = sign(v) * max(abs(v) - threshold, zero(eltype(out)))
    end
    return out
end

function _constrained_feasible(problem::ConstrainedLassoProblem, x)
    length(x) == size(problem.A, 2) || return false
    tol = 100 * eps(eltype(x)) * max(one(eltype(x)), problem.radius)
    sum(abs, x) <= problem.radius + tol || return false
    return problem.constraint != :nonnegative || minimum(x) >= -tol
end

"""
    constrained_dual_gap(problem, x)

Fenchel primal-dual gap for least squares on an L1 ball. The dual vector
is the data residual; the support function is `radius*norm(A'u, Inf)` for
signed coefficients and `radius*max(0, maximum(-A'u))` for nonnegative
coefficients.
"""
function constrained_dual_gap(problem::ConstrainedLassoProblem, x)
    _constrained_feasible(problem, x) ||
        throw(ArgumentError("x is not feasible for the L1 ball"))
    residual = problem.A * x - problem.b
    gradient = adjoint(problem.A) * residual
    support = problem.constraint == :signed ?
              maximum(abs, gradient) :
              max(zero(eltype(gradient)), maximum(-gradient))
    gap = sum(abs2, residual) + dot(problem.b, residual) +
          problem.radius * support
    return max(zero(gap), gap)
end

"""Projected-gradient KKT mapping norm, zero exactly at a constrained optimum."""
function constrained_optimality_residual(problem::ConstrainedLassoProblem,
                                         x::AbstractVector; step=nothing)
    _constrained_feasible(problem, x) ||
        throw(ArgumentError("x is not feasible for the L1 ball"))
    t = isnothing(step) ? inv(max(opnorm(problem.A)^2, 1)) : step
    t isa Real && isfinite(t) && t > 0 ||
        throw(ArgumentError("step must be positive and finite"))
    gradient = adjoint(problem.A) * (problem.A * x - problem.b)
    projected = similar(x)
    _project_l1_ball!(projected, x .- t .* gradient, problem.radius,
                      problem.constraint == :nonnegative)
    return norm(x - projected) / t
end

_constrained_tolerance(problem, stop) =
    stop.atol + stop.rtol * max(norm(adjoint(problem.A) * problem.b), 1)

function _constrained_result(problem, x, converged, reason, iterations,
                             diagnostics; step)
    owned = copy(x)
    residual = norm(problem.A * owned - problem.b)
    mapping = constrained_optimality_residual(problem, owned; step)
    T = promote_type(typeof(residual), typeof(problem.radius))
    return ConstrainedResult(owned, T(problem.radius), T(0.5 * residual^2),
                             T(residual), T(mapping), converged, reason,
                             iterations, diagnostics)
end

"""Reusable FISTA workspace for an L1-ball constrained least-squares problem."""
struct ConstrainedFISTASolver{P,S,T}
    problem::P
    base::S
    gradient_scale::T   # max(‖A'b‖₂, 1), fixed for the life of the solver
end

function ConstrainedFISTASolver(problem::ConstrainedLassoProblem;
                                step=:auto, stop=DualGapStop(), x0=nothing,
                                power_iterations=50, power_safety=1.2)
    problem.A isa StridedMatrix && problem.b isa StridedVector ||
        throw(ArgumentError("constrained FISTA currently requires CPU dense arrays"))
    isnothing(x0) || _constrained_feasible(problem, x0) ||
        throw(ArgumentError("x0 must satisfy the L1-ball constraint"))
    base = FISTASolver(LassoProblem(problem.A, problem.b;
                                    constraint=problem.constraint);
                       step, stop, x0, power_iterations, power_safety)
    scale = max(norm(adjoint(problem.A) * problem.b), 1)
    return ConstrainedFISTASolver(problem, base, scale)
end

# The constrained problem has no penalty; `nothing` stands in for lambda.
solve!(solver::ConstrainedFISTASolver; warm_start=true) =
    _fista_solve!(solver, nothing; warm_start)

_fista_workspace(solver::ConstrainedFISTASolver) = solver.base

function _fista_gradient!(solver::ConstrainedFISTASolver, y)
    p, base = solver.problem, solver.base
    mul!(base.residual, p.A, y)
    @. base.residual = base.residual - p.b
    mul!(base.gradient, adjoint(p.A), base.residual)
    return nothing
end

_fista_prox!(solver::ConstrainedFISTASolver, step, lambda) =
    _project_l1_ball!(solver.base.x, solver.base.prox_input,
                      solver.problem.radius,
                      solver.problem.constraint == :nonnegative)

_fista_kkt_reason(solver::ConstrainedFISTASolver) = :optimality

_fista_kkt_passed(solver::ConstrainedFISTASolver, lambda, stop::KKTStop) =
    constrained_optimality_residual(solver.problem, solver.base.x;
                                    step=solver.base.step) <=
    stop.atol + stop.rtol * solver.gradient_scale

_fista_gap_degenerate(solver::ConstrainedFISTASolver, lambda) = false

_fista_gap(solver::ConstrainedFISTASolver, lambda) =
    constrained_dual_gap(solver.problem, solver.base.x)

_fista_gap_tolerance(solver::ConstrainedFISTASolver, stop) =
    _dual_gap_tolerance(solver.problem.b, stop)

# The constrained mapping test uses the 2-norm and a fixed scale.
function _fista_check(solver::ConstrainedFISTASolver, lambda,
                      stop::GradientMappingStop, step)
    base = solver.base
    @. base.prox_input = (base.y - base.x) / step
    mapping = norm(base.prox_input)
    mapping <= stop.atol + stop.rtol * solver.gradient_scale || return nothing
    return (reason=:gradient_mapping,
            diagnostics=(gradient_mapping_residual=mapping,
                         gradient_mapping_scale=solver.gradient_scale))
end

function _fista_result(solver::ConstrainedFISTASolver, lambda, converged,
                       reason, iterations, diagnostics)
    base = solver.base
    result = _constrained_result(solver.problem, base.x, converged, reason,
                                 iterations, diagnostics; step=base.step)
    if base.stop isa GradientMappingStop && reason != :optimality
        _warn_fista_optimality(result.optimality_residual / solver.gradient_scale,
                               base.stop, :projected_gradient_over_scale)
    end
    return result
end

"""Reusable CPU ADMM workspace for an L1-ball constrained least-squares problem."""
struct ConstrainedADMMSolver{P,S,T}
    problem::P
    base::S
    mapping_step::T
end

function ConstrainedADMMSolver(problem::ConstrainedLassoProblem;
                               rho=1.0, stop=DualGapStop(), x0=nothing,
                               feasibility_atol=1e-8,
                               feasibility_rtol=1e-4)
    problem.A isa StridedMatrix && problem.b isa StridedVector ||
        throw(ArgumentError("constrained ADMM currently requires CPU dense arrays"))
    isnothing(x0) || _constrained_feasible(problem, x0) ||
        throw(ArgumentError("x0 must satisfy the L1-ball constraint"))
    base = ADMMSolver(LassoProblem(problem.A, problem.b;
                                   constraint=problem.constraint);
                      rho, stop, x0, feasibility_atol, feasibility_rtol)
    t = inv(max(opnorm(problem.A)^2, 1))
    return ConstrainedADMMSolver(problem, base, t)
end

function solve!(solver::ConstrainedADMMSolver; warm_start=true)
    p, base = solver.problem, solver.base
    if !warm_start
        fill!(base.z, zero(eltype(base.z)))
        fill!(base.u, zero(eltype(base.u)))
    end
    tolerance = base.stop isa DualGapStop ?
                _dual_gap_tolerance(p.b, base.stop) :
                _constrained_tolerance(p, base.stop)
    if base.stop isa DualGapStop
        gap = constrained_dual_gap(p, base.z)
        if gap <= tolerance
            return _constrained_result(p, base.z, true, :dual_gap, 0,
                                       (primal_residual=zero(base.rho),
                                        dual_residual=zero(base.rho),
                                        rho=base.rho, dual_gap=gap);
                                       step=solver.mapping_step)
        end
    else
        mapping = constrained_optimality_residual(p, base.z;
                                                   step=solver.mapping_step)
        if mapping <= tolerance
            return _constrained_result(p, base.z, true, :optimality, 0,
                                       (primal_residual=zero(base.rho),
                                        dual_residual=zero(base.rho),
                                        rho=base.rho); step=solver.mapping_step)
        end
    end
    primal = oftype(base.rho, Inf)
    dual = oftype(base.rho, Inf)
    for iteration in 1:base.stop.maxiter
        _admm_x_step!(base)
        copyto!(base.z_previous, base.z)
        @. base.q = base.x + base.u
        _project_l1_ball!(base.z, base.q, p.radius,
                          p.constraint == :nonnegative)
        @. base.u = base.u + base.x - base.z
        if iteration % base.stop.check_every == 0 ||
           iteration == base.stop.maxiter
            primal = norm(base.x - base.z)
            dual = base.rho * norm(base.z - base.z_previous)
            primal_tol = sqrt(length(base.z)) * base.feasibility_atol +
                         base.feasibility_rtol * max(norm(base.x), norm(base.z))
            dual_tol = sqrt(length(base.z)) * base.feasibility_atol +
                       base.feasibility_rtol * base.rho * norm(base.u)
            if primal <= primal_tol && dual <= dual_tol
                if base.stop isa DualGapStop
                    gap = constrained_dual_gap(p, base.z)
                    if gap <= tolerance
                        return _constrained_result(p, base.z, true,
                            :feasibility_and_dual_gap, iteration,
                            (primal_residual=primal, dual_residual=dual,
                             rho=base.rho, dual_gap=gap);
                            step=solver.mapping_step)
                    end
                else
                    mapping = constrained_optimality_residual(p, base.z;
                                                               step=solver.mapping_step)
                    if mapping <= tolerance
                        return _constrained_result(p, base.z, true,
                            :feasibility_and_optimality, iteration,
                            (primal_residual=primal, dual_residual=dual,
                             rho=base.rho); step=solver.mapping_step)
                    end
                end
            end
        end
    end
    return _constrained_result(p, base.z, false, :maxiter,
                               base.stop.maxiter,
                               (primal_residual=primal, dual_residual=dual,
                                rho=base.rho); step=solver.mapping_step)
end

initialize(problem::ConstrainedLassoProblem, method::FISTA; x0=nothing) =
    ConstrainedFISTASolver(problem; step=method.step, stop=method.stop, x0,
                            power_iterations=method.power_iterations,
                            power_safety=method.power_safety)

initialize(problem::ConstrainedLassoProblem, method::ADMM; x0=nothing) =
    ConstrainedADMMSolver(problem; rho=method.rho, stop=method.stop, x0,
                           feasibility_atol=method.feasibility_atol,
                           feasibility_rtol=method.feasibility_rtol)

function solve(problem::ConstrainedLassoProblem, method::Union{FISTA,ADMM};
               x0=nothing)
    return solve!(initialize(problem, method; x0))
end

export ConstrainedLassoProblem, ConstrainedResult
export ConstrainedFISTASolver, ConstrainedADMMSolver
export constrained_optimality_residual, constrained_dual_gap
