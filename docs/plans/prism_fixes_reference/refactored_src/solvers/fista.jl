"""
    FISTASolver(problem; step=:auto, stop=DualGapStop(), x0=nothing)

Reusable FISTA workspace. `:auto` uses an exact spectral norm only for CPU
dense matrices with at most 250,000 elements; larger/device matrices use a
power estimate. `:power` is fast but not a certified upper bound. The optional
`:frobenius` step is conservative because the spectral norm is no greater than
the Frobenius norm. The same matrix-vector implementation works on compatible
device arrays without a hard CUDA dependency.
"""
mutable struct FISTASolver{P,V,R,T,S}
    problem::P
    x::V
    y::V
    previous::V
    gradient::V
    prox_input::V
    residual::R
    step::T
    stop::S
end

function _power_step(A, b; iterations=50, safety=1.2)
    iterations isa Integer && iterations > 0 ||
        throw(ArgumentError("power iterations must be positive"))
    safety > 1 || throw(ArgumentError("power safety must exceed 1"))
    T = eltype(b)
    v = similar(b, T, size(A, 2))
    fill!(v, inv(sqrt(T(length(v)))))
    Av = similar(b, T, size(A, 1))
    AtAv = similar(b, T, size(A, 2))
    for _ in 1:iterations
        mul!(Av, A, v)
        mul!(AtAv, adjoint(A), Av)
        magnitude = norm(AtAv)
        if iszero(magnitude)
            iszero(norm(A)) && return one(T)
            throw(ArgumentError("power start lies in nullspace; pass an explicit step"))
        end
        @. v = AtAv / magnitude
    end
    mul!(Av, A, v)
    return inv(T(safety) * norm(Av)^2)
end

function FISTASolver(problem::LassoProblem; step=:auto, stop=DualGapStop(),
                     x0=nothing, power_iterations=50, power_safety=1.2)
    stop isa Union{KKTStop,GradientMappingStop,DualGapStop} ||
        throw(ArgumentError("FISTA stop must be KKTStop, GradientMappingStop, or DualGapStop"))
    T = eltype(problem.b)
    x = similar(problem.b, T, size(problem.A, 2))
    if isnothing(x0)
        fill!(x, zero(T))
    else
        length(x0) == length(x) || throw(DimensionMismatch("x0 has wrong length"))
        copyto!(x, x0)
    end
    if problem.constraint == :nonnegative && minimum(x) < 0
        throw(ArgumentError("x0 must be nonnegative"))
    end
    chosen = step === :auto ?
             (problem.A isa StridedMatrix && length(problem.A) <= 250_000 ?
              :exact : :power) : step
    t = if chosen === :exact
        lipschitz = opnorm(problem.A)^2
        iszero(lipschitz) ? one(T) : inv(T(lipschitz))
    elseif chosen === :power
        _power_step(problem.A, problem.b;
                    iterations=power_iterations, safety=power_safety)
    elseif chosen === :frobenius
        lipschitz = sum(abs2, problem.A)
        iszero(lipschitz) ? one(T) : inv(T(lipschitz))
    elseif chosen isa Real && isfinite(chosen) && chosen > 0
        T(chosen)
    else
        throw(ArgumentError("step must be :auto, :exact, :power, :frobenius, or positive"))
    end
    return FISTASolver(problem, x, copy(x), similar(x), similar(x), similar(x),
                       similar(problem.b), t, stop)
end

"""
    solve!(solver::FISTASolver, lambda; warm_start=true)

Mutate the reusable solver, but return an **owned** result snapshot. By default
the current coefficients initialize the solve; `warm_start=false` resets them
to zero. With `KKTStop`, convergence means KKT at the returned vector passes.
With the default `DualGapStop`, a small iterate update triggers an exact
returned-point primal-dual gap test. For `lambda == 0`, use a scaled KKT
fallback. `GradientMappingStop` remains available explicitly.
"""
function solve!(solver::FISTASolver, lambda; warm_start=true)
    _validate_lambda(lambda)
    return _fista_solve!(solver, lambda; warm_start)
end

_fista_workspace(solver::FISTASolver) = solver

function _fista_gradient!(solver::FISTASolver, y)
    problem = solver.problem
    mul!(solver.residual, problem.A, y)
    @. solver.residual = solver.residual - problem.b
    _loss_derivative!(problem.loss, solver.residual)
    mul!(solver.gradient, adjoint(problem.A), solver.residual)
    return nothing
end

_fista_prox!(solver::FISTASolver, step, lambda) =
    _penalty_prox!(solver.problem.penalty, solver.x, solver.prox_input,
                   step * lambda, solver.problem.constraint)

# Reuses the workspace; the gradient at y is no longer needed at this point.
_fista_kkt(solver::FISTASolver, lambda) =
    _lasso_kkt_residual!(solver.problem, solver.x, lambda, solver.residual,
                         solver.gradient, solver.prox_input)

_fista_kkt_scale(solver::FISTASolver) =
    max(norm(_loss_correlations(solver.problem.A, solver.problem.b,
                                solver.problem.loss), Inf), 1)

_fista_gap(solver::FISTASolver, lambda) =
    lasso_dual_gap(solver.problem, solver.x, lambda)

_fista_gap_tolerance(solver::FISTASolver, stop) =
    _dual_gap_tolerance(solver.problem.b, solver.problem.loss, stop)

function _fista_mapping_scale(solver::FISTASolver, lambda, step)
    if solver.problem.constraint == :nonnegative && !iszero(lambda)
        # Positive gradients at zero coefficients point out of the
        # feasible descent cone and must not loosen the scale.
        @. solver.prox_input = ifelse(solver.x > 0, abs(solver.gradient),
                                      max(-solver.gradient, 0))
        return max(lambda, maximum(solver.prox_input), eps(float(step)))
    end
    return max(norm(solver.gradient, Inf), eps(float(step)))
end

_fista_result(solver::FISTASolver, lambda, converged, reason, iterations,
              diagnostics) =
    _make_result(solver.problem, solver.x, lambda, converged, reason,
                 iterations, diagnostics)
