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
    problem, stop = solver.problem, solver.stop
    x, y, previous = solver.x, solver.y, solver.previous
    if !warm_start
        fill!(x, zero(eltype(x)))
    end
    copyto!(y, x)
    if stop isa KKTStop
        kkt = _lasso_kkt_residual!(problem, x, lambda, solver.residual,
                                   solver.gradient, solver.prox_input)
        if kkt <= _kkt_tolerance(stop, lambda)
            return _make_result(problem, x, lambda, true, :kkt, 0,
                                (step=solver.step,))
        end
    end
    t = one(solver.step)
    for iteration in 1:stop.maxiter
        copyto!(previous, x)
        mul!(solver.residual, problem.A, y)
        @. solver.residual = solver.residual - problem.b
        _loss_derivative!(problem.loss, solver.residual)
        mul!(solver.gradient, adjoint(problem.A), solver.residual)
        @. solver.prox_input = y - solver.step * solver.gradient
        _penalty_prox!(problem.penalty, x, solver.prox_input,
                       solver.step * lambda, problem.constraint)
        if iteration % stop.check_every == 0 || iteration == stop.maxiter
            if stop isa KKTStop
                kkt = _lasso_kkt_residual!(problem, x, lambda,
                                           solver.residual, solver.gradient,
                                           solver.prox_input)
                if kkt <= _kkt_tolerance(stop, lambda)
                    return _make_result(problem, x, lambda, true, :kkt,
                                        iteration, (step=solver.step,))
                end
            elseif stop isa DualGapStop
                change = norm(x - previous, Inf)
                coefficient_scale = norm(x, Inf)
                if change <= stop.tol * coefficient_scale
                    if iszero(lambda)
                        kkt = _lasso_kkt_residual!(problem, x, lambda,
                                                   solver.residual, solver.gradient,
                                                   solver.prox_input)
                        scale = max(norm(_loss_correlations(problem.A, problem.b,
                                                            problem.loss), Inf), 1)
                        if kkt <= stop.tol * scale
                            return _make_result(problem, x, lambda, true,
                                                :kkt_zero_penalty, iteration,
                                                (step=solver.step,
                                                 iterate_change=change))
                        end
                    else
                        gap = lasso_dual_gap(problem, x, lambda)
                        if gap <= _dual_gap_tolerance(problem.b, problem.loss, stop)
                            return _make_result(problem, x, lambda, true,
                                                :dual_gap, iteration,
                                                (step=solver.step,
                                                 iterate_change=change,
                                                 dual_gap=gap))
                        end
                    end
                end
            else
                @. solver.prox_input = (y - x) / solver.step
                mapping = norm(solver.prox_input, Inf)
                scale = if problem.constraint == :nonnegative && !iszero(lambda)
                    # Positive gradients at zero coefficients point out of the
                    # feasible descent cone and must not loosen the scale.
                    @. solver.prox_input = ifelse(x > 0, abs(solver.gradient),
                                                  max(-solver.gradient, 0))
                    max(lambda, maximum(solver.prox_input),
                        eps(float(solver.step)))
                else
                    max(norm(solver.gradient, Inf), eps(float(solver.step)))
                end
                if mapping <= stop.atol + stop.rtol * scale
                    result = _make_result(problem, x, lambda, true,
                                          :gradient_mapping, iteration,
                                          (step=solver.step,
                                           gradient_mapping_residual=mapping,
                                           gradient_mapping_scale=scale))
                    return result
                end
            end
        end
        next_t = (one(t) + sqrt(one(t) + 4t^2)) / 2
        momentum = (t - one(t)) / next_t
        @. y = x + momentum * (x - previous)
        t = next_t
    end
    result = _make_result(problem, x, lambda, false, :maxiter,
                          stop.maxiter, (step=solver.step,))
    return result
end
