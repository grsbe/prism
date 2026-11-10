# Familiar SRT entry points live here, apart from the new problem/solver API.
# Signatures and NamedTuple fields are retained for the common penalized LASSO
# calls. FISTA wrappers default to dual-gap stopping and report final
# returned-point KKT; explicit legacy tolerance keywords still request the
# old gradient-mapping rule. ADMM wrappers default to dual gap; explicit legacy
# tolerances request KKT stopping. See README.

function _compat_admm_stop(; abstol=nothing, reltol=nothing,
                      optimality_abstol=nothing, optimality_reltol=nothing,
                      maxiter=10_000, reference=DualGapStop())
    has_legacy_tolerance = !all(isnothing, (abstol, reltol,
                                            optimality_abstol,
                                            optimality_reltol))
    if !has_legacy_tolerance
        return reference isa DualGapStop ?
               DualGapStop(; tol=reference.tol, maxiter,
                             check_every=reference.check_every) :
               KKTStop(; atol=reference.atol, rtol=reference.rtol,
                         maxiter, check_every=reference.check_every)
    end
    absolute = something(optimality_abstol, abstol, 1e-10)
    relative = something(optimality_reltol, reltol, 1e-3)
    return KKTStop(; atol=absolute, rtol=relative, maxiter,
                    check_every=1)
end

function _compat_fista_stop(; abstol=nothing, reltol=nothing,
                             optimality_abstol=nothing,
                             optimality_reltol=nothing, maxiter=10_000,
                             reference=DualGapStop())
    has_legacy_tolerance = !all(isnothing, (abstol, reltol,
                                            optimality_abstol,
                                            optimality_reltol))
    if reference isa DualGapStop && !has_legacy_tolerance
        return DualGapStop(; tol=reference.tol, maxiter,
                            check_every=reference.check_every)
    end
    absolute = something(optimality_abstol, abstol,
                         reference isa DualGapStop ? 0 : reference.atol)
    relative = something(optimality_reltol, reltol,
                         reference isa DualGapStop ? 1e-3 : reference.rtol)
    if reference isa KKTStop
        return KKTStop(; atol=absolute, rtol=relative, maxiter,
                        check_every=reference.check_every)
    end
    return GradientMappingStop(; atol=absolute, rtol=relative, maxiter,
                                check_every=reference.check_every,
                                warn_optimality_ratio=reference isa GradientMappingStop ?
                                                      reference.warn_optimality_ratio : 0.1)
end

function _compat_fista_result(result, return_info)
    if !return_info
        return result.x
    end
    return (x=result.x, converged=result.converged,
            iterations=result.iterations,
            iterate_change=get(result.diagnostics, :iterate_change, NaN),
            optimality_residual=get(result.diagnostics, :gradient_mapping_residual,
                                    result.kkt_residual),
            kkt_residual=result.kkt_residual,
            kkt_ratio=result.kkt_ratio,
            step=result.diagnostics.step,
            stop_reason=result.stop_reason)
end

function _compat_admm_result(result, return_info)
    if !return_info
        return result.x
    end
    return (x=result.x, converged=result.converged,
            iterations=result.iterations,
            primal_residual=result.diagnostics.primal_residual,
            dual_residual=result.diagnostics.dual_residual,
            kkt_residual=result.kkt_residual,
            stop_reason=result.stop_reason)
end

function LassoFISTASolver(A, b; step=nothing, x0=nothing,
                          stop=DualGapStop())
    return FISTASolver(LassoProblem(A, b); step=isnothing(step) ? :auto : step,
                       x0, stop)
end

function LassoADMMSolver(A, b; rho=1.0, x0=nothing,
                         stop=DualGapStop())
    return ADMMSolver(LassoProblem(A, b); rho, x0, stop)
end

function _compat_fista(A, b, lambda, constraint;
                       step=nothing, x0=nothing, warm_start=true,
                       abstol=nothing, reltol=nothing,
                       optimality_abstol=nothing, optimality_reltol=nothing,
                       maxiter=10_000, return_info=false)
    stop = _compat_fista_stop(; abstol, reltol, optimality_abstol,
                               optimality_reltol, maxiter)
    solver = FISTASolver(LassoProblem(A, b; constraint);
                          step=isnothing(step) ? :auto : step, x0, stop)
    return _compat_fista_result(solve!(solver, lambda; warm_start), return_info)
end

function lasso_fista(A::AbstractMatrix, b::AbstractVector, lambda; kwargs...)
    return _compat_fista(A, b, lambda, :signed; kwargs...)
end

function nonnegative_lasso_fista(A::AbstractMatrix, b::AbstractVector, lambda;
                                 kwargs...)
    return _compat_fista(A, b, lambda, :nonnegative; kwargs...)
end

function _compat_set_constraint!(solver::FISTASolver, constraint, warm_start)
    if solver.problem.constraint != constraint
        solver.problem = LassoProblem(solver.problem.A, solver.problem.b;
                                      constraint, loss=solver.problem.loss,
                                      penalty=solver.problem.penalty)
        if warm_start && constraint == :nonnegative
            @. solver.x = max(solver.x, zero(eltype(solver.x)))
        end
    end
end

function _compat_set_constraint!(solver::ADMMSolver, constraint, warm_start)
    if solver.problem.constraint != constraint
        solver.problem = LassoProblem(solver.problem.A, solver.problem.b;
                                      constraint, loss=solver.problem.loss,
                                      penalty=solver.problem.penalty)
        if warm_start && constraint == :nonnegative
            @. solver.z = max(solver.z, zero(eltype(solver.z)))
        end
    end
end

function _compat_fista_solver(solver::FISTASolver, lambda, constraint;
                              warm_start=false, return_info=false,
                              abstol=nothing, reltol=nothing,
                              optimality_abstol=nothing,
                              optimality_reltol=nothing, maxiter=nothing)
    _compat_set_constraint!(solver, constraint, warm_start)
    if !all(isnothing, (abstol, reltol, optimality_abstol,
                        optimality_reltol, maxiter))
        chosen = _compat_fista_stop(; abstol, reltol, optimality_abstol,
                                     optimality_reltol,
                                     maxiter=something(maxiter, solver.stop.maxiter),
                                     reference=solver.stop)
        if typeof(chosen) != typeof(solver.stop)
            # The workspace's stop field is concretely typed. An explicit
            # legacy tolerance switches to a temporary mapping-rule solver;
            # copy its solution back so later warm starts still work.
            temporary = FISTASolver(solver.problem; step=solver.step,
                                     stop=chosen, x0=solver.x)
            result = solve!(temporary, lambda; warm_start)
            copyto!(solver.x, result.x)
            return _compat_fista_result(result, return_info)
        end
        solver.stop = chosen
    end
    return _compat_fista_result(solve!(solver, lambda; warm_start), return_info)
end

lasso_fista(solver::FISTASolver, lambda; kwargs...) =
    _compat_fista_solver(solver, lambda, :signed; kwargs...)
nonnegative_lasso_fista(solver::FISTASolver, lambda; kwargs...) =
    _compat_fista_solver(solver, lambda, :nonnegative; kwargs...)

function _compat_admm(A, b, lambda, constraint;
                      rho=1.0, x0=nothing, warm_start=true,
                      abstol=nothing, reltol=nothing,
                      maxiter=10_000, return_info=false)
    stop = _compat_admm_stop(; abstol, reltol, maxiter)
    solver = ADMMSolver(LassoProblem(A, b; constraint); rho, x0, stop)
    return _compat_admm_result(solve!(solver, lambda; warm_start), return_info)
end

lasso_admm(A::AbstractMatrix, b::AbstractVector, lambda; kwargs...) =
    _compat_admm(A, b, lambda, :signed; kwargs...)
nonnegative_lasso_admm(A::AbstractMatrix, b::AbstractVector, lambda;
                       kwargs...) =
    _compat_admm(A, b, lambda, :nonnegative; kwargs...)

function _compat_admm_solver(solver::ADMMSolver, lambda, constraint;
                             warm_start=false, return_info=false,
                             abstol=nothing, reltol=nothing, maxiter=nothing)
    _compat_set_constraint!(solver, constraint, warm_start)
    if !all(isnothing, (abstol, reltol, maxiter))
        solver.stop = _compat_admm_stop(; abstol, reltol,
                                        maxiter=something(maxiter, solver.stop.maxiter),
                                        reference=solver.stop)
    end
    return _compat_admm_result(solve!(solver, lambda; warm_start), return_info)
end

lasso_admm(solver::ADMMSolver, lambda; kwargs...) =
    _compat_admm_solver(solver, lambda, :signed; kwargs...)
nonnegative_lasso_admm(solver::ADMMSolver, lambda; kwargs...) =
    _compat_admm_solver(solver, lambda, :nonnegative; kwargs...)
