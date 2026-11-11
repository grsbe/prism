"""
    ADMMSolver(problem; rho=1, stop=DualGapStop(), feasibility_atol=1e-8,
               feasibility_rtol=1e-4)

CPU ADMM for LASSO. A tall problem factors `A'A + rho*I`; a wide problem
factors the smaller `AA' + rho*I` and uses Woodbury. Rho is fixed for the
life of the solver because its factorization is cached. A result converges
only when primal and dual residuals pass their tests, and a returned-point
dual gap passes (or KKT for the zero-penalty fallback).
"""
mutable struct ADMMSolver{P,V,F,T}
    problem::P
    x::V
    z::V
    u::V
    z_previous::V
    q::V
    Atb::V
    factor::F
    wide::Bool
    rho::T
    stop::Union{DualGapStop,KKTStop}
    feasibility_atol::T
    feasibility_rtol::T
end

function ADMMSolver(problem::LassoProblem; rho=1.0, stop=DualGapStop(),
                    feasibility_atol=1e-8, feasibility_rtol=1e-4,
                    x0=nothing)
    stop isa Union{DualGapStop,KKTStop} ||
        throw(ArgumentError("ADMM stop must be DualGapStop or KKTStop"))
    problem.A isa StridedMatrix ||
        throw(ArgumentError("ADMM currently requires a CPU dense matrix; use FISTA for device arrays"))
    problem.loss isa LeastSquares ||
        throw(ArgumentError("ADMM supports only the least-squares loss; use FISTA for Huber"))
    problem.penalty isa L1 ||
        throw(ArgumentError("ADMM supports only the L1 penalty; use FISTA for Huber"))
    rho isa Real && isfinite(rho) && rho > 0 ||
        throw(ArgumentError("rho must be a finite positive real"))
    _validate_stop_number(feasibility_atol, "feasibility_atol")
    _validate_stop_number(feasibility_rtol, "feasibility_rtol")
    A, b = problem.A, problem.b
    T = eltype(b)
    n, m = size(A, 2), size(A, 1)
    z = zeros(T, n)
    if !isnothing(x0)
        length(x0) == n || throw(DimensionMismatch("x0 has wrong length"))
        copyto!(z, x0)
    end
    problem.constraint == :nonnegative && minimum(z) < 0 &&
        throw(ArgumentError("x0 must be nonnegative"))
    ρ = T(rho)
    wide = n > m
    gram = wide ? A * adjoint(A) : adjoint(A) * A
    factor = cholesky(Hermitian(gram + ρ * I))
    return ADMMSolver(problem, similar(z), z, zeros(T, n), similar(z),
                      similar(z), adjoint(A) * b, factor, wide, ρ, stop,
                      T(feasibility_atol), T(feasibility_rtol))
end

function _admm_x_step!(solver::ADMMSolver)
    A, ρ = solver.problem.A, solver.rho
    @. solver.q = solver.Atb + ρ * (solver.z - solver.u)
    if solver.wide
        temporary = solver.factor \ (A * solver.q)
        mul!(solver.x, adjoint(A), temporary)
        @. solver.x = (solver.q - solver.x) / ρ
    else
        copyto!(solver.x, solver.factor \ solver.q)
    end
    return nothing
end

function _admm_diagnostics(solver, primal, dual)
    return (primal_residual=primal, dual_residual=dual, rho=solver.rho,
            feasibility_atol=solver.feasibility_atol,
            feasibility_rtol=solver.feasibility_rtol)
end

function _admm_optimality(problem, z, lambda, stop)
    if stop isa DualGapStop && !iszero(lambda)
        gap = lasso_dual_gap(problem, z, lambda)
        return gap <= _dual_gap_tolerance(problem.b, stop), :dual_gap,
               (dual_gap=gap,)
    end
    kkt = lasso_kkt_residual(problem, z, lambda)
    tolerance = stop isa DualGapStop ?
                stop.tol * norm(adjoint(problem.A) * problem.b, Inf) :
                _kkt_tolerance(stop, lambda)
    reason = stop isa DualGapStop ? :kkt_zero_penalty : :kkt
    return kkt <= tolerance, reason, (kkt_residual=kkt,)
end

"""Solve with cached factorization; the returned coefficient vector is owned."""
function solve!(solver::ADMMSolver, lambda; warm_start=true)
    _validate_lambda(lambda)
    if !warm_start
        fill!(solver.z, zero(eltype(solver.z)))
        fill!(solver.u, zero(eltype(solver.u)))
    end
    problem, stop, ρ = solver.problem, solver.stop, solver.rho
    passed, reason, certificate = _admm_optimality(problem, solver.z,
                                                    lambda, stop)
    if passed
        diagnostics = merge(_admm_diagnostics(solver, zero(ρ), zero(ρ)),
                            certificate)
        return _make_result(problem, solver.z, lambda, true, reason, 0,
                            diagnostics)
    end
    primal = oftype(ρ, Inf)
    dual = oftype(ρ, Inf)
    for iteration in 1:stop.maxiter
        _admm_x_step!(solver)
        copyto!(solver.z_previous, solver.z)
        threshold = lambda / ρ
        if problem.constraint == :nonnegative
            @. solver.z = max(solver.x + solver.u - threshold, 0)
        else
            @. solver.z = sign(solver.x + solver.u) *
                           max(abs(solver.x + solver.u) - threshold, 0)
        end
        @. solver.u = solver.u + solver.x - solver.z

        if iteration % stop.check_every == 0 || iteration == stop.maxiter
            primal = norm(solver.x - solver.z)
            dual = ρ * norm(solver.z - solver.z_previous)
            primal_tol = sqrt(length(solver.z)) * solver.feasibility_atol +
                         solver.feasibility_rtol * max(norm(solver.x), norm(solver.z))
            dual_tol = sqrt(length(solver.z)) * solver.feasibility_atol +
                       solver.feasibility_rtol * ρ * norm(solver.u)
            if primal <= primal_tol && dual <= dual_tol
                passed, reason, certificate = _admm_optimality(problem,
                    solver.z, lambda, stop)
                if passed
                    label = reason == :dual_gap ? :feasibility_and_dual_gap :
                            reason == :kkt_zero_penalty ?
                            :feasibility_and_kkt_zero_penalty :
                            :feasibility_and_kkt
                    return _make_result(problem, solver.z, lambda, true,
                                        label, iteration,
                                        merge(_admm_diagnostics(solver, primal, dual),
                                              certificate))
                end
            end
        end
    end
    return _make_result(problem, solver.z, lambda, false, :maxiter,
                        stop.maxiter, _admm_diagnostics(solver, primal, dual))
end
