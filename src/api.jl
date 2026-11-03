"""Configuration for reusable FISTA. The initialized solver owns its workspace."""
struct FISTA{S,T,U}
    step::S
    stop::U
    power_iterations::Int
    power_safety::T
end

function FISTA(; step=:auto, stop=DualGapStop(), power_iterations=50,
                power_safety=1.2)
    stop isa Union{KKTStop,GradientMappingStop,DualGapStop} ||
        throw(ArgumentError("FISTA stop must be KKTStop, GradientMappingStop, or DualGapStop"))
    return FISTA(step, stop, power_iterations, power_safety)
end

"""Configuration for reusable CPU ADMM with a fixed factorization parameter."""
struct ADMM{T}
    rho::T
    stop::Union{DualGapStop,KKTStop}
    feasibility_atol::T
    feasibility_rtol::T
end

function ADMM(; rho=1.0, stop=DualGapStop(), feasibility_atol=1e-8,
              feasibility_rtol=1e-4)
    stop isa Union{DualGapStop,KKTStop} ||
        throw(ArgumentError("ADMM stop must be DualGapStop or KKTStop"))
    T = float(promote_type(typeof(rho), typeof(feasibility_atol),
                           typeof(feasibility_rtol)))
    return ADMM(T(rho), stop, T(feasibility_atol), T(feasibility_rtol))
end

"""Optional Gurobi QP backend; requires Gurobi.jl and JuMP.jl."""
struct GurobiBackend
    verbose::Bool
    time_limit::Union{Nothing,Float64}
end

function GurobiBackend(; verbose=false, time_limit=nothing)
    if !isnothing(time_limit)
        time_limit isa Real && isfinite(time_limit) && time_limit > 0 ||
            throw(ArgumentError("time_limit must be positive and finite"))
    end
    return GurobiBackend(verbose, isnothing(time_limit) ? nothing :
                         Float64(time_limit))
end

initialize(problem::LassoProblem, method::FISTA; x0=nothing) =
    FISTASolver(problem; step=method.step, stop=method.stop, x0,
                power_iterations=method.power_iterations,
                power_safety=method.power_safety)

initialize(problem::LassoProblem, method::ADMM; x0=nothing) =
    ADMMSolver(problem; rho=method.rho, stop=method.stop, x0,
               feasibility_atol=method.feasibility_atol,
               feasibility_rtol=method.feasibility_rtol)

function solve(problem::LassoProblem, method::Union{FISTA,ADMM}, lambda;
               x0=nothing)
    solver = initialize(problem, method; x0)
    return solve!(solver, lambda)
end

solve(problem::LassoProblem, method::Union{FISTA,ADMM}; lambda, x0=nothing) =
    solve(problem, method, lambda; x0)

function solve_path(problem::LassoProblem, method::Union{FISTA,ADMM}, lambdas;
                    x0=nothing, kwargs...)
    solver = initialize(problem, method; x0)
    return solve_path!(solver, lambdas; kwargs...)
end

"""
    solve_path(problem::LassoProblem, method; min_ratio=1e-4,
               n_lambdas=30, x0=nothing, kwargs...)

Initialize a solver and compute the default LASSO path. Use
`lambda_grid(problem; ...)` to inspect the penalties, or pass an explicit
lambda sequence to the three-argument `solve_path` method.
"""
function solve_path(problem::LassoProblem, method::Union{FISTA,ADMM};
                    min_ratio=DEFAULT_LASSO_PATH_MIN_RATIO,
                    n_lambdas=DEFAULT_LASSO_PATH_POINTS,
                    x0=nothing, kwargs...)
    solver = initialize(problem, method; x0)
    return solve_path!(solver; min_ratio, n_lambdas, kwargs...)
end

export FISTA, ADMM, GurobiBackend, initialize, solve, solve_path
initialize(problem::GroupLassoProblem, method::FISTA; x0=nothing) =
    GroupFISTASolver(problem; step=method.step, stop=method.stop, x0,
                     power_iterations=method.power_iterations,
                     power_safety=method.power_safety)

solve(problem::GroupLassoProblem, method::FISTA, lambda; x0=nothing) =
    solve!(initialize(problem, method; x0), lambda)

solve(problem::GroupLassoProblem, method::FISTA; lambda, x0=nothing) =
    solve(problem, method, lambda; x0)

function solve_path(problem::GroupLassoProblem, method::FISTA, lambdas;
                    x0=nothing, kwargs...)
    return solve_path!(initialize(problem, method; x0), lambdas; kwargs...)
end

function solve_path(problem::GroupLassoProblem, method::FISTA;
                    min_ratio=DEFAULT_LASSO_PATH_MIN_RATIO,
                    n_lambdas=DEFAULT_LASSO_PATH_POINTS,
                    x0=nothing, kwargs...)
    return solve_path!(initialize(problem, method; x0);
                       min_ratio, n_lambdas, kwargs...)
end

export lambda_grid
