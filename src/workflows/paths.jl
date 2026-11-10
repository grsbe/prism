"""
    solve_path!(solver, lambdas; on_result=nothing, store=true,
                stop_on_failure=false, restart=false)

Solve in the supplied order, warm-starting from the solver's current state.
Set `restart=true` to reset to zero before the first point. Results are
owned snapshots. `on_result(index, result)` runs immediately after each point,
so callers can persist a path incrementally. `store=false` avoids retaining
the entire coefficient path in memory and returns `nothing`.
"""
function solve_path!(solver::Union{FISTASolver,GroupFISTASolver,ADMMSolver}, lambdas;
                     on_result=nothing, store=true, stop_on_failure=false,
                     restart=false)
    values = collect(lambdas)
    isempty(values) && throw(ArgumentError("lambdas must be nonempty"))
    foreach(_validate_lambda, values)
    results = store ? SolveResult[] : nothing
    for (index, lambda) in enumerate(values)
        result = solve!(solver, lambda; warm_start=index > 1 || !restart)
        if store
            push!(results, result)
        end
        if !isnothing(on_result)
            on_result(index, result)
        end
        if stop_on_failure && !result.converged
            break
        end
    end
    return results
end

const DEFAULT_LASSO_PATH_MIN_RATIO = 1e-4
const DEFAULT_LASSO_PATH_POINTS = 30

"""
    lambda_grid(problem::LassoProblem;
                min_ratio=1e-4, n_lambdas=30)

Return a descending, logarithmically spaced LASSO grid from the smallest
penalty with zero solution, `lambda_max`, through
`min_ratio * lambda_max`. For signed LASSO,
`lambda_max = maximum(abs, A' * ψ(b))`; for nonnegative LASSO,
`lambda_max = max(0, maximum(A' * ψ(b)))`, where `ψ` is the loss
derivative (`ψ(b) = b` for least squares).
If `lambda_max == 0`, return the single value `[0]`. The Huber penalty has
no zero-solution penalty, so it throws; pass explicit lambdas instead.
This grid does not select a scientifically appropriate penalty.
"""
function lambda_grid(problem::LassoProblem;
                     min_ratio=DEFAULT_LASSO_PATH_MIN_RATIO,
                     n_lambdas=DEFAULT_LASSO_PATH_POINTS)
    min_ratio isa Real && isfinite(min_ratio) && 0 < min_ratio < 1 ||
        throw(ArgumentError("min_ratio must be finite and between 0 and 1"))
    n_lambdas isa Integer && n_lambdas >= 2 ||
        throw(ArgumentError("n_lambdas must be an integer of at least 2"))
    problem.penalty isa L1 ||
        throw(ArgumentError("lambda_grid needs the L1 penalty; the Huber penalty has no zero-solution lambda_max, so pass lambdas explicitly"))
    correlations = _loss_correlations(problem.A, problem.b, problem.loss)
    lambda_max = problem.constraint == :signed ?
                 maximum(abs, correlations) :
                 max(zero(eltype(correlations)), maximum(correlations))
    iszero(lambda_max) && return [lambda_max]
    T = typeof(lambda_max)
    values = lambda_max .* exp.(range(zero(T), log(T(min_ratio));
                                      length=n_lambdas))
    values[1] = lambda_max
    values[end] = lambda_max * T(min_ratio)
    return values
end

lambda_grid(solver::Union{FISTASolver,GroupFISTASolver,ADMMSolver}; kwargs...) =
    lambda_grid(solver.problem; kwargs...)

"""
    solve_path!(solver; min_ratio=1e-4, n_lambdas=30, kwargs...)

Solve an automatically generated LASSO path from the zero-solution penalty
through `min_ratio * lambda_max`. The entire grid is attempted unless
`stop_on_failure=true`; there is no fit-plateau early stop. To continue a
particular grid or use another endpoint, pass `lambdas` explicitly.
"""
function solve_path!(solver::Union{FISTASolver,GroupFISTASolver,ADMMSolver};
                     min_ratio=DEFAULT_LASSO_PATH_MIN_RATIO,
                     n_lambdas=DEFAULT_LASSO_PATH_POINTS, kwargs...)
    values = lambda_grid(solver; min_ratio, n_lambdas)
    return solve_path!(solver, values; kwargs...)
end
