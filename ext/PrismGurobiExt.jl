module SpasmiGurobiExt

using Spasmi
using JuMP
using Gurobi

function Spasmi.solve(problem::Spasmi.LassoProblem,
                      method::Spasmi.GurobiBackend, lambda;
                      x0=nothing)
    Spasmi._validate_lambda(lambda)
    isnothing(x0) || throw(ArgumentError("GurobiBackend does not yet use x0"))
    A, b = problem.A, problem.b
    A isa StridedMatrix ||
        throw(ArgumentError("GurobiBackend currently expects a CPU dense matrix"))
    m, n = size(A)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", method.verbose ? 1 : 0)
    if !isnothing(method.time_limit)
        set_time_limit_sec(model, method.time_limit)
    end

    if problem.constraint == :nonnegative
        @variable(model, x[1:n] >= 0)
        @variable(model, residual[1:m])
        @constraint(model, [i=1:m],
                    residual[i] == sum(A[i, j] * x[j] for j=1:n) - b[i])
        @objective(model, Min,
                   0.5 * sum(residual[i]^2 for i=1:m) +
                   lambda * sum(x[j] for j=1:n))
    else
        @variable(model, x[1:n])
        @variable(model, magnitude[1:n] >= 0)
        @variable(model, residual[1:m])
        @constraint(model, [j=1:n], magnitude[j] >= x[j])
        @constraint(model, [j=1:n], magnitude[j] >= -x[j])
        @constraint(model, [i=1:m],
                    residual[i] == sum(A[i, j] * x[j] for j=1:n) - b[i])
        @objective(model, Min,
                   0.5 * sum(residual[i]^2 for i=1:m) +
                   lambda * sum(magnitude[j] for j=1:n))
    end

    optimize!(model)
    status = termination_status(model)
    has_values(model) || error("Gurobi returned no primal solution: $status")
    coefficients = value.(x)
    kkt = Spasmi.lasso_kkt_residual(problem, coefficients, lambda)
    converged = status == JuMP.MOI.OPTIMAL &&
                kkt <= Spasmi._kkt_tolerance(Spasmi.KKTStop(), lambda)
    return Spasmi._make_result(problem, coefficients, lambda, converged,
                              converged ? :optimal : :solver_not_optimal, 0,
                              (termination_status=string(status),))
end

end
