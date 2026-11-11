"""
    BlockLassoRidgeProblem(A, B, b; constraint=:signed,
                           x_constraint=constraint, z_constraint=constraint,
                           loss=LeastSquares(), penalty=L1())

The design for `min ℓ(Ax+Bz-b) + λR(x) + γ/2‖z‖²`, where the loss `ℓ`
defaults to ½‖r‖² and `loss=Huber(delta)` selects the Huber loss. The
sparse-block penalty `R` defaults to `L1()`; `penalty=Huber(delta)` selects
the (non-sparse) Huber penalty. With
`x_constraint` and `z_constraint` independently select `:signed` or
`:nonnegative`. The legacy `constraint` keyword sets both unless overridden.
"""
struct BlockLassoRidgeProblem{MA<:AbstractMatrix,MB<:AbstractMatrix,VB<:AbstractVector,L<:DataLoss,R<:CoefficientPenalty}
    A::MA
    B::MB
    b::VB
    x_constraint::Symbol
    z_constraint::Symbol
    loss::L
    penalty::R
end

function BlockLassoRidgeProblem(A::AbstractMatrix, B::AbstractMatrix,
                                b::AbstractVector; constraint=:signed,
                                x_constraint=constraint, z_constraint=constraint,
                                loss=LeastSquares(), penalty=L1())
    validated = LassoProblem(A, b; constraint=x_constraint, loss, penalty)
    z_constraint in (:signed, :nonnegative) ||
        throw(ArgumentError("z_constraint must be :signed or :nonnegative"))
    size(B, 1) == length(b) ||
        throw(DimensionMismatch("B rows must match length(b)"))
    size(B, 2) > 0 || throw(ArgumentError("B must have at least one column"))
    eltype(B) == eltype(b) ||
        throw(ArgumentError("A, B, and b must have the same float element type"))
    finite_B = B isa AbstractSparseMatrix ? all(isfinite, nonzeros(B)) :
               all(isfinite, B)
    finite_B || throw(ArgumentError("B must be finite"))
    return BlockLassoRidgeProblem(A, B, b, x_constraint, z_constraint,
                                  validated.loss, validated.penalty)
end

"""
    block_dual_gap(problem, x, z, lambda, ridge)

Primal-dual gap for a positive ridge penalty and positive L1 penalty.
The loss derivative `u = ψ(Ax+Bz-b)` is scaled to satisfy the sparse-block
dual constraint.
The ridge-block conjugate is quadratic; for nonnegative ridge coefficients
it uses only the negative part of `B'u`.
"""
function block_dual_gap(problem::BlockLassoRidgeProblem, x, z,
                        lambda, ridge)
    _validate_lambda(lambda)
    ridge > 0 || throw(ArgumentError("ridge must be positive for the dual gap"))
    lambda > 0 || throw(ArgumentError("lambda must be positive for the dual gap"))
    problem.x_constraint == :nonnegative && minimum(x) < 0 &&
        throw(ArgumentError("x must be nonnegative"))
    problem.z_constraint == :nonnegative && minimum(z) < 0 &&
        throw(ArgumentError("z must be nonnegative"))
    residual = problem.A * x + problem.B * z - problem.b
    primal = _loss_value(problem.loss, residual) +
             lambda * _penalty_value(problem.penalty, x) +
             0.5 * ridge * sum(abs2, z)
    u = _loss_derivative!(problem.loss, residual)
    gx = adjoint(problem.A) * u
    scale, penalty_dual = _penalty_dual(problem.penalty, gx, lambda,
                                        problem.x_constraint)
    gz = adjoint(problem.B) * u
    ridge_dual = problem.z_constraint == :signed ?
                 sum(abs2, gz) :
                 sum(abs2, min.(gz, zero(eltype(gz))))
    gap = primal + _loss_conjugate(problem.loss, u) / scale^2 +
          dot(problem.b, u) / scale +
          ridge_dual / (2 * ridge * scale^2) + penalty_dual / scale^2
    return max(zero(gap), gap)
end

"""Owned sparse and ridge coefficient snapshots with joint KKT diagnostics."""
struct BlockResult{VX,VZ,T,D}
    x::VX
    z::VZ
    lambda::T
    ridge::T
    objective::T
    residual_norm::T
    kkt_residual::T
    kkt_ratio::T
    converged::Bool
    stop_reason::Symbol
    iterations::Int
    diagnostics::D
end

function block_kkt_residual(problem::BlockLassoRidgeProblem, x, z,
                            lambda, ridge)
    _validate_lambda(lambda)
    _validate_lambda(ridge)
    length(x) == size(problem.A, 2) ||
        throw(DimensionMismatch("x must match the A columns"))
    length(z) == size(problem.B, 2) ||
        throw(DimensionMismatch("z must match the B columns"))
    problem.x_constraint == :nonnegative && minimum(x) < 0 &&
        throw(ArgumentError("x must be nonnegative"))
    problem.z_constraint == :nonnegative && minimum(z) < 0 &&
        throw(ArgumentError("z must be nonnegative"))
    residual = _loss_derivative!(problem.loss,
                                 problem.A * x + problem.B * z - problem.b)
    gx = adjoint(problem.A) * residual
    gz = adjoint(problem.B) * residual .+ ridge .* z
    vx = _kkt_violation.(problem.penalty, x, gx, lambda,
                         problem.x_constraint == :nonnegative)
    vz = problem.z_constraint == :nonnegative ?
         ifelse.(z .> 0, abs.(gz), max.(-gz, 0)) : abs.(gz)
    return max(maximum(vx), maximum(vz))
end

function _block_result(problem, coefficients, nx, lambda, ridge,
                       converged, reason, iterations, diagnostics)
    x = copy(view(coefficients, 1:nx))
    z = copy(view(coefficients, nx+1:length(coefficients)))
    r = problem.A * x + problem.B * z - problem.b
    residual = norm(r)
    objective = _loss_value(problem.loss, r) +
                lambda * _penalty_value(problem.penalty, x) +
                0.5 * ridge * sum(abs2, z)
    kkt = block_kkt_residual(problem, x, z, lambda, ridge)
    ratio = iszero(lambda) ? (iszero(kkt) ? zero(kkt) : oftype(kkt, Inf)) :
            kkt / (lambda * _penalty_slope(problem.penalty))
    T = promote_type(typeof(objective), typeof(float(lambda)),
                     typeof(float(ridge)))
    _warn_kkt_ratio(ratio, lambda, reason)
    return BlockResult(x, z, T(lambda), T(ridge), T(objective),
                       T(residual), T(kkt), T(ratio), converged, reason,
                       iterations, diagnostics)
end

"""Matrix-free `[A B]` operator, preserving dense or sparse device storage."""
struct BlockDesign{T,MA,MB,V} <: AbstractMatrix{T}
    A::MA
    B::MB
    x::V
    z::V
    observations::V
    gradient_x::V
    gradient_z::V
end

function BlockDesign(A, B, b)
    T = eltype(b)
    make(n) = similar(b, T, n)
    x = make(size(A, 2))
    return BlockDesign{T,typeof(A),typeof(B),typeof(x)}(
        A, B, x, make(size(B, 2)), make(length(b)),
        make(size(A, 2)), make(size(B, 2)))
end

Base.size(D::BlockDesign) = (size(D.A, 1), size(D.A, 2) + size(D.B, 2))
Base.size(D::BlockDesign, i::Integer) = i == 1 ? size(D.A, 1) :
    i == 2 ? size(D.A, 2) + size(D.B, 2) : 1
LinearAlgebra.norm(D::BlockDesign) = sqrt(sum(abs2, D.A) + sum(abs2, D.B))

function LinearAlgebra.mul!(out::AbstractVector, D::BlockDesign,
                            coefficients::AbstractVector)
    nx = size(D.A, 2)
    @views D.x .= coefficients[1:nx]
    @views D.z .= coefficients[nx+1:length(coefficients)]
    mul!(out, D.A, D.x)
    mul!(D.observations, D.B, D.z)
    out .+= D.observations
    return out
end

function LinearAlgebra.mul!(out::AbstractVector,
                            Dt::Adjoint{<:Any,<:BlockDesign},
                            residual::AbstractVector)
    D = parent(Dt)
    nx = size(D.A, 2)
    mul!(D.gradient_x, adjoint(D.A), residual)
    mul!(D.gradient_z, adjoint(D.B), residual)
    @views out[1:nx] .= D.gradient_x
    @views out[nx+1:length(out)] .= D.gradient_z
    return out
end

"""
Reusable FISTA solver for the sparse-plus-ridge block objective. CPU dense
designs materialize `[A B]`; other designs use a matrix-free block operator.
"""
mutable struct BlockFISTASolver{P,S,T}
    problem::P
    base::S
    nx::Int
    ridge::T
    step::T
end

function BlockFISTASolver(problem::BlockLassoRidgeProblem, ridge;
                          step=:auto, stop=DualGapStop(), x0=nothing,
                          z0=nothing, power_iterations=50,
                          power_safety=1.2)
    _validate_lambda(ridge)
    T = eltype(problem.b)
    nx, nz = size(problem.A, 2), size(problem.B, 2)
    x_initial = similar(problem.b, T, nx)
    z_initial = similar(problem.b, T, nz)
    length(isnothing(x0) ? x_initial : x0) == nx ||
        throw(DimensionMismatch("x0 has wrong length"))
    length(isnothing(z0) ? z_initial : z0) == nz ||
        throw(DimensionMismatch("z0 has wrong length"))
    isnothing(x0) ? fill!(x_initial, zero(T)) : copyto!(x_initial, x0)
    isnothing(z0) ? fill!(z_initial, zero(T)) : copyto!(z_initial, z0)
    all(isfinite, x_initial) && all(isfinite, z_initial) ||
        throw(ArgumentError("initial coefficients must be finite"))
    problem.x_constraint == :nonnegative && minimum(x_initial) < 0 &&
        throw(ArgumentError("x0 must be nonnegative"))
    problem.z_constraint == :nonnegative && minimum(z_initial) < 0 &&
        throw(ArgumentError("z0 must be nonnegative"))
    dense_cpu = problem.A isa StridedMatrix && problem.B isa StridedMatrix
    combined = dense_cpu ? hcat(problem.A, problem.B) :
               BlockDesign(problem.A, problem.B, problem.b)
    # The two constituent matrices and b were validated by the block problem.
    base_problem = dense_cpu ?
                   LassoProblem(combined, problem.b; loss=problem.loss) :
                   LassoProblem{typeof(combined),typeof(problem.b),
                                typeof(problem.loss),L1}(
                       combined, problem.b, :signed, problem.loss, L1())
    initial = similar(problem.b, T, nx+nz)
    @views initial[1:nx] .= x_initial
    @views initial[nx+1:end] .= z_initial
    base_step = if combined isa BlockDesign && step === :frobenius
        power = sum(abs2, problem.A) + sum(abs2, problem.B)
        iszero(power) ? one(T) : inv(T(power))
    else
        step
    end
    base = FISTASolver(base_problem;
                       step=base_step, stop, x0=initial,
                       power_iterations, power_safety)
    # A numeric step is interpreted as the step for the full block objective.
    actual_step = step isa Real ? base.step : inv(inv(base.step) + ridge)
    actual_step > 0 || throw(ArgumentError("block step underflowed to zero"))
    return BlockFISTASolver(problem, base, nx, T(ridge), T(actual_step))
end

function solve!(solver::BlockFISTASolver, lambda; warm_start=true)
    _validate_lambda(lambda)
    return _fista_solve!(solver, lambda; warm_start)
end

_fista_workspace(solver::BlockFISTASolver) = solver.base
_fista_step(solver::BlockFISTASolver) = solver.step
_block_sparse(solver::BlockFISTASolver) = view(solver.base.x, 1:solver.nx)
_block_smooth(solver::BlockFISTASolver) =
    view(solver.base.x, solver.nx+1:length(solver.base.x))

function _fista_gradient!(solver::BlockFISTASolver, y)
    p, base, nx = solver.problem, solver.base, solver.nx
    mul!(base.residual, base.problem.A, y)
    @. base.residual = base.residual - p.b
    _loss_derivative!(p.loss, base.residual)
    mul!(base.gradient, adjoint(base.problem.A), base.residual)
    @views base.gradient[nx+1:end] .+= solver.ridge .* y[nx+1:end]
    return nothing
end

function _fista_prox!(solver::BlockFISTASolver, step, lambda)
    p, base, nx = solver.problem, solver.base, solver.nx
    @views begin
        sparse_input = base.prox_input[1:nx]
        smooth_input = base.prox_input[nx+1:end]
        sparse_output = base.x[1:nx]
        smooth_output = base.x[nx+1:end]
        _penalty_prox!(p.penalty, sparse_output, sparse_input,
                       step * lambda, p.x_constraint)
        if p.z_constraint == :nonnegative
            @. smooth_output = max(smooth_input, 0)
        else
            copyto!(smooth_output, smooth_input)
        end
    end
    return nothing
end

_fista_kkt(solver::BlockFISTASolver, lambda) =
    block_kkt_residual(solver.problem, _block_sparse(solver),
                       _block_smooth(solver), lambda, solver.ridge)

# The dual certificate needs both penalties to be positive.
_fista_gap_degenerate(solver::BlockFISTASolver, lambda) =
    iszero(lambda) || iszero(solver.ridge)
_fista_degenerate_reason(solver::BlockFISTASolver) = :kkt_dual_degenerate

function _fista_kkt_scale(solver::BlockFISTASolver)
    p = solver.problem
    return max(norm(_loss_correlations(p.A, p.b, p.loss), Inf),
               norm(_loss_correlations(p.B, p.b, p.loss), Inf), 1)
end

_fista_gap(solver::BlockFISTASolver, lambda) =
    block_dual_gap(solver.problem, _block_sparse(solver),
                   _block_smooth(solver), lambda, solver.ridge)

_fista_gap_tolerance(solver::BlockFISTASolver, stop) =
    _dual_gap_tolerance(solver.problem.b, solver.problem.loss, stop)

_fista_result(solver::BlockFISTASolver, lambda, converged, reason, iterations,
              diagnostics) =
    _block_result(solver.problem, solver.base.x, solver.nx, lambda,
                  solver.ridge, converged, reason, iterations, diagnostics)

initialize(problem::BlockLassoRidgeProblem, method::FISTA, ridge;
           x0=nothing, z0=nothing) =
    BlockFISTASolver(problem, ridge; step=method.step, stop=method.stop,
                      power_iterations=method.power_iterations,
                      power_safety=method.power_safety, x0, z0)

function solve(problem::BlockLassoRidgeProblem, method::FISTA, lambda, ridge;
               x0=nothing, z0=nothing)
    return solve!(initialize(problem, method, ridge; x0, z0), lambda)
end

"""Warm-start a sparse-plus-ridge solve over λ values at fixed ridge penalty."""
function solve_path!(solver::BlockFISTASolver, lambdas;
                     on_result=nothing, store=true, stop_on_failure=false,
                     restart=false)
    values = collect(lambdas)
    isempty(values) && throw(ArgumentError("lambdas must be nonempty"))
    foreach(_validate_lambda, values)
    results = store ? BlockResult[] : nothing
    for (index, lambda) in enumerate(values)
        result = solve!(solver, lambda; warm_start=index > 1 || !restart)
        store && push!(results, result)
        isnothing(on_result) || on_result(index, result)
        stop_on_failure && !result.converged && break
    end
    return results
end

function solve_path(problem::BlockLassoRidgeProblem, method::FISTA,
                    lambdas, ridge; x0=nothing, z0=nothing, kwargs...)
    solver = initialize(problem, method, ridge; x0, z0)
    return solve_path!(solver, lambdas; kwargs...)
end

export BlockLassoRidgeProblem, BlockResult, BlockFISTASolver
export block_kkt_residual, block_dual_gap
