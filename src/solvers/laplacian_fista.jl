"""L1-penalized least squares with squared graph-Laplacian smoothing."""
struct LaplacianLassoProblem{MA<:AbstractMatrix,VB<:AbstractVector,T}
    A::MA
    b::VB
    image_size::Tuple{Int,Int}
    smooth::T
    constraint::Symbol
end

function LaplacianLassoProblem(A::AbstractMatrix, b::AbstractVector,
                               image_size::Tuple, smooth;
                               constraint=:signed)
    LassoProblem(A, b; constraint)
    A isa StridedMatrix && b isa StridedVector ||
        throw(ArgumentError("Laplacian FISTA currently requires CPU dense arrays"))
    length(image_size) == 2 && all(d -> d isa Integer && d > 0, image_size) ||
        throw(ArgumentError("image_size must contain two positive integers"))
    dims = (Int(image_size[1]), Int(image_size[2]))
    prod(dims) == size(A, 2) ||
        throw(DimensionMismatch("A must have one column per image pixel"))
    _validate_lambda(smooth)
    return LaplacianLassoProblem(A, b, dims, eltype(b)(smooth), constraint)
end

"""Positive 2D graph Laplacian in Julia column-major pixel order."""
function laplacian2d(x::AbstractVector, image_size::Tuple{Int,Int})
    result = similar(x)
    _laplacian2d!(result, x, image_size)
    return result
end

function _laplacian2d!(out, x, image_size)
    height, width = image_size
    length(x) == height * width ||
        throw(DimensionMismatch("x must have prod(image_size) entries"))
    @inbounds for j in 1:width, i in 1:height
        k = i + (j - 1) * height
        value = zero(eltype(out))
        i > 1 && (value += x[k] - x[k - 1])
        i < height && (value += x[k] - x[k + 1])
        j > 1 && (value += x[k] - x[k - height])
        j < width && (value += x[k] - x[k + height])
        out[k] = value
    end
    return out
end

"""
    laplacian_dual_gap(problem, x, lambda)

Primal-dual gap for L1 plus squared-Laplacian smoothing. The dual
candidates `u=Ax-b` and `v=smooth*L*x` are scaled together until
`A'u + L'v` satisfies the L1 dual constraint.
"""
function laplacian_dual_gap(problem::LaplacianLassoProblem, x, lambda)
    _validate_lambda(lambda)
    lambda > 0 || throw(ArgumentError("lambda must be positive for the dual gap"))
    problem.constraint == :nonnegative && minimum(x) < 0 &&
        throw(ArgumentError("x must be nonnegative"))
    residual = problem.A * x - problem.b
    lap = laplacian2d(x, problem.image_size)
    gradient = adjoint(problem.A) * residual
    if !iszero(problem.smooth)
        gradient .+= problem.smooth .* laplacian2d(lap, problem.image_size)
    end
    bound = problem.constraint == :signed ?
            maximum(abs, gradient) :
            max(zero(lambda), maximum(-gradient))
    scale = max(one(float(lambda)), bound / lambda)
    primal = 0.5 * sum(abs2, residual) + lambda * sum(abs, x) +
             0.5 * problem.smooth * sum(abs2, lap)
    gap = primal + (sum(abs2, residual) +
                    problem.smooth * sum(abs2, lap)) / (2 * scale^2) +
          dot(problem.b, residual) / scale
    return max(zero(gap), gap)
end

"""Owned result of the Laplacian-regularized LASSO problem."""
struct LaplacianResult{V,T,D}
    x::V
    lambda::T
    smooth::T
    objective::T
    residual_norm::T
    kkt_residual::T
    kkt_ratio::T
    converged::Bool
    stop_reason::Symbol
    iterations::Int
    diagnostics::D
end

function laplacian_kkt_residual(problem::LaplacianLassoProblem,
                                x::AbstractVector, lambda)
    _validate_lambda(lambda)
    length(x) == size(problem.A, 2) ||
        throw(DimensionMismatch("x must match the A columns"))
    if problem.constraint == :nonnegative && minimum(x) < 0
        throw(ArgumentError("x must be nonnegative"))
    end
    gradient = adjoint(problem.A) * (problem.A * x - problem.b)
    if problem.smooth != 0
        lap = laplacian2d(x, problem.image_size)
        gradient .+= problem.smooth .* laplacian2d(lap, problem.image_size)
    end
    if problem.constraint == :nonnegative
        violation = ifelse.(x .> 0, abs.(gradient .+ lambda),
                            max.(-gradient .- lambda, 0))
    else
        violation = ifelse.(x .> 0, abs.(gradient .+ lambda),
                            ifelse.(x .< 0, abs.(gradient .- lambda),
                                    max.(abs.(gradient) .- lambda, 0)))
    end
    return maximum(violation)
end

function _laplacian_result(problem, x, lambda, converged, reason,
                           iterations, diagnostics)
    owned = copy(x)
    residual = norm(problem.A * owned - problem.b)
    lap = laplacian2d(owned, problem.image_size)
    objective = 0.5 * residual^2 + lambda * sum(abs, owned) +
                0.5 * problem.smooth * sum(abs2, lap)
    kkt = laplacian_kkt_residual(problem, owned, lambda)
    ratio = iszero(lambda) ? (iszero(kkt) ? zero(kkt) : oftype(kkt, Inf)) :
            kkt / lambda
    T = promote_type(typeof(objective), typeof(float(lambda)))
    _warn_kkt_ratio(ratio, lambda, reason)
    return LaplacianResult(owned, T(lambda), T(problem.smooth),
                           T(objective), T(residual), T(kkt), T(ratio),
                           converged, reason, iterations, diagnostics)
end

mutable struct LaplacianFISTASolver{P,S,V,T}
    problem::P
    base::S
    laplacian::V
    bilaplacian::V
    step::T
end

function LaplacianFISTASolver(problem::LaplacianLassoProblem;
                              step=:auto, stop=DualGapStop(), x0=nothing,
                              power_iterations=50, power_safety=1.2)
    base = FISTASolver(LassoProblem(problem.A, problem.b;
                                    constraint=problem.constraint);
                       step, stop, x0, power_iterations, power_safety)
    # ‖L‖₂ ≤ 8 for the 2D nearest-neighbor graph Laplacian.
    actual_step = step isa Real ? base.step :
                  inv(inv(base.step) + 64 * problem.smooth)
    return LaplacianFISTASolver(problem, base, similar(base.x),
                                similar(base.x), actual_step)
end

function solve!(solver::LaplacianFISTASolver, lambda; warm_start=true)
    _validate_lambda(lambda)
    return _fista_solve!(solver, lambda; warm_start)
end

_fista_workspace(solver::LaplacianFISTASolver) = solver.base
_fista_step(solver::LaplacianFISTASolver) = solver.step

function _fista_gradient!(solver::LaplacianFISTASolver, y)
    p, base = solver.problem, solver.base
    mul!(base.residual, p.A, y)
    @. base.residual = base.residual - p.b
    mul!(base.gradient, adjoint(p.A), base.residual)
    if p.smooth != 0
        _laplacian2d!(solver.laplacian, y, p.image_size)
        _laplacian2d!(solver.bilaplacian, solver.laplacian, p.image_size)
        @. base.gradient = base.gradient + p.smooth * solver.bilaplacian
    end
    return nothing
end

function _fista_prox!(solver::LaplacianFISTASolver, step, lambda)
    base = solver.base
    if solver.problem.constraint == :nonnegative
        @. base.x = max(base.prox_input - step * lambda, 0)
    else
        @. base.x = sign(base.prox_input) *
                    max(abs(base.prox_input) - step * lambda, 0)
    end
    return nothing
end

_fista_kkt(solver::LaplacianFISTASolver, lambda) =
    laplacian_kkt_residual(solver.problem, solver.base.x, lambda)

_fista_kkt_scale(solver::LaplacianFISTASolver) =
    max(norm(adjoint(solver.problem.A) * solver.problem.b, Inf), 1)

_fista_gap(solver::LaplacianFISTASolver, lambda) =
    laplacian_dual_gap(solver.problem, solver.base.x, lambda)

_fista_gap_tolerance(solver::LaplacianFISTASolver, stop) =
    _dual_gap_tolerance(solver.problem.b, stop)

_fista_result(solver::LaplacianFISTASolver, lambda, converged, reason,
              iterations, diagnostics) =
    _laplacian_result(solver.problem, solver.base.x, lambda, converged,
                      reason, iterations, diagnostics)

initialize(problem::LaplacianLassoProblem, method::FISTA; x0=nothing) =
    LaplacianFISTASolver(problem; step=method.step, stop=method.stop,
                          power_iterations=method.power_iterations,
                          power_safety=method.power_safety, x0)

function solve(problem::LaplacianLassoProblem, method::FISTA, lambda;
               x0=nothing)
    return solve!(initialize(problem, method; x0), lambda)
end

export LaplacianLassoProblem, LaplacianResult, LaplacianFISTASolver
export laplacian2d, laplacian_kkt_residual, laplacian_dual_gap
