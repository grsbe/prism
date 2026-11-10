"""
    GroupLassoProblem(A, b, groups; constraint=:signed, weights=nothing,
                      loss=LeastSquares())

Disjoint group LASSO: minimize ℓ(Ax-b) + λ∑₍g₎w₍g₎‖x₍g₎‖₂,
optionally with x ≥ 0. The loss defaults to ½‖r‖²; `loss=Huber(delta)`
selects the Huber loss. Every column must occur in exactly one nonempty group.
Default weights are √(group size). Groups may contain noncontiguous indices.
"""
struct GroupLassoProblem{MA<:AbstractMatrix,VB<:AbstractVector,T,I,L<:DataLoss}
    A::MA
    b::VB
    groups::Vector{Vector{Int}}
    indices::Vector{I}
    weights::Vector{T}
    constraint::Symbol
    loss::L
end

function GroupLassoProblem(A::AbstractMatrix, b::AbstractVector, groups;
                           constraint=:signed, weights=nothing,
                           loss=LeastSquares())
    loss = LassoProblem(A, b; constraint, loss).loss
    n = size(A, 2)
    isempty(groups) && throw(ArgumentError("groups must be nonempty"))
    owned = Vector{Vector{Int}}()
    seen = falses(n)
    for raw in groups
        isempty(raw) && throw(ArgumentError("each group must be nonempty"))
        g = Int[]
        for j in raw
            j isa Integer && 1 <= j <= n ||
                throw(ArgumentError("group indices must be integers in 1:$n"))
            seen[j] && throw(ArgumentError("groups must be disjoint"))
            seen[j] = true
            push!(g, Int(j))
        end
        push!(owned, g)
    end
    all(seen) || throw(ArgumentError("groups must cover every column"))
    T = eltype(b)
    w = isnothing(weights) ? T[sqrt(T(length(g))) for g in owned] :
        T.(collect(weights))
    length(w) == length(owned) ||
        throw(DimensionMismatch("one weight is required per group"))
    all(v -> isfinite(v) && v > 0, w) ||
        throw(ArgumentError("group weights must be finite and positive"))
    indices = [begin
        idx = similar(b, Int, length(g))
        copyto!(idx, g)
        idx
    end for g in owned]
    return GroupLassoProblem(A, b, owned, indices, w, constraint, loss)
end

function _group_penalty(problem::GroupLassoProblem, x)
    total = zero(eltype(problem.b))
    for (idx, w) in zip(problem.indices, problem.weights)
        total += w * norm(view(x, idx))
    end
    return total
end

function objective_value(problem::GroupLassoProblem, x::AbstractVector, lambda)
    _validate_lambda(lambda)
    length(x) == size(problem.A, 2) ||
        throw(DimensionMismatch("x must have one entry per column"))
    problem.constraint == :nonnegative && minimum(x) < 0 && return Inf
    return _loss_value(problem.loss, problem.A * x - problem.b) +
           lambda * _group_penalty(problem, x)
end

"""
Largest groupwise KKT violation divided by its group weight, in gradient units. At a zero group this is
max(‖g₍g₎‖₂-λw₍g₎,0) for signed coefficients, using only negative
gradient components for nonnegative coefficients.
"""
function group_lasso_kkt_residual(problem::GroupLassoProblem, x, lambda)
    _validate_lambda(lambda)
    length(x) == size(problem.A, 2) ||
        throw(DimensionMismatch("x must have one entry per column"))
    problem.constraint == :nonnegative && minimum(x) < 0 &&
        throw(ArgumentError("x must be nonnegative"))
    gradient = adjoint(problem.A) *
               _loss_derivative!(problem.loss, problem.A * x - problem.b)
    worst = zero(eltype(problem.b))
    for (idx, w) in zip(problem.indices, problem.weights)
        xv, gv = view(x, idx), view(gradient, idx)
        size_x = norm(xv)
        if iszero(size_x)
            size_g = problem.constraint == :signed ?
                     norm(gv) : norm(min.(gv, zero(eltype(gv))))
            violation = max(size_g - lambda * w, zero(size_g))
        elseif problem.constraint == :signed
            violation = norm(gv .+ (lambda * w / size_x) .* xv)
        else
            shifted = gv .+ (lambda * w / size_x) .* xv
            violation = norm(ifelse.(xv .> 0, abs.(shifted),
                                      max.(-shifted, zero(eltype(gv)))))
        end
        worst = max(worst, violation / w)
    end
    return worst
end

"""
Feasible primal-dual gap for positive-λ disjoint group LASSO. The dual
point is the scaled loss derivative u = ψ(Ax-b); the dual constraints are ‖A₍g₎ᵀu‖₂ ≤ λw₍g₎ (signed) or
‖(-A₍g₎ᵀu)₊‖₂ ≤ λw₍g₎ (nonnegative).
"""
function group_lasso_dual_gap(problem::GroupLassoProblem, x, lambda)
    _validate_lambda(lambda)
    length(x) == size(problem.A, 2) ||
        throw(DimensionMismatch("x must have one entry per column"))
    problem.constraint == :nonnegative && minimum(x) < 0 &&
        throw(ArgumentError("x must be nonnegative"))
    r = problem.A * x - problem.b
    primal = _loss_value(problem.loss, r) + lambda * _group_penalty(problem, x)
    iszero(lambda) && return primal # The solver uses KKT instead at λ=0.
    u = _loss_derivative!(problem.loss, r)
    gradient = adjoint(problem.A) * u
    scale = one(float(lambda))
    for (idx, w) in zip(problem.indices, problem.weights)
        gv = view(gradient, idx)
        size_g = problem.constraint == :signed ?
                 norm(gv) : norm(min.(gv, zero(eltype(gv))))
        scale = max(scale, size_g / (lambda * w))
    end
    gap = primal + _loss_conjugate(problem.loss, u) / scale^2 +
          dot(problem.b, u) / scale
    return max(zero(gap), gap)
end

function lambda_grid(problem::GroupLassoProblem; min_ratio=1e-4, n_lambdas=30)
    min_ratio isa Real && isfinite(min_ratio) && 0 < min_ratio < 1 ||
        throw(ArgumentError("min_ratio must be finite and between 0 and 1"))
    n_lambdas isa Integer && n_lambdas >= 2 ||
        throw(ArgumentError("n_lambdas must be at least two"))
    correlations = _loss_correlations(problem.A, problem.b, problem.loss)
    largest = zero(eltype(problem.b))
    for (idx, w) in zip(problem.indices, problem.weights)
        cv = view(correlations, idx)
        size_c = problem.constraint == :signed ?
                 norm(cv) : norm(max.(cv, zero(eltype(cv))))
        largest = max(largest, size_c / w)
    end
    iszero(largest) && return [largest]
    T = typeof(largest)
    values = largest .* exp.(range(zero(T), log(T(min_ratio));
                                    length=n_lambdas))
    values[1] = largest
    values[end] = largest * T(min_ratio)
    return values
end

"""Reusable group-LASSO FISTA solver; supports CPU and compatible GPU arrays."""
mutable struct GroupFISTASolver{P,S}
    problem::P
    base::S
end

function GroupFISTASolver(problem::GroupLassoProblem; step=:auto,
                          stop=DualGapStop(), x0=nothing,
                          power_iterations=50, power_safety=1.2)
    base = FISTASolver(LassoProblem(problem.A, problem.b;
                                   constraint=problem.constraint,
                                   loss=problem.loss);
                       step, stop, x0, power_iterations, power_safety)
    return GroupFISTASolver(problem, base)
end

function _group_prox!(x, z, problem::GroupLassoProblem, threshold)
    for (idx, w) in zip(problem.indices, problem.weights)
        xv, zv = view(x, idx), view(z, idx)
        copyto!(xv, zv)
        if problem.constraint == :nonnegative
            xv .= max.(xv, zero(eltype(x)))
        end
        magnitude = norm(xv)
        if magnitude <= threshold * w
            fill!(xv, zero(eltype(x)))
        else
            xv .*= 1 - threshold * w / magnitude
        end
    end
    return x
end

function _group_result(problem, x, lambda, converged, reason, iterations,
                       diagnostics)
    owned = copy(x)
    r = problem.A * owned - problem.b
    residual = norm(r)
    objective = _loss_value(problem.loss, r) + lambda * _group_penalty(problem, owned)
    kkt = group_lasso_kkt_residual(problem, owned, lambda)
    ratio = iszero(lambda) ? (iszero(kkt) ? zero(kkt) : oftype(kkt, Inf)) :
            kkt / lambda
    T = promote_type(typeof(float(lambda)), typeof(objective))
    _warn_kkt_ratio(ratio, lambda, reason)
    return SolveResult(owned, T(lambda), T(objective), T(residual), T(kkt),
                       T(ratio), converged, reason, iterations, diagnostics)
end

function solve!(solver::GroupFISTASolver, lambda; warm_start=true)
    _validate_lambda(lambda)
    problem, base = solver.problem, solver.base
    x, y, previous = base.x, base.y, base.previous
    stop = base.stop
    !warm_start && fill!(x, zero(eltype(x)))
    copyto!(y, x)
    if stop isa KKTStop &&
       group_lasso_kkt_residual(problem, x, lambda) <=
       _kkt_tolerance(stop, lambda)
        return _group_result(problem, x, lambda, true, :kkt, 0,
                             (step=base.step,))
    end
    t = one(base.step)
    for iteration in 1:stop.maxiter
        copyto!(previous, x)
        mul!(base.residual, problem.A, y)
        @. base.residual = base.residual - problem.b
        _loss_derivative!(problem.loss, base.residual)
        mul!(base.gradient, adjoint(problem.A), base.residual)
        @. base.prox_input = y - base.step * base.gradient
        _group_prox!(x, base.prox_input, problem, base.step * lambda)
        if iteration % stop.check_every == 0 || iteration == stop.maxiter
            if stop isa KKTStop
                if group_lasso_kkt_residual(problem, x, lambda) <=
                   _kkt_tolerance(stop, lambda)
                    return _group_result(problem, x, lambda, true, :kkt,
                                         iteration, (step=base.step,))
                end
            elseif stop isa DualGapStop
                change = norm(x - previous, Inf)
                if change <= stop.tol * norm(x, Inf)
                    if iszero(lambda)
                        kkt = group_lasso_kkt_residual(problem, x, lambda)
                        scale = max(norm(_loss_correlations(problem.A, problem.b,
                                                            problem.loss), Inf), 1)
                        if kkt <= stop.tol * scale
                            return _group_result(problem, x, lambda, true,
                                                 :kkt_zero_penalty, iteration,
                                                 (step=base.step,
                                                  iterate_change=change))
                        end
                    else
                        gap = group_lasso_dual_gap(problem, x, lambda)
                        if gap <= _dual_gap_tolerance(problem.b, problem.loss, stop)
                            return _group_result(problem, x, lambda, true,
                                                 :dual_gap, iteration,
                                                 (step=base.step,
                                                  iterate_change=change,
                                                  dual_gap=gap))
                        end
                    end
                end
            else
                @. base.prox_input = (y - x) / base.step
                mapping = norm(base.prox_input, Inf)
                scale = max(norm(base.gradient, Inf), eps(float(base.step)))
                if mapping <= stop.atol + stop.rtol * scale
                    return _group_result(problem, x, lambda, true,
                                         :gradient_mapping, iteration,
                                         (step=base.step,
                                          gradient_mapping_residual=mapping,
                                          gradient_mapping_scale=scale))
                end
            end
        end
        next_t = (one(t) + sqrt(one(t) + 4t^2)) / 2
        momentum = (t - one(t)) / next_t
        @. y = x + momentum * (x - previous)
        t = next_t
    end
    return _group_result(problem, x, lambda, false, :maxiter,
                         stop.maxiter, (step=base.step,))
end
