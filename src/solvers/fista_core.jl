# The single FISTA iteration shared by every FISTA-based solver in Prism.
#
# A solver type takes part by defining the small "hook" functions below. The
# loop itself, the momentum rule, and the three stopping rules live only here.
#
# Required hooks (no default):
#   _fista_workspace(solver)   -> the FISTASolver holding x, y, previous,
#                                 gradient, prox_input, residual and stop
#   _fista_gradient!(solver, y)        write the smooth-part gradient at y
#                                      into the workspace `gradient`
#   _fista_prox!(solver, step, lambda) write the prox of workspace
#                                      `prox_input` into workspace `x`
#   _fista_kkt(solver, lambda)         optimality residual at workspace `x`
#   _fista_gap(solver, lambda)         primal-dual gap at workspace `x`
#                                      (both may overwrite the workspace
#                                      `residual`, `gradient` and `prox_input`,
#                                      which are recomputed in each iteration)
#   _fista_result(solver, lambda, converged, reason, iterations, diagnostics)
#
# Optional hooks (defaults below): _fista_step, _fista_kkt_reason,
# _fista_kkt_passed, _fista_gap_degenerate, _fista_degenerate_reason,
# _fista_kkt_scale, _fista_gap_tolerance, _fista_mapping_scale, _fista_check.

_fista_step(solver) = _fista_workspace(solver).step
_fista_kkt_reason(solver) = :kkt
_fista_kkt_passed(solver, lambda, stop::KKTStop) =
    _fista_kkt(solver, lambda) <= _kkt_tolerance(stop, lambda)

# With DualGapStop: when no usable dual certificate exists, fall back to a
# scaled KKT test and report `_fista_degenerate_reason`.
_fista_gap_degenerate(solver, lambda) = iszero(lambda)
_fista_degenerate_reason(solver) = :kkt_zero_penalty
_fista_mapping_scale(solver, lambda, step) =
    max(norm(_fista_workspace(solver).gradient, Inf), eps(float(step)))

# Each check returns `nothing` to continue, or `(reason, diagnostics)` to stop.
function _fista_check(solver, lambda, stop::KKTStop, step)
    _fista_kkt_passed(solver, lambda, stop) || return nothing
    return (reason=_fista_kkt_reason(solver), diagnostics=(;))
end

# `gap_tolerance` depends only on the data, so `_fista_solve!` computes it once
# per solve instead of once per check. The iterate change is formed in
# `prox_input`, which is free after the prox step: no array is allocated here.
function _fista_check(solver, lambda, stop::DualGapStop, step,
                      gap_tolerance=_fista_gap_tolerance(solver, stop))
    ws = _fista_workspace(solver)
    @. ws.prox_input = ws.x - ws.previous
    change = norm(ws.prox_input, Inf)
    change <= stop.tol * norm(ws.x, Inf) || return nothing
    if _fista_gap_degenerate(solver, lambda)
        kkt = _fista_kkt(solver, lambda)
        kkt <= stop.tol * _fista_kkt_scale(solver) || return nothing
        return (reason=_fista_degenerate_reason(solver),
                diagnostics=(iterate_change=change,))
    end
    gap = _fista_gap(solver, lambda)
    gap <= gap_tolerance || return nothing
    return (reason=:dual_gap, diagnostics=(iterate_change=change, dual_gap=gap))
end

function _fista_check(solver, lambda, stop::GradientMappingStop, step)
    ws = _fista_workspace(solver)
    @. ws.prox_input = (ws.y - ws.x) / step
    mapping = norm(ws.prox_input, Inf)
    scale = _fista_mapping_scale(solver, lambda, step)
    mapping <= stop.atol + stop.rtol * scale || return nothing
    return (reason=:gradient_mapping,
            diagnostics=(gradient_mapping_residual=mapping,
                         gradient_mapping_scale=scale))
end

function _fista_solve!(solver, lambda; warm_start=true)
    ws = _fista_workspace(solver)
    stop, step = ws.stop, _fista_step(solver)
    x, y, previous = ws.x, ws.y, ws.previous
    if !warm_start
        fill!(x, zero(eltype(x)))
    end
    copyto!(y, x)
    if stop isa KKTStop && _fista_kkt_passed(solver, lambda, stop)
        return _fista_result(solver, lambda, true, _fista_kkt_reason(solver),
                             0, (step=step,))
    end
    gap_tolerance = stop isa DualGapStop ?
                    _fista_gap_tolerance(solver, stop) : nothing
    t = one(step)
    for iteration in 1:stop.maxiter
        copyto!(previous, x)
        _fista_gradient!(solver, y)
        @. ws.prox_input = y - step * ws.gradient
        _fista_prox!(solver, step, lambda)
        if iteration % stop.check_every == 0 || iteration == stop.maxiter
            outcome = stop isa DualGapStop ?
                      _fista_check(solver, lambda, stop, step, gap_tolerance) :
                      _fista_check(solver, lambda, stop, step)
            if !isnothing(outcome)
                return _fista_result(solver, lambda, true, outcome.reason,
                                     iteration,
                                     merge((step=step,), outcome.diagnostics))
            end
        end
        next_t = (one(t) + sqrt(one(t) + 4t^2)) / 2
        momentum = (t - one(t)) / next_t
        @. y = x + momentum * (x - previous)
        t = next_t
    end
    return _fista_result(solver, lambda, false, :maxiter, stop.maxiter,
                         (step=step,))
end
