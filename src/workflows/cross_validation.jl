"""
    ordered_group_folds(groups, n_folds; order=nothing,
                        assignment=:round_robin)

Keep every row with the same group in one validation fold. `order` is an
explicit ordering of *all* distinct groups (e.g. sorted UTC dates); the
default is first-appearance order. `:round_robin` distributes consecutive
groups over folds, while `:contiguous` makes contiguous blocks of groups.
Neither mode shuffles rows. Returns `fold_ids`, `ordered_groups`, and
`group_fold_ids`.
"""
function ordered_group_folds(groups, n_folds;
                             order=nothing, assignment=:round_robin)
    group_values = collect(groups)
    ordered = isnothing(order) ? unique(group_values) : collect(order)
    isempty(group_values) && throw(ArgumentError("groups must be nonempty"))
    n_folds isa Integer && 2 <= n_folds <= length(ordered) ||
        throw(ArgumentError("n_folds must be between 2 and the number of groups"))
    assignment in (:round_robin, :contiguous) ||
        throw(ArgumentError("assignment must be :round_robin or :contiguous"))
    length(unique(ordered)) == length(ordered) &&
        Set(ordered) == Set(group_values) ||
        throw(ArgumentError("order must contain each observed group exactly once"))
    n_groups = length(ordered)
    group_fold_ids = [assignment == :round_robin ?
                      mod1(i, n_folds) :
                      fld((i - 1) * n_folds, n_groups) + 1
                      for i in eachindex(ordered)]
    lookup = Dict(group => fold for (group, fold) in
                  zip(ordered, group_fold_ids))
    fold_ids = [lookup[group] for group in group_values]
    return (; fold_ids, ordered_groups=ordered, group_fold_ids)
end

function _validated_cv_folds(fold_ids, n_rows)
    ids = collect(fold_ids)
    length(ids) == n_rows ||
        throw(DimensionMismatch("fold_ids must have one entry per observation"))
    all(id -> id isa Integer && id > 0, ids) ||
        throw(ArgumentError("fold IDs must be positive integers"))
    folds = sort(unique(ids))
    length(folds) >= 2 && folds == collect(1:length(folds)) ||
        throw(ArgumentError("fold IDs must be consecutive integers starting at 1"))
    return ids, length(folds)
end

_cv_rmse(prediction, observed, _) =
    norm(prediction - observed) / sqrt(length(observed))

"""
    cross_validate(problem::LassoProblem, method; fold_ids,
                   lambda_ratios=..., score=_cv_rmse)

Grouped K-fold CV for a signed or nonnegative LASSO. The caller supplies fold
IDs (or obtains them from `ordered_group_folds`); Prism never randomly
splits observations. For each training fold, compute its own zero-solution
`lambda_max`, then solve the *same descending lambda/lambda_max ratios* with
warm starts. The default score is held-out RMSE in the units of `problem.b`.
A custom `score(prediction, observed, validation_rows)` must return one finite
number to minimize; it can use `validation_rows` to evaluate raw-unit errors
when the fitted problem was whitened.

The result contains per-fold scores, convergence flags, iterations, and
fold-specific lambdas, plus mean/median/SE curves. `best_index` minimizes
mean score; `one_se_index` chooses the largest lambda whose mean is within
one SE of the best. Only ratios that converged in **every** fold are eligible
for selection. Both indices are `nothing` if none qualify. An unconverged
fold's score is still retained for diagnosis, not treated as a valid model.
"""
function cross_validate(problem::LassoProblem, method::Union{FISTA,ADMM};
                        fold_ids,
                        lambda_ratios=exp.(range(0.0,
                                                log(DEFAULT_LASSO_PATH_MIN_RATIO);
                                                length=DEFAULT_LASSO_PATH_POINTS)),
                        score=_cv_rmse)
    ids, n_folds = _validated_cv_folds(fold_ids, length(problem.b))
    ratios = collect(lambda_ratios)
    isempty(ratios) && throw(ArgumentError("lambda_ratios must be nonempty"))
    all(r -> r isa Real && isfinite(r) && 0 < r <= 1, ratios) &&
        all(diff(ratios) .< 0) ||
        throw(ArgumentError("lambda_ratios must be finite, positive, at most 1, and strictly descending"))
    n_ratios = length(ratios)
    fold_scores = Matrix{Float64}(undef, n_ratios, n_folds)
    fold_converged = falses(n_ratios, n_folds)
    fold_iterations = Matrix{Int}(undef, n_ratios, n_folds)
    fold_lambdas = Matrix{Float64}(undef, n_ratios, n_folds)
    fold_lambda_max = Vector{Float64}(undef, n_folds)
    validation_counts = Vector{Int}(undef, n_folds)
    for fold in 1:n_folds
        training = findall(!=(fold), ids)
        validation = findall(==(fold), ids)
        validation_counts[fold] = length(validation)
        train_problem = LassoProblem(problem.A[training, :],
                                     problem.b[training];
                                     constraint=problem.constraint)
        lambda_max = first(lambda_grid(train_problem; n_lambdas=2))
        lambda_max > 0 ||
            throw(ArgumentError("fold $fold has lambda_max=0; a positive normalized path is undefined"))
        fold_lambda_max[fold] = Float64(lambda_max)
        lambdas = lambda_max .* ratios
        fold_lambdas[:, fold] .= lambdas
        solver = initialize(train_problem, method)
        heldout_A = problem.A[validation, :]
        observed = problem.b[validation]
        solve_path!(solver, lambdas; store=false, restart=true,
                    on_result=(index, fit) -> begin
                        prediction = heldout_A * fit.x
                        value = score(prediction, observed, validation)
                        value isa Real && isfinite(value) ||
                            throw(ArgumentError("score must return a finite real number"))
                        fold_scores[index, fold] = Float64(value)
                        fold_converged[index, fold] = fit.converged
                        fold_iterations[index, fold] = fit.iterations
                    end)
    end
    mean_scores = vec(mean(fold_scores; dims=2))
    median_scores = vec(median(fold_scores; dims=2))
    se_scores = vec(std(fold_scores; dims=2)) ./ sqrt(n_folds)
    all_converged = vec(all(fold_converged; dims=2))
    eligible = findall(all_converged)
    best_index = isempty(eligible) ? nothing :
                 eligible[argmin(mean_scores[eligible])]
    one_se_index = isnothing(best_index) ? nothing :
                   first(filter(i -> all_converged[i] &&
                                  mean_scores[i] <=
                                  mean_scores[best_index] + se_scores[best_index],
                                eachindex(ratios)))
    return (; lambda_ratios=ratios, fold_ids=ids, fold_lambda_max,
            fold_lambdas, fold_scores, fold_converged, fold_iterations,
            validation_counts, mean_scores, median_scores, se_scores,
            all_converged, best_index, one_se_index)
end

export ordered_group_folds, cross_validate
