using Test
using LinearAlgebra
using SparseArrays
using Prism

@testset "Prism" begin
    @testset "problem validation and exact KKT" begin
        A = Matrix{Float64}(I, 3, 3)
        b = [2.0, -1.0, 0.5]
        signed = LassoProblem(A, b)
        positive = LassoProblem(A, b; constraint=:nonnegative)
        @test_throws DimensionMismatch LassoProblem(A, b[1:2])
        @test_throws ArgumentError LassoProblem(A, b; constraint=:field_positive)
        @test_throws ArgumentError LassoProblem(A, [NaN, 1.0, 2.0])
        @test_throws ArgumentError LassoProblem(sparse([NaN 0.0; 0.0 1.0]),
                                                [1.0, 2.0])
        @test_throws ArgumentError KKTStop(rtol=-1)
        @test_throws ArgumentError lasso_kkt_residual(signed, zeros(3), -0.1)
        @test lasso_kkt_residual(signed, zeros(3), 2.0) == 0
        @test lasso_kkt_residual(positive, zeros(3), 2.0) == 0
        @test lasso_kkt_residual(signed, [1.8, -0.8, 0.3], 0.2) < 1e-14
        @test lasso_kkt_residual(positive, [1.8, 0.0, 0.3], 0.2) < 1e-14
        @test objective_value(positive, [1.0, -1.0, 0.0], 0.2) == Inf
    end

    @testset "FISTA on analytical cases" begin
        A = Matrix{Float64}(I, 3, 3)
        b = [2.0, -1.0, 0.5]
        stop = KKTStop(atol=1e-12, rtol=1e-8, check_every=1,
                       maxiter=1_000)
        for (constraint, expected) in ((:signed, [1.8, -0.8, 0.3]),
                                       (:nonnegative, [1.8, 0.0, 0.3]))
            problem = LassoProblem(A, b; constraint)
            result = solve(problem, FISTA(step=:exact, stop=stop), 0.2)
            @test result.converged
            @test result.stop_reason == :kkt
            @test result.x ≈ expected atol=1e-10
            @test result.kkt_ratio < 1e-7
            @test solve(problem, FISTA(step=:exact, stop=stop);
                        lambda=0.2).x ≈ expected
        end
        zero_result = solve(LassoProblem(A, b), FISTA(step=:exact), 2.0)
        @test zero_result.converged && zero_result.iterations == 1
        @test all(iszero, zero_result.x)
        @test_throws ArgumentError FISTASolver(LassoProblem(A, b); step=0)
        @test_throws ArgumentError solve(LassoProblem(A, b), FISTA(), -1)
    end

    @testset "dual-gap default and explicit gradient-mapping stop" begin
        A = Matrix{Float64}(I, 3, 3)
        b = [2.0, -1.0, 0.5]
        stop = GradientMappingStop(atol=1e-12, rtol=1e-8,
                                   check_every=1, maxiter=100)
        for constraint in (:signed, :nonnegative)
            problem = LassoProblem(A, b; constraint)
            result = solve(problem, FISTA(step=:exact, stop=stop), 0.2)
            @test result.converged
            @test result.stop_reason == :gradient_mapping
            @test result.iterations == 2
            @test result.diagnostics.gradient_mapping_residual <=
                  stop.atol + stop.rtol *
                  result.diagnostics.gradient_mapping_scale
            @test result.kkt_ratio <= 1e-8
        end
        @test_throws ArgumentError GradientMappingStop(rtol=-1)
        @test_throws ArgumentError FISTA(stop=:unknown)
        constrained = ConstrainedLassoProblem(A, b, 1.0)
        @test solve(constrained, FISTA(stop=stop)).converged
        @test FISTA().stop isa DualGapStop
        @test FISTASolver(LassoProblem(A, b)).stop isa DualGapStop
        @test LassoFISTASolver(A, b).stop isa DualGapStop
        @test_throws ArgumentError GradientMappingStop(warn_optimality_ratio=-1)
        loose = GradientMappingStop(rtol=1.0, warn_optimality_ratio=0.1)
        bad = @test_logs (:warn, r"large final optimality") solve(
            LassoProblem(reshape([1.0], 1, 1), [2.0]),
            FISTA(step=0.25, stop=loose), 0.2)
        @test bad.converged && bad.kkt_ratio > 0.1
        # An inactive zero coefficient may have a huge positive gradient.
        # It cannot justify a loose stop on the coefficient still moving.
        skewed_b = [1.0, -100.0]
        skewed = LassoProblem(Matrix{Float64}(I, 2, 2), skewed_b;
                              constraint=:nonnegative)
        one_step = GradientMappingStop(rtol=1.0, maxiter=1,
                                       warn_optimality_ratio=1e6)
        early = solve(skewed, FISTA(step=0.1, stop=one_step), 0.2)
        @test early.converged
        @test early.diagnostics.gradient_mapping_scale ≈ 1.0
        strict_step = GradientMappingStop(rtol=0.01, maxiter=1,
                                          warn_optimality_ratio=1e6)
        @test !solve(skewed, FISTA(step=0.1, stop=strict_step), 0.2).converged
        signed_early = solve(LassoProblem(Matrix{Float64}(I, 2, 2), skewed_b),
                             FISTA(step=0.1, stop=one_step), 0.2)
        @test signed_early.diagnostics.gradient_mapping_scale ≈ 100.0
    end

    @testset "two-stage dual-gap certificate" begin
        A = Matrix{Float64}(I, 2, 2)
        b = [2.0, -1.0]
        for constraint in (:signed, :nonnegative)
            p = LassoProblem(A, b; constraint)
            result = solve(p, FISTA(step=:exact), 0.2)
            @test result.converged
            @test result.stop_reason == :dual_gap
            @test result.diagnostics.dual_gap <=
                  1e-4 * sum(abs2, b)
            @test lasso_dual_gap(p, result.x, 0.2) <=
                  1e-4 * sum(abs2, b)
        end
        @test_throws ArgumentError DualGapStop(tol=-1)
        @test_throws ArgumentError DualGapStop(check_every=0)
        p = LassoProblem(A, b)
        stalled = solve(p, FISTA(step=1e-8,
                                stop=DualGapStop(maxiter=1)),
                        0.2; x0=[1.0, 0.0])
        @test !stalled.converged
        @test stalled.stop_reason == :maxiter
        @test lasso_dual_gap(p, stalled.x, 0.2) >
              1e-4 * sum(abs2, b)
        zero_penalty = solve(p, FISTA(step=:exact), 0.0)
        @test zero_penalty.converged
        @test zero_penalty.stop_reason == :kkt_zero_penalty
    end

    @testset "sparse and Float32 FISTA" begin
        A = sparse(Matrix{Float64}(I, 3, 3))
        b = [2.0, -1.0, 0.5]
        stop = KKTStop(atol=1e-10, rtol=1e-6,
                       check_every=5, maxiter=10_000)
        sparse_result = solve(LassoProblem(A, b),
                              FISTA(step=:power, stop=stop), 0.2)
        @test sparse_result.converged
        @test sparse_result.x ≈ [1.8, -0.8, 0.3] atol=1e-5

        A32 = Matrix{Float32}(I, 3, 3)
        b32 = Float32[2, -1, 0.5]
        result32 = solve(LassoProblem(A32, b32),
                         FISTA(step=:exact,
                               stop=KKTStop(atol=1e-5, rtol=1e-3)), 0.2f0)
        @test result32.converged
        @test result32.x ≈ Float32[1.8, -0.8, 0.3] atol=1e-5
        @test solve(LassoProblem(zeros(2, 2), [1.0, 2.0]),
                    FISTA(step=:power), 0.2).iterations == 1
        @test solve(LassoProblem(Matrix{Float64}(I, 3, 3), b),
                    FISTA(step=:frobenius,
                          stop=KKTStop(atol=1e-12, rtol=1e-8,
                                       check_every=1)),
                    0.2).x ≈ [1.8, -0.8, 0.3] atol=1e-6
        large_identity = Matrix{Float64}(I, 501, 501)
        large_solver = FISTASolver(LassoProblem(large_identity,
                                                zeros(501)); step=:auto)
        @test large_solver.step ≈ 1 / 1.2 atol=1e-5
        @test_throws ArgumentError FISTASolver(LassoProblem(A, b);
                                                step=:unknown)
    end

    @testset "ADMM on analytical, wide, and premature-stop cases" begin
        A = Matrix{Float64}(I, 3, 3)
        b = [2.0, -1.0, 0.5]
        stop = KKTStop(atol=1e-12, rtol=1e-6, check_every=1,
                       maxiter=10_000)
        for (constraint, expected) in ((:signed, [1.8, -0.8, 0.3]),
                                       (:nonnegative, [1.8, 0.0, 0.3]))
            result = solve(LassoProblem(A, b; constraint),
                           ADMM(stop=stop, feasibility_atol=1e-10), 0.2)
            @test result.converged
            @test result.x ≈ expected atol=1e-5
            @test result.kkt_ratio <= 1e-6 + 1e-10
        end

        # Old SRT ADMM said "converged" after one step with KKT/lambda = 50.
        tiny = LassoProblem(reshape([1.0], 1, 1), [1e-6])
        @test ADMM().stop isa DualGapStop
        @test ADMMSolver(tiny).stop isa DualGapStop
        default_tiny = @test_logs (:warn, r"large final optimality") solve(
            tiny, ADMM(), 1e-8)
        @test default_tiny.converged
        @test default_tiny.stop_reason == :feasibility_and_dual_gap
        @test lasso_dual_gap(tiny, default_tiny.x, 1e-8) <=
              1e-4 * sum(abs2, tiny.b)
        accurate = solve(tiny, ADMM(stop=KKTStop()), 1e-8)
        @test accurate.converged
        @test accurate.iterations > 1
        @test accurate.kkt_ratio <= 1e-3 + 0.011
        @test accurate.x[1] ≈ 9.9e-7 atol=2e-8
        zero_penalty_admm = solve(tiny, ADMM(), 0.0)
        @test zero_penalty_admm.converged
        @test zero_penalty_admm.stop_reason == :feasibility_and_kkt_zero_penalty
        @test zero_penalty_admm.kkt_residual <= KKTStop().atol

        wide_A = [1.0 0.0 0.5 0.0; 0.0 1.0 0.0 0.5]
        wide = LassoProblem(wide_A, [1.0, -0.5])
        admm = solve(wide, ADMM(stop=stop), 0.1)
        fista = solve(wide, FISTA(step=:exact, stop=stop), 0.1)
        @test admm.converged
        @test wide_A * admm.x ≈ wide_A * fista.x atol=1e-5
        @test_throws ArgumentError ADMMSolver(wide; rho=0)
    end

    @testset "correlated-source identifiability" begin
        A = [1.0 1.0; 0.0 0.0]
        problem = LassoProblem(A, [1.0, 0.0]; constraint=:nonnegative)
        fista = solve(problem, FISTA(step=:exact), 0.1)
        admm = solve(problem, ADMM(), 0.1)
        @test fista.converged && admm.converged
        @test A * fista.x ≈ [0.9, 0.0] atol=1e-4
        @test A * admm.x ≈ A * fista.x atol=1e-4
        @test fista.kkt_ratio <= 1e-3 + 1e-9
        @test admm.kkt_ratio <= 1e-3 + 1e-9
        # No test asserts unique coefficients: these columns are identical.
    end

    @testset "paths, ownership, callbacks, incomplete solves" begin
        A = Matrix{Float64}(I, 2, 2)
        p = LassoProblem(A, [2.0, 1.0])
        solver = FISTASolver(p; step=:exact,
                             stop=KKTStop(check_every=1))
        first = solve!(solver, 1.0)
        saved = copy(first.x)
        second = solve!(solver, 0.2)
        @test first.x == saved
        @test first.x !== solver.x
        @test second.x !== solver.x
        @test first.x != second.x

        called = Int[]
        path = solve_path(p, FISTA(step=:exact), [2.0, 1.0, 0.2];
                          on_result=(i, _) -> push!(called, i))
        @test called == [1, 2, 3]
        @test length(path) == 3
        @test path[1].x == [0.0, 0.0]
        @test path[3].x ≈ [1.8, 0.8]
        @test solve_path!(FISTASolver(p; step=:exact), [2.0, 1.0];
                          store=false) === nothing
        @test_throws ArgumentError solve_path(p, FISTA(), Float64[])
        grid = lambda_grid(p)
        @test length(grid) == 30
        @test grid[1] == 2.0
        @test lambda_grid(LassoProblem(A, [-2.0, -1.0]))[1] == 2.0
        @test lambda_grid(LassoProblem(A, [0.0, 0.0])) == [0.0]
        @test grid[end] ≈ 0.0002
        @test all(diff(grid) .< 0)
        @test lambda_grid(p; min_ratio=1e-2, n_lambdas=3) ≈
              [2.0, 0.2, 0.02]
        @test lambda_grid(LassoProblem(A, [-2.0, -1.0];
                                       constraint=:nonnegative)) == [0.0]
        @test lambda_grid(LassoProblem(A, [-2.0, 1.0];
                                       constraint=:nonnegative))[1] == 1.0
        @test_throws ArgumentError lambda_grid(p; min_ratio=0)
        @test_throws ArgumentError lambda_grid(p; min_ratio=1)
        @test_throws ArgumentError lambda_grid(p; n_lambdas=1)
        @test length(solve_path(p, FISTA(step=:exact))) == 30
        @test length(solve_path!(FISTASolver(p; step=:exact);
                                 n_lambdas=3)) == 3
        @test length(solve_path(p, ADMM(); n_lambdas=3)) == 3

        # The first path point must respect a supplied initial iterate.
        exact_start = solve_path(p, FISTA(step=:exact), [0.2];
                                 x0=[1.8, 0.8])
        @test exact_start[1].converged
        @test exact_start[1].iterations == 1
        restarted = solve_path!(FISTASolver(p; step=:exact,
                                            x0=[1.8, 0.8]), [0.2];
                                restart=true)
        @test restarted[1].iterations > 0

        slow = LassoProblem([1.0 0.9; 0.9 1.0], [1.0, 0.0])
        incomplete = solve(slow, FISTA(step=0.01,
                           stop=KKTStop(maxiter=1, check_every=1,
                                        atol=0, rtol=1e-10)), 0.01)
        @test !incomplete.converged
        @test incomplete.stop_reason == :maxiter
        @test incomplete.kkt_ratio > 1e-10
    end

    @testset "ordered grouped cross-validation" begin
        groups = [:b, :a, :c, :a, :b, :c]
        rr = ordered_group_folds(groups, 2; order=[:a, :b, :c])
        @test rr.group_fold_ids == [1, 2, 1]
        @test rr.fold_ids == [2, 1, 1, 1, 2, 1]
        ct = ordered_group_folds(groups, 2; order=[:a, :b, :c],
                                 assignment=:contiguous)
        @test ct.group_fold_ids == [1, 1, 2]
        @test ct.fold_ids == [1, 1, 2, 1, 1, 2]
        @test ordered_group_folds((g for g in groups), 2).fold_ids ==
              ordered_group_folds(groups, 2).fold_ids
        @test_throws ArgumentError ordered_group_folds(groups, 4)
        @test_throws ArgumentError ordered_group_folds(groups, 2;
                                                        order=[:a, :b, :b])
        A = [1.0 0.0; 0.0 1.0;
             1.0 0.0; 0.0 1.0;
             1.0 0.0; 0.0 1.0]
        b = [2.0, -1.0, 2.0, -1.0, 2.0, -1.0]
        fold_ids = ordered_group_folds([1, 1, 2, 2, 3, 3], 3).fold_ids
        ratios = [1.0, 0.1, 0.01]
        p = LassoProblem(A, b)
        seen = Vector{Vector{Int}}()
        scorer = (pred, obs, rows) -> begin
            push!(seen, copy(rows))
            norm(pred - obs) / sqrt(length(obs))
        end
        cv = cross_validate(p, FISTA(step=:exact);
                            fold_ids, lambda_ratios=ratios, score=scorer)
        @test size(cv.fold_scores) == (3, 3)
        @test cv.fold_lambda_max ≈ fill(4.0, 3)
        @test cv.fold_lambdas[:, 1] ≈ 4.0 .* ratios
        @test cv.validation_counts == [2, 2, 2]
        @test all(cv.fold_converged)
        @test cv.best_index == argmin(cv.mean_scores)
        @test cv.one_se_index <= cv.best_index
        @test all(rows -> all(fold_ids[rows] .== fold_ids[first(rows)]), seen)
        @test cv.fold_scores[1, :] ≈ fill(sqrt(2.5), 3)
        positive = cross_validate(LassoProblem(A, b; constraint=:nonnegative),
                                  FISTA(step=:exact);
                                  fold_ids, lambda_ratios=ratios)
        @test all(positive.fold_converged)
        @test positive.fold_scores[end, 1] > cv.fold_scores[end, 1]
        varying_b = [2.0, -1.0, 3.0, -1.0, 4.0, -1.0]
        varying = cross_validate(LassoProblem(A, varying_b),
                                 FISTA(step=:exact);
                                 fold_ids, lambda_ratios=[1.0, 0.1])
        @test varying.fold_lambda_max ≈ [7.0, 6.0, 5.0]
        @test varying.fold_lambdas[2, :] ≈
              0.1 .* varying.fold_lambda_max
        cv_admm = cross_validate(p, ADMM();
                                 fold_ids, lambda_ratios=[1.0, 0.1])
        @test all(cv_admm.fold_converged)
        @test cv_admm.best_index == 2
        @test_throws DimensionMismatch cross_validate(p, FISTA();
                                                       fold_ids=[1, 2])
        @test_throws ArgumentError cross_validate(p, FISTA();
                                                  fold_ids=ones(Int, 6))
        @test_throws ArgumentError cross_validate(p, FISTA();
                                                  fold_ids, lambda_ratios=[0.1, 1.0])
        @test_throws ArgumentError cross_validate(p, FISTA();
                                                  fold_ids, lambda_ratios=[1.0],
                                                  score=(args...) -> NaN)
        unfinished = cross_validate(p, FISTA(step=1e-10,
                                              stop=DualGapStop(maxiter=1));
                                    fold_ids, lambda_ratios=[0.5])
        @test !any(unfinished.all_converged)
        @test isnothing(unfinished.best_index)
        @test isnothing(unfinished.one_se_index)
    end

    @testset "compatibility surface" begin
        A = Matrix{Float64}(I, 2, 2)
        b = [1.0, -1.0]
        @test lasso_fista(A, b, 0.2) ≈ [0.8, -0.8]
        @test nonnegative_lasso_fista(A, b, 0.2) ≈ [0.8, 0.0]
        @test lasso_admm(A, b, 0.2) ≈ [0.8, -0.8] atol=1e-4
        @test lasso_admm(A, b, 0.2; return_info=true).stop_reason ==
              :feasibility_and_dual_gap
        @test lasso_admm(A, b, 0.2; reltol=1e-6,
                         return_info=true).stop_reason ==
              :feasibility_and_kkt
        info = lasso_fista(A, b, 0.2; return_info=true)
        @test info.converged && info.kkt_residual < 1e-4
        @test info.stop_reason == :dual_gap
        @test info.kkt_ratio < 1e-4
        @test lasso_fista(A, b, 0.2;
                          reltol=1e-5, return_info=true).stop_reason ==
              :gradient_mapping
        legacy = LassoFISTASolver(A, b; step=1.0)
        r1 = lasso_fista(legacy, 0.5; return_info=true)
        saved = copy(r1.x)
        r2 = lasso_fista(legacy, 0.2; warm_start=true,
                         return_info=true)
        @test r1.x == saved
        @test r1.x != r2.x
        @test r1.stop_reason == :dual_gap
        legacy_mapped = lasso_fista(legacy, 0.1;
                                    reltol=1e-5, warm_start=true,
                                    return_info=true)
        @test legacy_mapped.stop_reason == :gradient_mapping
        @test lasso_fista(legacy, 0.05; warm_start=true,
                          return_info=true).stop_reason == :dual_gap
        legacy_admm = LassoADMMSolver(A, b)
        @test legacy_admm.stop isa DualGapStop
        @test lasso_admm(legacy_admm, 0.2) ≈ [0.8, -0.8] atol=1e-4
        @test nonnegative_lasso_admm(legacy_admm, 0.2;
                                     warm_start=true) ≈ [0.8, 0.0] atol=1e-4
    end
end

include("group_lasso.jl")
include("new_problem_families.jl")
include("laplacian_tests.jl")
include("tv_tests.jl")
