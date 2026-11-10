using Random

@testset "Huber loss" begin
    @test_throws ArgumentError Huber(0.0)
    @test_throws ArgumentError LassoProblem(ones(2, 1), ones(2); loss=:huber)

    # Identity design separates per coordinate: the solution is soft
    # thresholding while λ < δ, and zero once λ ≥ δ (|ψ| ≤ δ).
    A = Matrix{Float64}(I, 3, 3)
    b = [2.0, -1.0, 0.5]
    problem = LassoProblem(A, b; loss=Huber(1.0))
    @test lambda_grid(problem; n_lambdas=2)[1] == 1.0  # max|clamp(b)| = δ
    @test solve(problem, FISTA(), 0.2).x ≈ [1.8, -0.8, 0.3] atol=1e-6
    @test all(iszero, solve(problem, FISTA(), 1.0).x)
    @test_throws ArgumentError ADMMSolver(problem)

    # Tall problem with gross outliers in b.
    rng = MersenneTwister(7)
    m, n = 60, 12
    A = randn(rng, m, n)
    truth = zeros(n); truth[[2, 5, 9]] .= [1.5, 2.0, 0.8]
    b = A * truth .+ 0.05 .* randn(rng, m)
    b[1:4] .+= 25.0
    huber, ls = Huber(0.2), LeastSquares()
    tight = DualGapStop(tol=1e-10, maxiter=200_000)

    @testset "LASSO $constraint" for constraint in (:signed, :nonnegative)
        problem = LassoProblem(A, b; constraint, loss=huber)
        lambda = 0.1 * lambda_grid(problem; n_lambdas=2)[1]
        result = solve(problem, FISTA(), lambda)
        @test result.converged && result.stop_reason == :dual_gap
        @test result.kkt_ratio < 1e-2
        reference = solve(problem, FISTA(stop=tight), lambda)
        # The gap certifies suboptimality, including at a perturbed point.
        optimum = objective_value(problem, reference.x, lambda)
        @test 0 <= result.objective - optimum <= result.diagnostics.dual_gap
        perturbed = abs.(reference.x .+ 0.1 .* randn(rng, n))
        @test objective_value(problem, perturbed, lambda) - optimum <=
              lasso_dual_gap(problem, perturbed, lambda)
        @test lasso_dual_gap(problem, reference.x, lambda) < 1e-6
        # Robustness: Huber recovers the sources far better than least squares.
        ls_fit = solve(LassoProblem(A, b; constraint, loss=ls), FISTA(), lambda)
        @test norm(reference.x - truth) < 0.25 * norm(ls_fit.x - truth)
    end

    @testset "large delta reproduces least squares" begin
        lambda = 1.0
        wide = solve(LassoProblem(A, b; loss=Huber(1e6)), FISTA(stop=tight), lambda)
        plain = solve(LassoProblem(A, b), FISTA(stop=tight), lambda)
        @test wide.x ≈ plain.x atol=1e-6
        @test wide.objective ≈ plain.objective rtol=1e-8
    end

    @testset "group LASSO" begin
        groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        problem = GroupLassoProblem(A, b, groups; loss=huber,
                                    constraint=:nonnegative)
        lambda_max = lambda_grid(problem; n_lambdas=2)[1]
        @test all(iszero, solve(problem, FISTA(), lambda_max).x)
        lambda = 0.1 * lambda_max
        result = solve(problem, FISTA(), lambda)
        @test result.converged && result.stop_reason == :dual_gap
        reference = solve(problem, FISTA(stop=tight), lambda)
        optimum = objective_value(problem, reference.x, lambda)
        @test 0 <= result.objective - optimum <= result.diagnostics.dual_gap
        @test group_lasso_dual_gap(problem, reference.x, lambda) < 1e-6
        @test group_lasso_kkt_residual(problem, reference.x, lambda) < 1e-4
    end

    @testset "block LASSO-ridge" begin
        B = randn(rng, m, 3)
        problem = BlockLassoRidgeProblem(A, B, b; loss=huber,
                                         z_constraint=:nonnegative)
        lambda, ridge = 0.5, 0.3
        result = solve(problem, FISTA(), lambda, ridge)
        @test result.converged && result.stop_reason == :dual_gap
        reference = solve(problem, FISTA(stop=tight), lambda, ridge)
        @test 0 <= result.objective - reference.objective <=
              result.diagnostics.dual_gap
        @test block_dual_gap(problem, reference.x, reference.z,
                             lambda, ridge) < 1e-6
        @test reference.kkt_ratio < 1e-4
    end
end

@testset "Huber penalty" begin
    # Identity design: x = prox_{λρ_δ}(b), quadratic or linear branch.
    I3 = Matrix{Float64}(I, 4, 4)
    b = [2.0, -1.0, 0.5, -3.0]
    for (constraint, expected) in ((:signed, [1.5, -1 / 1.5, 0.5 / 1.5, -2.5]),
                                   (:nonnegative, [1.5, 0.0, 0.5 / 1.5, 0.0]))
        problem = LassoProblem(I3, b; constraint, penalty=Huber(1.0))
        result = solve(problem, FISTA(), 0.5)
        @test result.converged && result.stop_reason == :dual_gap
        @test result.x ≈ expected atol=1e-6
    end
    problem = LassoProblem(I3, b; penalty=Huber(1.0))
    @test_throws ArgumentError lambda_grid(problem)
    @test_throws ArgumentError ADMMSolver(problem)

    rng = MersenneTwister(11)
    m, n = 60, 12
    A = randn(rng, m, n)
    truth = zeros(n); truth[[2, 5, 9]] .= [1.5, 2.0, 0.8]
    b = A * truth .+ 0.05 .* randn(rng, m)
    b[1:4] .+= 25.0
    tight = DualGapStop(tol=1e-10, maxiter=200_000)
    lambda = 2.0

    @testset "$constraint, $(nameof(typeof(loss))) loss" for
            constraint in (:signed, :nonnegative),
            loss in (LeastSquares(), Huber(0.2))
        problem = LassoProblem(A, b; constraint, loss, penalty=Huber(0.5))
        result = solve(problem, FISTA(), lambda)
        @test result.converged && result.stop_reason == :dual_gap
        reference = solve(problem, FISTA(stop=tight), lambda)
        optimum = objective_value(problem, reference.x, lambda)
        @test 0 <= result.objective - optimum <= result.diagnostics.dual_gap
        perturbed = abs.(reference.x .+ 0.1 .* randn(rng, n))
        @test objective_value(problem, perturbed, lambda) - optimum <=
              lasso_dual_gap(problem, perturbed, lambda)
        @test lasso_dual_gap(problem, reference.x, lambda) < 1e-6
        @test reference.kkt_ratio < 1e-4
    end

    @testset "small delta approaches LASSO" begin
        # (λ/δ)ρ_δ(x) = λ|x| - λδ/2 for |x| > δ, so solutions agree to O(δ).
        delta = 1e-4
        huber = solve(LassoProblem(A, b; penalty=Huber(delta)),
                      FISTA(stop=tight), lambda / delta)
        lasso = solve(LassoProblem(A, b), FISTA(stop=tight), lambda)
        @test huber.x ≈ lasso.x atol=1e-3
    end

    @testset "block LASSO-ridge" begin
        B = randn(rng, m, 3)
        problem = BlockLassoRidgeProblem(A, B, b; loss=Huber(0.2),
                                         penalty=Huber(0.5),
                                         x_constraint=:nonnegative)
        result = solve(problem, FISTA(), lambda, 0.3)
        @test result.converged && result.stop_reason == :dual_gap
        reference = solve(problem, FISTA(stop=tight), lambda, 0.3)
        @test 0 <= result.objective - reference.objective <=
              result.diagnostics.dual_gap
        @test block_dual_gap(problem, reference.x, reference.z,
                             lambda, 0.3) < 1e-6
        @test reference.kkt_ratio < 1e-4
    end
end
