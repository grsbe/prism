using Test
using LinearAlgebra
using SparseArrays
using Prism

@testset "disjoint group LASSO" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [3.0, 4.0, -2.0]
    groups = [[1, 2], [3]]
    expected = [(1 - sqrt(2) / 5) * 3,
                (1 - sqrt(2) / 5) * 4, -1.0]
    for constraint in (:signed, :nonnegative)
        problem = GroupLassoProblem(A, b, groups; constraint)
        answer = constraint == :signed ? expected : [expected[1], expected[2], 0.0]
        result = solve(problem, FISTA(step=:exact), 1.0)
        @test result.converged
        @test result.stop_reason == :dual_gap
        @test result.x ≈ answer atol=1e-12
        @test result.kkt_residual < 1e-12
        @test group_lasso_dual_gap(problem, result.x, 1.0) < 1e-10
        @test result.objective ≈ objective_value(problem, result.x, 1.0)
        @test solve(problem, FISTA(step=:exact,
                                   stop=KKTStop(atol=1e-12, rtol=1e-8,
                                                check_every=1)), 1.0).x ≈ answer
        grid = lambda_grid(problem)
        @test length(grid) == 30
        @test all(iszero, solve(problem, FISTA(step=:exact), grid[1]).x)
        solver = initialize(problem, FISTA(step=:exact))
        path = solve_path!(solver, [grid[1], 1.0])
        @test length(path) == 2
        @test path[2].x ≈ answer
        @test all(iszero, path[1].x) # Owned warm-start snapshot.
        @test length(solve_path(problem, FISTA(step=:exact);
                                n_lambdas=3)) == 3
    end

    p = GroupLassoProblem(A, b, [[1, 3], [2]];
                          weights=[2.0, 1.0], constraint=:nonnegative)
    @test p.weights == [2.0, 1.0]
    @test solve(p, FISTA(step=:exact), 1.0).x ≈ [1.0, 3.0, 0.0]
    @test lambda_grid(p)[1] ≈ 4.0
    @test group_lasso_kkt_residual(p, zeros(3), 4.0) == 0.0
    @test objective_value(p, [1.0, 2.0, -1.0], 1.0) == Inf

    @test_throws ArgumentError GroupLassoProblem(A, b, [])
    @test_throws ArgumentError GroupLassoProblem(A, b, [[1, 2], []])
    @test_throws ArgumentError GroupLassoProblem(A, b, [[1, 2], [2, 3]])
    @test_throws ArgumentError GroupLassoProblem(A, b, [[1, 2]])
    @test_throws ArgumentError GroupLassoProblem(A, b, [[1, 2], [4]])
    @test_throws DimensionMismatch GroupLassoProblem(A, b, groups;
                                                      weights=[1.0])
    @test_throws ArgumentError GroupLassoProblem(A, b, groups;
                                                 weights=[1.0, 0.0])

    for constraint in (:signed, :nonnegative)
        singleton = GroupLassoProblem(A, b, [[1], [2], [3]];
                                      constraint)
        ordinary = LassoProblem(A, b; constraint)
        @test solve(singleton, FISTA(step=:exact), 0.5).x ≈
              solve(ordinary, FISTA(step=:exact), 0.5).x
        @test solve(singleton, FISTA(step=:exact), 0.0).stop_reason ==
              :kkt_zero_penalty
    end

    sparse_problem = GroupLassoProblem(sparse(A), b, groups)
    @test solve(sparse_problem, FISTA(step=:power,
                                        stop=KKTStop(atol=1e-10,
                                                     rtol=1e-8,
                                                     check_every=1)),
                1.0).x ≈ expected atol=1e-5

    # The two group columns are identical: predictions, not coefficients,
    # are identified. The dual certificate should still hold.
    correlated = GroupLassoProblem([1.0 1.0; 0.0 0.0], [1.0, 0.0],
                                    [[1, 2]])
    r = solve(correlated, FISTA(step=:exact), 0.1)
    @test r.converged
    @test group_lasso_dual_gap(correlated, r.x, 0.1) <=
          1e-4 * sum(abs2, correlated.b)
end
