using Test
using LinearAlgebra
using Prism

@testset "Laplacian-regularized FISTA" begin
    A = Matrix{Float64}(I, 4, 4)
    @test laplacian2d(ones(4), (2, 2)) == zeros(4)
    @test laplacian2d([1.0, 0.0, 0.0, 0.0], (2, 2)) ==
          [2.0, -1.0, -1.0, 0.0]
    @test_throws DimensionMismatch LaplacianLassoProblem(A, ones(4),
                                                           (2, 3), 0.5)
    @test_throws ArgumentError LaplacianLassoProblem(A, ones(4),
                                                     (2, 2), -0.1)

    tight = KKTStop(atol=1e-10, rtol=1e-7,
                    check_every=1, maxiter=30_000)
    for constraint in (:signed, :nonnegative)
        constant = LaplacianLassoProblem(A, fill(2.0, 4), (2, 2), 0.5;
                                         constraint)
        result = solve(constant, FISTA(step=:exact, stop=tight), 0.2)
        @test result.converged
        @test result.x ≈ fill(1.8, 4) atol=1e-5
        @test result.kkt_residual <= tight.atol + tight.rtol * 0.2

        varied = [2.0, -0.5, 1.0, 0.1]
        without_smoothing = LaplacianLassoProblem(A, varied, (2, 2), 0.0;
                                                    constraint)
        ordinary = LassoProblem(A, varied; constraint)
        @test solve(without_smoothing, FISTA(step=:exact), 0.2).x ≈
              solve(ordinary, FISTA(step=:exact), 0.2).x atol=1e-6
    end

    problem = LaplacianLassoProblem(A, [2.0, 0.0, 1.0, 0.0],
                                    (2, 2), 0.1)
    solver = initialize(problem, FISTA(step=:exact))
    @test solver.base.stop isa DualGapStop
    first = solve!(solver, 0.2)
    saved = copy(first.x)
    solve!(solver, 0.1)
    @test first.x == saved
    @test first.objective >= 0
    @test first.stop_reason == :dual_gap
    @test laplacian_dual_gap(problem, first.x, 0.2) <=
          1e-4 * sum(abs2, problem.b)
    @test first.kkt_residual ≈ laplacian_kkt_residual(problem, first.x, 0.2)

    incomplete = solve(problem,
                       FISTA(step=:exact,
                             stop=KKTStop(atol=0, rtol=0,
                                          check_every=1, maxiter=1)), 0.2)
    @test !incomplete.converged
    @test incomplete.stop_reason == :maxiter
end
