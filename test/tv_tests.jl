using Test
using LinearAlgebra
using Prism

@testset "TV-PDHG primal-dual stopping" begin
    A = Matrix{Float64}(I, 4, 4)
    @test total_variation2d(ones(4), (2, 2)) == 0
    @test total_variation2d([1.0, 0.0, 0.0, 0.0], (2, 2)) ≈ sqrt(2)
    @test total_variation2d([1.0, 0.0, 0.0, 0.0], (2, 2);
                              isotropic=false) == 2
    @test_throws DimensionMismatch TVProblem(A, ones(4), (2, 3), 0.1)
    @test_throws ArgumentError TVProblem(A, ones(4), (2, 2), -0.1)
    @test_throws ArgumentError PDHG(theta=1.1)

    stop = KKTStop(atol=1e-8, rtol=1e-5,
                   check_every=10, maxiter=30_000)
    for constraint in (:signed, :nonnegative)
        constant = TVProblem(A, fill(2.0, 4), (2, 2), 0.5;
                              constraint)
        result = solve(constant, PDHG(stop=stop), 0.2)
        @test result.converged
        @test result.x ≈ fill(1.8, 4) atol=1e-4
        @test result.stationarity_residual <= stop.atol + stop.rtol * 0.2
        @test result.data_dual_residual <= stop.atol + stop.rtol * 2.0
        @test result.tv_complementarity_gap <= 1e-4
        @test result.dual_feasibility_residual <= 1e-10
    end

    problem = TVProblem(A, [1.5, -0.3, 0.8, 0.1], (2, 2), 0.03)
    solver = initialize(problem, PDHG(stop=stop))
    first = solve!(solver, 0.12)
    saved = copy(first.x)
    solve!(solver, 0.08)
    @test first.x == saved
    @test first.objective > 0

    anisotropic = TVProblem(A, [1.5, -0.3, 0.8, 0.1], (2, 2), 0.03;
                            isotropic=false)
    anisotropic_result = solve(anisotropic, PDHG(stop=stop), 0.12)
    @test anisotropic_result.converged
    @test anisotropic_result.tv_complementarity_gap < 1e-4

    incomplete = solve(problem,
                       PDHG(stop=KKTStop(atol=0, rtol=0,
                                         check_every=1, maxiter=1)), 0.12)
    @test !incomplete.converged
    @test incomplete.stop_reason == :maxiter
end
