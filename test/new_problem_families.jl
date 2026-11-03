using Test
using LinearAlgebra
using Prism

@testset "constrained LASSO" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [2.0, -1.0, 0.5]
    tight = KKTStop(atol=1e-10, rtol=1e-7, check_every=1,
                    maxiter=20_000)
    for constraint in (:signed, :nonnegative)
        problem = ConstrainedLassoProblem(A, b, 1.0; constraint)
        for method in (FISTA(step=:exact, stop=tight),
                       ADMM(stop=tight, feasibility_atol=1e-10,
                            feasibility_rtol=1e-7))
            result = solve(problem, method)
            @test result.converged
            @test result.x ≈ [1.0, 0.0, 0.0] atol=1e-5
            @test sum(abs, result.x) <= 1.0 + 1e-8
            @test result.optimality_residual <=
                  tight.atol + tight.rtol * norm(b) + 1e-10
        end
    end

    for constraint in (:signed, :nonnegative)
        p = ConstrainedLassoProblem(A, b, 1.0; constraint)
        result = solve(p, FISTA(step=:exact))
        @test result.stop_reason == :dual_gap
        @test constrained_dual_gap(p, result.x) <=
              1e-4 * sum(abs2, b)
        admm_default = solve(p, ADMM())
        @test admm_default.converged
        @test admm_default.stop_reason == :feasibility_and_dual_gap
        @test constrained_dual_gap(p, admm_default.x) <=
              1e-4 * sum(abs2, b)
    end

    free = ConstrainedLassoProblem(A, b, 10.0)
    @test solve(free, FISTA(step=:exact)).x ≈ b atol=1e-6
    zero = ConstrainedLassoProblem(A, b, 0.0)
    @test solve(zero, FISTA(step=:exact)).x == zeros(3)
    @test solve(zero, ADMM()).x == zeros(3)
    @test_throws ArgumentError ConstrainedLassoProblem(A, b, -1)
    @test_throws ArgumentError initialize(ConstrainedLassoProblem(A, b, 1.0),
                                          FISTA(); x0=b)

    wide = ConstrainedLassoProblem([1.0 1.0], [2.0], 1.0;
                                    constraint=:nonnegative)
    wide_result = solve(wide, ADMM(stop=tight))
    @test wide_result.converged
    @test sum(wide_result.x) ≈ 1.0 atol=1e-6
    @test wide_result.optimality_residual < 1e-6

    problem = ConstrainedLassoProblem(A, b, 1.0)
    solver = initialize(problem, FISTA(step=:exact))
    first = solve!(solver)
    saved = copy(first.x)
    solve!(solver; warm_start=false)
    @test first.x == saved
    @test_throws ArgumentError constrained_optimality_residual(problem, b)

    incomplete = solve(problem,
                       FISTA(step=:frobenius,
                             stop=KKTStop(atol=0, rtol=0,
                                          check_every=1, maxiter=1)))
    @test !incomplete.converged
    @test incomplete.stop_reason == :maxiter
end

@testset "sparse-plus-ridge block FISTA" begin
    A = Matrix{Float64}(I, 2, 2)
    b = [2.0, -1.0]
    tight = KKTStop(atol=1e-10, rtol=1e-7,
                    check_every=1, maxiter=20_000)
    for (constraint, expected_x, expected_z) in
        ((:signed, [1.6, -0.6], [0.2, -0.2]),
         (:nonnegative, [1.6, 0.0], [0.2, 0.0]))
        problem = BlockLassoRidgeProblem(A, A, b; constraint)
        result = solve(problem, FISTA(step=:exact, stop=tight), 0.2, 1.0)
        @test result.converged
        @test result.x ≈ expected_x atol=1e-5
        @test result.z ≈ expected_z atol=1e-5
        @test result.kkt_residual <= tight.atol + tight.rtol * 0.2
        @test block_kkt_residual(problem, result.x, result.z, 0.2, 1.0) ≈
              result.kkt_residual
    end

    for (xc, zc, expected_x, expected_z) in
        ((:nonnegative, :signed, [1.6, 0.0], [0.2, -0.5]),
         (:signed, :nonnegative, [1.6, -0.8], [0.2, 0.0]))
        mixed = BlockLassoRidgeProblem(A, A, b;
                                       x_constraint=xc, z_constraint=zc)
        result = solve(mixed, FISTA(step=:exact, stop=tight), 0.2, 1.0)
        @test result.converged
        @test result.x ≈ expected_x atol=1e-5
        @test result.z ≈ expected_z atol=1e-5
        @test result.kkt_residual < 1e-6
        default_solver = initialize(mixed, FISTA(step=:exact), 1.0)
        @test default_solver.base.stop isa DualGapStop
        mapped = solve!(default_solver, 0.2)
        @test mapped.converged
        @test mapped.stop_reason == :dual_gap
        @test block_dual_gap(mixed, mapped.x, mapped.z, 0.2, 1.0) <=
              1e-4 * sum(abs2, b)
        @test mapped.kkt_residual ≈
              block_kkt_residual(mixed, mapped.x, mapped.z, 0.2, 1.0)
    end
    @test_throws ArgumentError BlockLassoRidgeProblem(A, A, b;
                                                       z_constraint=:wrong)
    @test_throws ArgumentError BlockLassoRidgeProblem(A, A, b;
                                                       x_constraint=:wrong)
    @test_throws ArgumentError initialize(
        BlockLassoRidgeProblem(A, A, b; z_constraint=:nonnegative),
        FISTA(), 1.0; z0=[0.0, -1.0])

    problem = BlockLassoRidgeProblem(A, A, b)
    solver = initialize(problem, FISTA(step=:exact), 1.0)
    @test solver.base.stop isa DualGapStop
    first = solve!(solver, 0.2)
    saved_x, saved_z = copy(first.x), copy(first.z)
    solve!(solver, 0.1)
    @test first.x == saved_x
    @test first.z == saved_z
    path = solve_path(problem, FISTA(step=:exact), [0.5, 0.2], 1.0)
    @test length(path) == 2
    @test all(result -> result.converged, path)
    @test path[1].x != path[2].x
    seen = Int[]
    @test isnothing(solve_path!(solver, [0.5, 0.2]; store=false,
                                restart=true, on_result=(i, _) -> push!(seen, i)))
    @test seen == [1, 2]
    @test_throws DimensionMismatch BlockLassoRidgeProblem(A, ones(3, 2), b)
    @test_throws ArgumentError initialize(problem, FISTA(), -1.0)
    @test_throws ArgumentError block_dual_gap(problem, first.x, first.z,
                                              0.2, 0.0)
    zero_ridge = solve(problem, FISTA(step=:exact), 0.2, 0.0)
    @test zero_ridge.converged
    @test zero_ridge.stop_reason == :kkt_dual_degenerate

    incomplete = solve(problem,
                       FISTA(step=:exact,
                             stop=KKTStop(atol=0, rtol=0,
                                          check_every=1, maxiter=1)),
                       0.2, 1.0)
    @test !incomplete.converged
    @test incomplete.stop_reason == :maxiter
end
