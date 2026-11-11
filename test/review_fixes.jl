using Test
using LinearAlgebra
using Random
using SparseArrays
using Prism

@testset "review fixes" begin
    @testset "TV converges at zero L1 penalty" begin
        rng = MersenneTwister(1)
        A = randn(rng, 40, 64)
        truth = zeros(8, 8); truth[3:6, 3:6] .= 1
        b = A * vec(truth) + 0.01 * randn(rng, 40)
        pure_tv = solve(TVProblem(A, b, (8, 8), 0.1), PDHG())  # lambda = 0
        @test pure_tv.converged
        @test pure_tv.stop_reason == :primal_dual
        @test pure_tv.stationarity_residual <= 1e-10 + 1e-3 * 0.1
        unregularized = solve(TVProblem(A, b, (8, 8), 0.0), PDHG(), 0.0)
        @test unregularized.converged
    end

    @testset "ADMM zero-penalty stop is relative to the data scale" begin
        rng = MersenneTwister(2)
        A = randn(rng, 60, 20)
        b = randn(rng, 60)
        for scale in (1e3, 1.0, 1e-3)
            result = solve(LassoProblem(scale .* A, scale .* b), ADMM(rho=scale^2), 0.0)
            @test result.converged
            @test result.stop_reason == :feasibility_and_kkt_zero_penalty
            @test result.x ≈ A \ b rtol=1e-3
        end
    end

    @testset "matrix-free block design can be displayed" begin
        rng = MersenneTwister(3)
        A = sprandn(rng, 20, 30, 0.3)
        B = sprandn(rng, 20, 4, 0.5)
        solver = Prism.BlockFISTASolver(BlockLassoRidgeProblem(A, B, randn(rng, 20)), 1.0)
        @test occursin("BlockDesign", sprint(show, solver.base.problem.A))
        @test !isempty(sprint(show, MIME"text/plain"(), solver))
    end

    @testset "L1-ball projection stays feasible for long vectors" begin
        rng = MersenneTwister(4)
        for T in (Float64, Float32), nonnegative in (false, true), trial in 1:10
            n = 400_000
            v = randn(rng, T, n) .* T(10)
            radius = T(n / 50)
            out = similar(v)
            Prism._project_l1_ball!(out, v, radius, nonnegative)
            problem = ConstrainedLassoProblem(ones(T, 1, n), ones(T, 1), radius;
                                              constraint=nonnegative ? :nonnegative : :signed)
            @test Prism._constrained_feasible(problem, out)
        end
    end

    @testset "cross-validation rejects the Huber penalty clearly" begin
        rng = MersenneTwister(5)
        problem = LassoProblem(randn(rng, 40, 10), randn(rng, 40); penalty=Huber(0.5))
        error = try
            cross_validate(problem, FISTA(); fold_ids=repeat(1:4, 10))
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("cross_validate needs the L1 penalty", sprint(showerror, error))
    end

    @testset "large dense TV and constrained ADMM avoid the full SVD" begin
        rng = MersenneTwister(6)
        A = randn(rng, 600, 900)            # 540,000 elements: above the exact-norm limit
        b = randn(rng, 600)
        automatic = initialize(TVProblem(A, b, (30, 30), 0.1), PDHG())
        exact = initialize(TVProblem(A, b, (30, 30), 0.1), PDHG(step=:exact))
        @test automatic.step <= exact.step              # the estimate is safe
        @test automatic.step >= 0.8 * exact.step        # and not needlessly small
        small = randn(rng, 20, 16)
        @test initialize(TVProblem(small, randn(rng, 20), (4, 4), 0.1), PDHG()).step ==
              initialize(TVProblem(small, randn(rng, 20), (4, 4), 0.1), PDHG(step=:exact)).step
        constrained = initialize(ConstrainedLassoProblem(A, b, 5.0), ADMM())
        @test 0.8 * inv(opnorm(A)^2) <= constrained.mapping_step <= inv(opnorm(A)^2)
    end
end
