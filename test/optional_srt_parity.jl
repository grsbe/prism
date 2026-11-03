# Run from an environment containing both Prism and SparseReconstructionToolkit.
using Test
using LinearAlgebra
using Prism
import SparseReconstructionToolkit as SRT

@testset "SRT gradient-mapping stop parity" begin
    A = [1.0 0.3 0.0; 0.1 1.0 0.2; 0.2 0.1 1.0; 0.0 0.4 0.8]
    b = [1.3, -0.4, 0.8, 0.3]
    lambda = 0.1
    step = inv(opnorm(A)^2)
    stop = GradientMappingStop(atol=0.0, rtol=1e-5,
                               check_every=1, maxiter=20_000)
    # Nonnegative Prism intentionally changes the scale, so only signed
    # LASSO retains exact stopping-iteration parity with SRT.
    for (constraint, old_solve) in
        ((:signed, SRT.lasso_fista),)
        old = old_solve(A, b, lambda; step, maxiter=stop.maxiter,
                        optimality_abstol=stop.atol,
                        optimality_reltol=stop.rtol, return_info=true)
        current = solve(LassoProblem(A, b; constraint),
                        FISTA(step=step, stop=stop), lambda)
        @test current.converged == old.converged
        @test current.iterations == old.iterations
        @test current.x ≈ old.x atol=1e-12
        @test current.diagnostics.gradient_mapping_residual ≈
              old.optimality_residual atol=1e-12
    end
end

@testset "SRT objective parity for migrated families" begin
    A = [1.0 0.3 0.0; 0.1 1.0 0.2; 0.2 0.1 1.0; 0.0 0.4 0.8]
    b = [1.3, -0.4, 0.8, 0.3]
    stop = KKTStop(atol=1e-10, rtol=1e-7, check_every=1,
                   maxiter=30_000)
    for (constraint, old_fista) in
        ((:signed, SRT.constrained_lasso_fista),
         (:nonnegative, SRT.nonnegative_constrained_lasso_fista))
        problem = ConstrainedLassoProblem(A, b, 1.2; constraint)
        current = solve(problem, FISTA(step=:exact, stop=stop))
        legacy = old_fista(A, b, 1.2; maxiter=30_000,
                            optimality_reltol=1e-8)
        @test current.converged
        @test norm(A * current.x - b)^2 ≈ norm(A * legacy - b)^2 atol=1e-5
        @test sum(abs, legacy) <= 1.2 + 1e-6
    end

    B = [0.5 0.0; 0.2 0.4; 0.0 0.3; 0.1 0.2]
    for (constraint, old_fista) in
        ((:signed, SRT.lasso_ridge_fista),
         (:nonnegative, SRT.nonnegative_lasso_ridge_fista))
        problem = BlockLassoRidgeProblem(A, B, b; constraint)
        current = solve(problem, FISTA(step=:exact, stop=stop), 0.15, 0.8)
        legacy = old_fista(A, B, b, 0.15, 0.8; maxiter=30_000,
                            optimality_reltol=1e-8)
        legacy_x, legacy_z = legacy
        old_objective = 0.5 * norm(A * legacy_x + B * legacy_z - b)^2 +
                        0.15 * sum(abs, legacy_x) + 0.4 * sum(abs2, legacy_z)
        @test current.converged
        @test current.objective ≈ old_objective atol=1e-5
    end
end

@testset "SRT TV objective parity" begin
    A = Matrix{Float64}(I, 4, 4)
    b = [1.5, -0.3, 0.8, 0.1]
    stop = KKTStop(atol=1e-8, rtol=1e-5, check_every=10,
                   maxiter=30_000)
    for (constraint, old_solver) in
        ((:signed, SRT.l1_total_variation_pdhg),
         (:nonnegative, SRT.nonnegative_l1_total_variation_pdhg))
        problem = TVProblem(A, b, (2, 2), 0.03; constraint)
        current = solve(problem, PDHG(stop=stop), 0.12)
        legacy = old_solver(A, b, (2, 2), 0.12, 0.03;
                            abstol=1e-12, reltol=1e-10,
                            maxiter=30_000)
        old_objective = 0.5 * norm(A * legacy - b)^2 +
                        0.12 * sum(abs, legacy) +
                        0.03 * total_variation2d(legacy, (2, 2))
        @test current.converged
        @test current.objective ≈ old_objective atol=1e-5
    end
    anisotropic = TVProblem(A, b, (2, 2), 0.03; isotropic=false)
    current = solve(anisotropic, PDHG(stop=stop), 0.12)
    legacy = SRT.l1_total_variation_pdhg(A, b, (2, 2), 0.12, 0.03;
                                           isotropic=false, abstol=1e-12,
                                           reltol=1e-10, maxiter=30_000)
    old_objective = 0.5 * norm(A * legacy - b)^2 +
                    0.12 * sum(abs, legacy) +
                    0.03 * total_variation2d(legacy, (2, 2); isotropic=false)
    @test current.converged
    @test current.objective ≈ old_objective atol=1e-5
end

@testset "SRT Laplacian objective parity" begin
    A = Matrix{Float64}(I, 4, 4)
    b = [1.5, -0.3, 0.8, 0.1]
    stop = KKTStop(atol=1e-10, rtol=1e-7, check_every=1,
                   maxiter=30_000)
    for (constraint, old_fista) in
        ((:signed, SRT.laplacian_lasso_fista),
         (:nonnegative, SRT.nonnegative_laplacian_lasso_fista))
        problem = LaplacianLassoProblem(A, b, (2, 2), 0.03; constraint)
        current = solve(problem, FISTA(step=:exact, stop=stop), 0.12)
        legacy = old_fista(A, b, (2, 2), 0.12, 0.03;
                            abstol=1e-12, reltol=1e-10,
                            maxiter=30_000)
        old_objective = 0.5 * norm(A * legacy - b)^2 +
                        0.12 * sum(abs, legacy) +
                        0.015 * sum(abs2, laplacian2d(legacy, (2, 2)))
        @test current.converged
        @test current.objective ≈ old_objective atol=1e-5
    end
end
