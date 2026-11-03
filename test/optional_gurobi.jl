# Run from an environment containing Prism, JuMP, and Gurobi with a license.
using Test
using LinearAlgebra
using Prism
using JuMP
using Gurobi

@test Base.get_extension(Prism, :PrismGurobiExt) !== nothing

@testset "optional Gurobi parity" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [2.0, -1.0, 0.5]
    for constraint in (:signed, :nonnegative)
        problem = LassoProblem(A, b; constraint)
        oracle = solve(problem, GurobiBackend(), 0.2)
        fista = solve(problem, FISTA(step=:exact), 0.2)
        admm = solve(problem, ADMM(), 0.2)
        @test oracle.converged
        @test oracle.kkt_ratio < 1e-6
        @test fista.x ≈ oracle.x atol=1e-4
        @test admm.x ≈ oracle.x atol=1e-4
    end

    # Duplicate columns make coefficients nonunique; compare the identifiable
    # prediction and objective instead of individual atom weights.
    A_duplicate = [1.0 1.0; 0.0 0.0]
    problem = LassoProblem(A_duplicate, [1.0, 0.0];
                           constraint=:nonnegative)
    oracle = solve(problem, GurobiBackend(), 0.1)
    fista = solve(problem, FISTA(step=:exact), 0.1)
    @test oracle.converged
    @test A_duplicate * oracle.x ≈ A_duplicate * fista.x atol=1e-4
    @test oracle.objective ≈ fista.objective atol=1e-5
end
