# Run from a Julia environment containing both Prism and CUDA.jl.
# CUDA is optional and this file is intentionally not included in Pkg.test().
using Test
using LinearAlgebra
using Random
using SparseArrays
using Prism
using CUDA

CUDA.functional() || error("CUDA is installed but no functional device is available")
CUDA.device!(parse(Int, get(ENV, "PRISM_CUDA_DEVICE", "0")))
@info "Testing CUDA device" CUDA.device()
CUDA.allowscalar(false)

@testset "real CUDA FISTA parity" begin
    for T in (Float32, Float64)
        A = T[1 0 0.2; 0 1 0.1; 0.3 0 1; 0 0.4 0]
        b = T[2, -1, 0.5, 0.1]
        lambda = T(0.15)
        tolerance = T == Float32 ? 5e-3 : 5e-5
        stop = KKTStop(atol=tolerance * lambda / 10,
                       rtol=tolerance, check_every=10, maxiter=20_000)
        for constraint in (:signed, :nonnegative)
            cpu = solve(LassoProblem(A, b; constraint),
                        FISTA(step=:exact, stop=stop), lambda)
            gpu = solve(LassoProblem(CuArray(A), CuArray(b); constraint),
                        FISTA(step=:power, stop=stop), lambda)
            @test cpu.converged
            @test gpu.converged
            @test Array(gpu.x) ≈ cpu.x atol=T(5e-3) rtol=T(5e-3)
            @test gpu.kkt_ratio <= tolerance + tolerance / 10
            @test gpu.x isa CuArray
        end
    end
    solver = initialize(LassoProblem(CuArray(Float32[1 0; 0 1]),
                                     CuArray(Float32[1, -1])),
                        FISTA(step=:power))
    first = solve!(solver, 0.5f0)
    saved = Array(first.x)
    solve!(solver, 0.1f0)
    @test Array(first.x) == saved
end

@testset "CUDA CSR FISTA" begin
    A = Float64[1 0 0.2; 0 1 0; 0.3 0 1; 0 0.4 0]
    b = Float64[2, -1, 0.5, 0.1]
    lambda = 0.15
    stop = GradientMappingStop(atol=1e-10, rtol=1e-8,
                                check_every=1, maxiter=1000)
    cpu = solve(LassoProblem(sparse(A), b; constraint=:nonnegative),
                FISTA(step=:frobenius, stop=stop), lambda)
    csr = CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(A))
    gpu = solve(LassoProblem(csr, CuArray(b); constraint=:nonnegative),
                FISTA(step=:frobenius, stop=stop), lambda)
    @test cpu.converged && gpu.converged
    @test Array(gpu.x) ≈ cpu.x atol=1e-7
    @test gpu.kkt_residual ≈ cpu.kkt_residual atol=1e-8
end

@testset "correlated GPU columns" begin
    rng = MersenneTwister(2026)
    base = randn(rng, Float32, 80, 48)
    A = hcat(base, base) # Nonunique coefficients: compare predictions.
    b = A[:, 1] .+ 0.5f0 .* A[:, 3] .+ 0.01f0 .* randn(rng, Float32, 80)
    lambda = 0.1f0 * norm(adjoint(A) * b, Inf)
    stop = KKTStop(atol=1e-5, rtol=1e-3,
                   check_every=10, maxiter=10_000)
    cpu = solve(LassoProblem(A, b), FISTA(step=:exact, stop=stop), lambda)
    gpu = solve(LassoProblem(CuArray(A), CuArray(b)),
                FISTA(step=:power, stop=stop), lambda)
    @test cpu.converged && gpu.converged
    @test norm(A * Array(gpu.x) - A * cpu.x) / norm(b) < 1e-3
    @test gpu.kkt_residual ≈
          lasso_kkt_residual(LassoProblem(A, b), Array(gpu.x), lambda) rtol=1e-3 atol=2e-6
end

include("optional_cuda_group.jl")
include("optional_cuda_block.jl")
