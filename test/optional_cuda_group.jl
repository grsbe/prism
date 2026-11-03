using Test
using LinearAlgebra
using SparseArrays
using Spasmi
using CUDA

CUDA.functional() || error("CUDA is not functional")
CUDA.device!(parse(Int, get(ENV, "SPASMI_CUDA_DEVICE", "0")))
CUDA.allowscalar(false)

@testset "CUDA group LASSO, noncontiguous groups" begin
    for T in (Float32, Float64)
        A = Matrix{T}(I, 4, 4)
        b = T[3, 4, -2, 0.5]
        groups = [[1, 3], [2, 4]]
        for constraint in (:signed, :nonnegative)
            cpu = solve(GroupLassoProblem(A, b, groups; constraint),
                        FISTA(step=:frobenius), T(0.5))
            for device_A in (CuArray(A),
                             CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(A)))
                problem = GroupLassoProblem(device_A, CuArray(b), groups;
                                            constraint)
                result = solve(problem, FISTA(step=:frobenius), T(0.5))
                @test result.converged
                @test result.x isa CuArray
                @test Array(result.x) ≈ cpu.x atol=T(2e-3)
                @test group_lasso_dual_gap(problem, result.x, T(0.5)) <=
                      T(1e-4) * sum(abs2, b) + T(1e-6)
            end
        end
    end
    A = CuArray(Matrix{Float32}(I, 2, 2))
    b = CuArray(Float32[2, -1])
    problem = GroupLassoProblem(A, b, [[1], [2]];
                                constraint=:nonnegative)
    path = solve_path!(initialize(problem, FISTA(step=:frobenius));
                       n_lambdas=3, min_ratio=0.1)
    @test length(path) == 3
    @test all(iszero, Array(path[1].x))
    @test path[end].x isa CuArray
    @test all(r -> r.converged, path)
end
