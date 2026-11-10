@testset "CUDA Huber-loss FISTA parity" begin
    A = Float32[1 0 0.2; 0 1 0.1; 0.3 0 1; 0 0.4 0; 0.5 0.2 0.1]
    B = Float32[1 0; 0 1; 0.2 0.1; 0 0.3; 0.1 0.1]
    b = Float32[2, -1, 0.5, 0.1, 6]  # last row is an outlier
    loss = Huber(0.5)
    stop = DualGapStop(tol=1f-6, maxiter=50_000)
    for sparse_design in (false, true)
        cpu_A = sparse_design ? sparse(A) : A
        cpu_B = sparse_design ? sparse(B) : B
        device(M) = sparse_design ? CUDA.CUSPARSE.CuSparseMatrixCSR(M) : CuArray(M)
        for constraint in (:signed, :nonnegative)
            cpu = solve(LassoProblem(cpu_A, b; constraint, loss),
                        FISTA(stop=stop), 0.1f0)
            gpu = solve(LassoProblem(device(cpu_A), CuArray(b); constraint, loss),
                        FISTA(stop=stop), 0.1f0)
            @test gpu.converged && gpu.x isa CuArray
            @test Array(gpu.x) ≈ cpu.x atol=1e-3
            @test gpu.objective ≈ cpu.objective rtol=1e-4

            groups = [[1, 2], [3]]
            cpu = solve(GroupLassoProblem(cpu_A, b, groups; constraint, loss),
                        FISTA(stop=stop), 0.1f0)
            gpu = solve(GroupLassoProblem(device(cpu_A), CuArray(b), groups;
                                          constraint, loss),
                        FISTA(stop=stop), 0.1f0)
            @test gpu.converged && gpu.x isa CuArray
            @test Array(gpu.x) ≈ cpu.x atol=1e-3

            cpu = solve(BlockLassoRidgeProblem(cpu_A, cpu_B, b;
                                               constraint, loss),
                        FISTA(stop=stop), 0.1f0, 0.3f0)
            gpu = solve(BlockLassoRidgeProblem(device(cpu_A), device(cpu_B),
                                               CuArray(b); constraint, loss),
                        FISTA(stop=stop), 0.1f0, 0.3f0)
            @test gpu.converged && gpu.x isa CuArray && gpu.z isa CuArray
            @test Array(gpu.x) ≈ cpu.x atol=1e-3
            @test Array(gpu.z) ≈ cpu.z atol=1e-3
        end
    end
end

@testset "CUDA Huber-penalty FISTA parity" begin
    A = Float32[1 0 0.2; 0 1 0.1; 0.3 0 1; 0 0.4 0; 0.5 0.2 0.1]
    B = Float32[1 0; 0 1; 0.2 0.1; 0 0.3; 0.1 0.1]
    b = Float32[2, -1, 0.5, 0.1, 6]
    penalty = Huber(0.3)
    stop = DualGapStop(tol=1f-6, maxiter=50_000)
    for sparse_design in (false, true), loss in (LeastSquares(), Huber(0.5))
        cpu_A = sparse_design ? sparse(A) : A
        cpu_B = sparse_design ? sparse(B) : B
        device(M) = sparse_design ? CUDA.CUSPARSE.CuSparseMatrixCSR(M) : CuArray(M)
        for constraint in (:signed, :nonnegative)
            cpu = solve(LassoProblem(cpu_A, b; constraint, loss, penalty),
                        FISTA(stop=stop), 0.5f0)
            gpu = solve(LassoProblem(device(cpu_A), CuArray(b);
                                     constraint, loss, penalty),
                        FISTA(stop=stop), 0.5f0)
            @test gpu.converged && gpu.stop_reason == :dual_gap
            @test gpu.x isa CuArray
            @test Array(gpu.x) ≈ cpu.x atol=1e-3

            cpu = solve(BlockLassoRidgeProblem(cpu_A, cpu_B, b;
                                               constraint, loss, penalty),
                        FISTA(stop=stop), 0.5f0, 0.3f0)
            gpu = solve(BlockLassoRidgeProblem(device(cpu_A), device(cpu_B),
                                               CuArray(b); constraint, loss,
                                               penalty),
                        FISTA(stop=stop), 0.5f0, 0.3f0)
            @test gpu.converged && gpu.x isa CuArray
            @test Array(gpu.x) ≈ cpu.x atol=1e-3
            @test Array(gpu.z) ≈ cpu.z atol=1e-3
        end
    end
end
