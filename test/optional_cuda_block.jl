@testset "CUDA block LASSO-ridge FISTA" begin
    A = Float32[1 0 0.2; 0 1 0; 0.3 0 1; 0 0.4 0]
    B = Float32[1 0; 0 1; 0.2 0.1; 0 0.3]
    b = Float32[2, -1, 0.5, 0.1]
    lambda, ridge = 0.15f0, 0.7f0
    stop = KKTStop(atol=1e-5, rtol=1e-3, check_every=10, maxiter=20_000)
    for sparse_design in (false, true)
        cpu_A = sparse_design ? sparse(A) : A
        cpu_B = sparse_design ? sparse(B) : B
        gpu_A = sparse_design ? CUDA.CUSPARSE.CuSparseMatrixCSR(cpu_A) : CuArray(A)
        gpu_B = sparse_design ? CUDA.CUSPARSE.CuSparseMatrixCSR(cpu_B) : CuArray(B)
        for (xc, zc) in ((:signed, :signed), (:nonnegative, :nonnegative),
                         (:signed, :nonnegative), (:nonnegative, :signed))
            cpu_problem = BlockLassoRidgeProblem(cpu_A, cpu_B, b;
                              x_constraint=xc, z_constraint=zc)
            gpu_problem = BlockLassoRidgeProblem(gpu_A, gpu_B, CuArray(b);
                              x_constraint=xc, z_constraint=zc)
            cpu = solve(cpu_problem, FISTA(step=:auto, stop=stop), lambda, ridge)
            gpu = solve(gpu_problem, FISTA(step=:auto, stop=stop), lambda, ridge)
            @test cpu.converged && gpu.converged
            @test gpu.x isa CuArray && gpu.z isa CuArray
            @test norm(A * (Array(gpu.x) - cpu.x) +
                       B * (Array(gpu.z) - cpu.z)) < 2e-2
            @test gpu.objective ≈ cpu.objective atol=2e-3 rtol=2e-3
            @test gpu.kkt_ratio < 2e-3
        end
    end
    problem = BlockLassoRidgeProblem(CuArray(A), CuArray(B), CuArray(b))
    solver = initialize(problem, FISTA(step=:auto), ridge)
    path = solve_path!(solver, Float32[0.4, 0.2, 0.1])
    @test length(path) == 3
    @test all(r -> r.converged && r.x isa CuArray, path)
end
