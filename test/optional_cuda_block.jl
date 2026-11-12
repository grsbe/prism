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

# Device temporaries are invisible to Julia's garbage collector, so a stopping
# check must not allocate device arrays: over a long warm-started sweep they
# fill the memory pool and force ever more expensive collections.
@testset "CUDA block FISTA checks allocate no device arrays" begin
    rng = MersenneTwister(5)
    m, nx, nz = 4_000, 300, 60
    A, B = randn(rng, m, nx), randn(rng, m, nz)
    b = A * (rand(rng, nx) .* (rand(rng, nx) .< 0.1)) .+ 0.1 .* randn(rng, m)
    lambda, ridge = 20.0, 0.7
    for sparse_design in (false, true), constraint in (:signed, :nonnegative)
        to_gpu(M) = sparse_design ?
                    CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(M)) : CuArray(M)
        problem = BlockLassoRidgeProblem(to_gpu(A), to_gpu(B), CuArray(b);
                                         constraint)
        solver = initialize(problem, FISTA(stop=DualGapStop(tol=1e-6,
                                                            maxiter=50_000)),
                            ridge)
        # No design copy on the device: dense and CSR are both matrix-free.
        @test solver.base.problem.A isa Prism.BlockDesign
        sparse_design || @test (CUDA.@allocated initialize(
            problem, FISTA(stop=DualGapStop(maxiter=10)), ridge)) <
                                (sizeof(A) + sizeof(B)) ÷ 10
        result = solve!(solver, lambda)
        @test result.stop_reason == :dual_gap
        @test Prism._fista_gap(solver, lambda) ≈
              block_dual_gap(problem, result.x, result.z, lambda, ridge) rtol=1e-8
        @test Prism._fista_kkt(solver, lambda) ≈
              block_kkt_residual(problem, result.x, result.z, lambda, ridge) rtol=1e-6 atol=1e-9
        base, stop = solver.base, solver.base.stop
        tolerance = Prism._fista_gap_tolerance(solver, stop)
        check() = Prism._fista_check(solver, lambda, stop, solver.step, tolerance)
        @test check().reason == :dual_gap   # the gate passes, so the gap runs
        # Baseline: what the two matrix-vector products of one gradient
        # allocate inside CUDA (nothing for dense, a work buffer for CSR).
        gradient!() = Prism._fista_gradient!(solver, base.y)
        gradient!()
        matvec_bytes = CUDA.@allocated gradient!()
        # Beyond that a check keeps only reduction scalars on the device;
        # one temporary vector would be 32 kB.
        @test (CUDA.@allocated check()) <= matvec_bytes + 1024
        # Per iteration, a whole solve allocates no more than its gradient.
        function solve_bytes(maxiter)
            fresh = initialize(problem, FISTA(stop=DualGapStop(tol=1e-14,
                                                               maxiter=maxiter)),
                               ridge)
            return CUDA.@allocated solve!(fresh, lambda)
        end
        @test (solve_bytes(1_000) - solve_bytes(50)) / 950 <= matvec_bytes + 256
    end
end
