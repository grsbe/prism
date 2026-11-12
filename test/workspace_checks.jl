using Random

# The FISTA stopping checks run in the solver workspace. They must return the
# same numbers as the public (allocating, validating) functions, and a check
# must not allocate arrays: on a GPU such temporaries are invisible to Julia's
# garbage collector and pile up over long warm-started sweeps.
@testset "workspace optimality checks" begin
    rng = MersenneTwister(11)
    m, nx, nz = 300, 40, 12
    A = randn(rng, m, nx)
    B = randn(rng, m, nz)
    truth = zeros(nx); truth[[3, 17, 30]] .= [1.5, 2.0, 0.8]
    b = A * truth .+ B * (0.3 .* rand(rng, nz)) .+ 0.05 .* randn(rng, m)
    stop = DualGapStop(tol=1e-6, maxiter=50_000)
    vector_bytes = sizeof(b)

    # Bytes allocated by one more check at the converged iterate, where the
    # change gate passes and the dual gap is evaluated.
    function check_bytes(solver, lambda)
        ws = Prism._fista_workspace(solver)
        step = Prism._fista_step(solver)
        tolerance = Prism._fista_gap_tolerance(solver, ws.stop)
        check() = Prism._fista_check(solver, lambda, ws.stop, step, tolerance)
        @test check().reason == :dual_gap
        return @allocated check()
    end

    @testset "LASSO $constraint, $(nameof(typeof(loss))) loss, $(nameof(typeof(penalty))) penalty" for
            constraint in (:signed, :nonnegative),
            loss in (LeastSquares(), Huber(0.5)),
            penalty in (L1(), Huber(0.4))
        problem = LassoProblem(A, b; constraint, loss, penalty)
        lambda = 2.0  # lambda_grid is undefined for the Huber penalty
        solver = initialize(problem, FISTA(stop=stop))
        result = solve!(solver, lambda)
        @test result.stop_reason == :dual_gap
        @test Prism._fista_gap(solver, lambda) ≈
              lasso_dual_gap(problem, result.x, lambda) rtol=1e-10 atol=1e-12
        @test result.diagnostics.dual_gap ≈
              lasso_dual_gap(problem, result.x, lambda) rtol=1e-10 atol=1e-12
        @test check_bytes(solver, lambda) < vector_bytes ÷ 4
    end

    @testset "block $(sparse_design ? "sparse" : "dense") $xc/$zc, $(nameof(typeof(loss))) loss" for
            sparse_design in (false, true),
            (xc, zc) in ((:signed, :signed), (:nonnegative, :nonnegative),
                         (:signed, :nonnegative), (:nonnegative, :signed)),
            loss in (LeastSquares(), Huber(0.5))
        problem = BlockLassoRidgeProblem(sparse_design ? sparse(A) : A,
                                         sparse_design ? sparse(B) : B, b;
                                         x_constraint=xc, z_constraint=zc, loss)
        lambda, ridge = 2.0, 0.7
        solver = initialize(problem, FISTA(stop=stop), ridge)
        result = solve!(solver, lambda)
        @test result.stop_reason == :dual_gap
        @test Prism._fista_gap(solver, lambda) ≈
              block_dual_gap(problem, result.x, result.z, lambda, ridge) rtol=1e-10 atol=1e-12
        @test Prism._fista_kkt(solver, lambda) ≈
              block_kkt_residual(problem, result.x, result.z, lambda, ridge) rtol=1e-8 atol=1e-12
        @test check_bytes(solver, lambda) < vector_bytes ÷ 4
        # The KKT stop uses the workspace residual too.
        kkt = solve(problem, FISTA(stop=KKTStop(atol=1e-8, rtol=1e-6)), lambda, ridge)
        @test kkt.converged
        @test kkt.x ≈ result.x atol=1e-4
        @test kkt.z ≈ result.z atol=1e-4
    end

    @testset "group $constraint" for constraint in (:signed, :nonnegative)
        groups = [collect(i:i+3) for i in 1:4:nx]
        problem = GroupLassoProblem(A, b, groups; constraint)
        lambda = 0.05 * lambda_grid(problem; n_lambdas=2)[1]
        solver = initialize(problem, FISTA(stop=stop))
        result = solve!(solver, lambda)
        @test result.stop_reason == :dual_gap
        @test Prism._fista_gap(solver, lambda) ≈
              group_lasso_dual_gap(problem, result.x, lambda) rtol=1e-10 atol=1e-12
    end

    # Allocation of a whole solve must not grow with the iteration count.
    problem = BlockLassoRidgeProblem(A, B, b; constraint=:nonnegative)
    function solve_bytes(maxiter)
        solver = initialize(problem, FISTA(stop=DualGapStop(tol=1e-14, maxiter=maxiter)), 0.7)
        solve!(solver, 2.0)
        return @allocated solve!(solver, 2.0; warm_start=false)
    end
    short, long = solve_bytes(50), solve_bytes(2_000)
    @test long <= short + vector_bytes
end

@testset "release_memory" begin
    calls = Ref(0)
    reclaim() = (calls[] += 1)
    @test release_memory() === nothing
    @test release_memory(reclaim; full=false) === nothing
    @test calls[] == 1
    @test_throws ArgumentError release_memory_callback(reclaim; every=0)

    A = [1.0 0.2; 0.1 1.0; 0.3 0.4]
    b = [1.0, -0.5, 0.2]
    lambdas = [0.4, 0.2, 0.1, 0.05, 0.02]
    calls[] = 0
    solver = initialize(LassoProblem(A, b), FISTA())
    plain = solve_path!(solver, lambdas; restart=true)
    cleaned = solve_path!(solver, lambdas; restart=true,
                          on_result=release_memory_callback(reclaim; every=2))
    @test calls[] == 2                       # after points 2 and 4
    @test all(p.x == c.x for (p, c) in zip(plain, cleaned))

    block = initialize(BlockLassoRidgeProblem(A, reshape([0.5, 0.1, 0.2], 3, 1), b),
                       FISTA(), 0.7)
    solve_path!(block, lambdas; store=false,
                on_result=release_memory_callback(reclaim))
    @test calls[] == 2 + length(lambdas)
end

@testset "block design storage" begin
    A, B, b = rand(6, 3), rand(6, 2), rand(6)
    combined(A, B) = initialize(BlockLassoRidgeProblem(A, B, b), FISTA(), 0.5).base.problem.A
    @test combined(A, B) isa Matrix                       # CPU dense: materialized
    @test combined(view(A, :, 1:2), B) isa Matrix
    @test combined(sparse(A), sparse(B)) isa Prism.BlockDesign
    @test combined(A, sparse(B)) isa Prism.BlockDesign
    @test Prism._host_dense(A) && Prism._host_dense(view(A, :, 1:2))
    @test !Prism._host_dense(sparse(A))
end
