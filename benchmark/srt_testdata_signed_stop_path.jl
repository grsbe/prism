# Compare two signed SNR30 LASSO paths on the Blackwell GPU in CSR storage.
using CUDA
using JLD2
using LinearAlgebra
using SparseArrays
using Prism

const DATA_DIR = "/home/tobias-grasberger/sparse-reconstruction-toolkit/testdata"
const OUT_DIR = "/home/tobias-grasberger/prism/benchmark/results/srt_testdata_20261020/signed_lasso_blackwell_csr_stop_comparison"
const GPU_DEVICE = parse(Int, get(ENV, "PRISM_SRT_GPU_DEVICE", "1"))
const BASE_POINTS = 30
const BASE_MIN_RATIO = 1e-4
const MIN_RATIO = 1e-8
const RELATIVE_RMSE_GAIN = 0.01
const GAIN_STEPS = 3

load_single(name) = JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "index,lambda,lambda_ratio,iterations,converged,stop_reason,objective,residual_norm,rmse,residual_over_noise,relative_rmse_gain,relative_field_error,kkt_over_lambda,dual_gap_over_threshold,nonzeros,relative_coefficient_change,solve_seconds")
        for row in rows
            println(io, join(row, ","))
        end
    end
end

function run_path(label, solver, prescribed, lambda_max, truth, noise_norm;
                  adaptive=false)
    output = joinpath(OUT_DIR, label)
    mkpath(output)
    lambdas = Float64[]
    vectors = Vector{Float64}[]
    rows = Tuple[]
    gap_threshold = 1e-4 * sum(abs2, solver.problem.b)

    function append_solve!(lambda)
        start = time_ns()
        result = solve!(solver, lambda; warm_start=!isempty(lambdas))
        CUDA.synchronize()
        seconds = (time_ns() - start) / 1e9
        x = Array(result.x)
        rmse = result.residual_norm / sqrt(length(solver.problem.b))
        gain = isempty(rows) ? NaN : (rows[end][9] - rmse) / rows[end][9]
        change = isempty(vectors) ? NaN :
                 norm(x - vectors[end]) / max(norm(x), norm(vectors[end]), eps())
        gap_ratio = lasso_dual_gap(solver.problem, result.x, lambda) / gap_threshold
        push!(lambdas, lambda)
        push!(vectors, x)
        push!(rows, (length(lambdas), lambda, lambda / lambda_max,
                     result.iterations, result.converged, result.stop_reason,
                     result.objective, result.residual_norm, rmse,
                     result.residual_norm / noise_norm, gain,
                     norm(x - truth) / norm(truth), result.kkt_ratio,
                     gap_ratio, count(!iszero, x), change, seconds))
        JLD2.jldsave(joinpath(output, "coefficients.jld2");
                     lambdas, coefficients=hcat(vectors...), status="in_progress")
        write_csv(joinpath(output, "path_diagnostics.csv"), rows)
        println(label, " point=", length(lambdas), " lambda=", lambda,
                " iterations=", result.iterations,
                " converged=", result.converged,
                " RMSE=", rmse, " field_error=", rows[end][12],
                " gap/threshold=", gap_ratio, " solve_s=", seconds)
        flush(stdout)
        return result
    end

    status = "prescribed_grid"
    for lambda in prescribed
        result = append_solve!(lambda)
        if adaptive && !result.converged
            status = "solver_maxiter"
            break
        end
    end
    if adaptive && status != "solver_maxiter"
        status = "minimum_lambda_ratio"
        log_step = log(BASE_MIN_RATIO) / (BASE_POINTS - 1)
        n_extra = floor(Int, log(MIN_RATIO / BASE_MIN_RATIO) / log_step + 1e-4)
        for i in 1:n_extra
            lambda = lambda_max * BASE_MIN_RATIO * exp(i * log_step)
            result = append_solve!(lambda)
            if !result.converged
                status = "solver_maxiter"
                break
            end
            if all(row -> row[11] < RELATIVE_RMSE_GAIN,
                   rows[end-GAIN_STEPS+1:end])
                status = "relative_rmse_plateau"
                break
            end
        end
    end
    if !adaptive && any(row -> !row[5], rows)
        status = "prescribed_grid_with_maxiter"
    end
    JLD2.jldsave(joinpath(output, "coefficients.jld2");
                 lambdas, coefficients=hcat(vectors...), status)
    write_csv(joinpath(output, "path_diagnostics.csv"), rows)
    println("summary ", label, " status=", status,
            " points=", length(rows),
            " failed=", count(row -> !row[5], rows),
            " solve_seconds=", sum(row[17] for row in rows))
    flush(stdout)
    return lambdas, rows
end

function main()
    CUDA.functional() || error("CUDA is not functional")
    CUDA.device!(GPU_DEVICE)
    occursin("Blackwell", CUDA.name(CUDA.device())) ||
        error("CUDA device $GPU_DEVICE is not Blackwell: $(CUDA.name(CUDA.device()))")
    CUDA.allowscalar(false)
    A = load_single("A.jld2")
    b = load_single("y_noisy_SNR30.jld2")
    truth = vec(permutedims(load_single("target.jld2")))
    noise_norm = load_single("L2_norm_of_noise_SNR30.jld2")
    Ag = CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(A))
    bg = CuArray(b)
    problem = LassoProblem(Ag, bg; constraint=:signed)
    lambda_max = maximum(abs, adjoint(Ag) * bg)
    base_grid = lambda_max .* collect(exp.(range(0, log(BASE_MIN_RATIO);
                                                 length=BASE_POINTS)))
    base_grid[1] = lambda_max
    base_grid[end] = lambda_max * BASE_MIN_RATIO
    precise = initialize(problem, FISTA(step=:power, stop=DualGapStop(tol=1e-4)))
    earlier = initialize(problem,
                         FISTA(step=precise.step,
                               stop=GradientMappingStop(atol=0, rtol=1e-3)))
    mkpath(OUT_DIR)
    println("device=", CUDA.name(CUDA.device()), " matrix=", size(A),
            " constraint=signed lambda_max=", lambda_max,
            " base_points=", BASE_POINTS, " base_min_ratio=", BASE_MIN_RATIO,
            " step=", precise.step, " noise_norm=", noise_norm,
            " precise_stop=", precise.stop, " earlier_stop=", earlier.stop)
    flush(stdout)
    grid, _ = run_path("precise_dual_gap", precise, base_grid, lambda_max,
                       truth, noise_norm; adaptive=true)
    run_path("earlier_gradient_mapping", earlier, grid, lambda_max,
             truth, noise_norm)
    println("done output=", OUT_DIR)
end

main()
