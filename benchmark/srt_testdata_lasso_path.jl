# Prism-only nonnegative LASSO path on SRT's SNR30 testdata.
# CPU defines the real-inversion lambda grid; GPU variants replay exactly that grid.
using LinearAlgebra
using SparseArrays
using Prism
using JLD2
using CUDA

const DATA_DIR = get(ENV, "PRISM_SRT_DATA_DIR",
                     "/home/tobias-grasberger/sparse-reconstruction-toolkit/testdata")
const OUT_DIR = get(ENV, "PRISM_PATH_OUT_DIR",
                    "/home/tobias-grasberger/prism/benchmark/results/srt_testdata_20261020/lasso_paths")
const BASE_POINTS = parse(Int, get(ENV, "PRISM_PATH_BASE_POINTS", "30"))
const BASE_MIN_RATIO = parse(Float64, get(ENV, "PRISM_PATH_BASE_MIN_RATIO", "1e-4"))
const MIN_RATIO = parse(Float64, get(ENV, "PRISM_PATH_MIN_RATIO", "1e-8"))
const RELATIVE_GAIN = parse(Float64, get(ENV, "PRISM_PATH_RELATIVE_GAIN", "0.01"))
const GAIN_STEPS = parse(Int, get(ENV, "PRISM_PATH_GAIN_STEPS", "3"))
const INCLUDE_CPU = get(ENV, "PRISM_PATH_INCLUDE_CPU", "1") == "1"
const GPU_DEVICE = parse(Int, get(ENV, "PRISM_SRT_GPU_DEVICE", "1"))
const BLAS_THREADS = parse(Int, get(ENV, "PRISM_SRT_BLAS_THREADS", "32"))

load_single(name) =
    JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]

function initial_grid(A, b)
    BASE_POINTS >= 2 || error("base points must be at least two")
    0 < MIN_RATIO < BASE_MIN_RATIO < 1 || error("invalid path ratio limits")
    0 < RELATIVE_GAIN < 1 && GAIN_STEPS > 0 || error("invalid RMSE plateau settings")
    lambda_max = max(0.0, maximum(adjoint(A) * b))
    lambda_max > 0 || error("lambda_max must be positive")
    grid = lambda_max .* collect(exp.(range(0, log(BASE_MIN_RATIO);
                                          length=BASE_POINTS)))
    grid[1] = lambda_max
    grid[end] = lambda_max * BASE_MIN_RATIO
    return grid, lambda_max
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "index,lambda,lambda_ratio,iterations,converged,stop_reason,objective,residual_norm,rmse,relative_rmse_gain,kkt_residual,kkt_over_lambda,dual_gap,dual_gap_over_threshold,gradient_mapping_residual,gradient_mapping_scale,l1_norm,nonzeros,relative_coefficient_change,solve_seconds")
        for row in rows
            println(io, join(row, ","))
        end
    end
end

function solve_variant(label, solver, prescribed, lambda_max; gpu=false, adaptive=false)
    output = joinpath(OUT_DIR, label)
    mkpath(output)
    lambdas = Float64[]
    vectors = Vector{Float64}[]
    rows = Tuple[]
    total_seconds = 0.0
    nobs = length(solver.problem.b)

    function append_solve!(lambda)
        t0 = time_ns()
        result = solve!(solver, lambda; warm_start=!isempty(lambdas))
        gpu && CUDA.synchronize()
        seconds = (time_ns() - t0) / 1e9
        total_seconds += seconds
        x = Vector{Float64}(gpu ? Array(result.x) : result.x)
        change = isempty(vectors) ? NaN :
                 norm(x - vectors[end]) /
                 max(norm(x), norm(vectors[end]), eps(Float64))
        rmse = result.residual_norm / sqrt(nobs)
        gain = isempty(rows) ? NaN :
               (rows[end][9] - rmse) / max(rows[end][9], eps(Float64))
        push!(lambdas, lambda)
        push!(vectors, x)
        gm = get(result.diagnostics, :gradient_mapping_residual, NaN)
        gap = lasso_dual_gap(solver.problem, result.x, lambda)
        gap_ratio = gap / (solver.stop.tol * sum(abs2, solver.problem.b))
        gs = get(result.diagnostics, :gradient_mapping_scale, NaN)
        push!(rows, (length(lambdas), lambda, lambda / lambda_max,
                     result.iterations, result.converged, result.stop_reason,
                     result.objective, result.residual_norm, rmse, gain,
                     result.kkt_residual, result.kkt_ratio, gap, gap_ratio, gm, gs,
                     sum(abs, x), count(!iszero, x), change, seconds))
        println(label, " ", length(lambdas), " lambda=", lambda,
                " iterations=", result.iterations,
                " converged=", result.converged,
                " RMSE=", rmse, " relative_RMSE_gain=", gain,
                " KKT/lambda=", result.kkt_ratio,
                " solve_s=", seconds)
        flush(stdout)
        coefficients = hcat(vectors...)
        JLD2.jldsave(joinpath(output, "coefficients.jld2");
                     lambdas, coefficients, status="in_progress")
        write_csv(joinpath(output, "path_diagnostics.csv"), rows)
        return result, change
    end

    for lambda in prescribed
        append_solve!(lambda)
    end
    status = adaptive ? "not_reached" : "prescribed_grid"
    if adaptive
        # Match SRE: continue the 30-point 1→1e-4 log grid with the same
        # logarithmic spacing, stopping after three <1% relative RMSE gains.
        log_step = log(BASE_MIN_RATIO) / (BASE_POINTS - 1)
        n_extra = floor(Int, log(MIN_RATIO / BASE_MIN_RATIO) / log_step + 1e-4)
        for i in 1:n_extra
            lambda = lambda_max * BASE_MIN_RATIO * exp(i * log_step)
            result, _ = append_solve!(lambda)
            if !result.converged
                status = "solver_maxiter"
                break
            end
            if length(rows) > GAIN_STEPS &&
               all(row -> row[10] < RELATIVE_GAIN, rows[end-GAIN_STEPS+1:end])
                status = "relative_rmse_plateau"
                break
            end
        end
        status == "not_reached" && (status = "minimum_lambda_ratio")
    end

    coefficients = hcat(vectors...)
    JLD2.jldsave(joinpath(output, "coefficients.jld2");
                 lambdas, coefficients, status)
    write_csv(joinpath(output, "path_diagnostics.csv"), rows)
    println("summary,", label, ",status,", status,
            ",points,", length(lambdas),
            ",total_solve_seconds,", total_seconds,
            ",failed_points,", count(row -> !row[5], rows),
            ",output,", output)
    flush(stdout)
    return lambdas, coefficients, status
end

function main()
    BLAS.set_num_threads(BLAS_THREADS)
    CUDA.functional() || error("CUDA is not functional")
    CUDA.device!(GPU_DEVICE)
    CUDA.allowscalar(false)
    A = load_single("A.jld2")
    b = load_single("y_noisy_SNR30.jld2")
    base_grid, lambda_max = initial_grid(A, b)
    mkpath(OUT_DIR)
    cpu_solver = initialize(LassoProblem(A, b; constraint=:nonnegative),
                            FISTA(step=:power))
    step = cpu_solver.step
    device_name = CUDA.name(CUDA.device())
    device_label = occursin("Blackwell", device_name) ? "blackwell" :
                   occursin("Ada", device_name) ? "ada" :
                   "device$(GPU_DEVICE)"
    println("matrix=", size(A), " device=", device_name,
            " lambda_max=", lambda_max,
            " base_points=", BASE_POINTS, " base_min_ratio=", BASE_MIN_RATIO,
            " min_ratio=", MIN_RATIO, " relative_gain=", RELATIVE_GAIN,
            " gain_steps=", GAIN_STEPS,
            " step=", step, " stop=", cpu_solver.stop,
            " BLAS_threads=", BLAS.get_num_threads())
    flush(stdout)

    lambdas, reference, cpu_status = if INCLUDE_CPU
        solve_variant("cpu_dense", cpu_solver, base_grid, lambda_max; adaptive=true)
    else
        reference_path = joinpath(OUT_DIR, "cpu_dense", "coefficients.jld2")
        isfile(reference_path) ||
            error("CPU reference path not found: $reference_path")
        data = JLD2.load(reference_path)
        data["lambdas"], data["coefficients"], data["status"]
    end
    println("shared_grid_from_cpu_status=", cpu_status,
            " points=", length(lambdas), " final_lambda=", lambdas[end])
    flush(stdout)

    Ag = CuArray(A)
    bg = CuArray(b)
    dense_solver = initialize(
        LassoProblem(Ag, bg; constraint=:nonnegative), FISTA(step=step))
    _, dense_coefficients, _ =
        solve_variant("gpu_$(device_label)_dense", dense_solver, lambdas, lambda_max;
                      gpu=true)
    Acsr = CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(A))
    csr_solver = initialize(
        LassoProblem(Acsr, bg; constraint=:nonnegative), FISTA(step=step))
    _, csr_coefficients, _ =
        solve_variant("gpu_$(device_label)_csr", csr_solver, lambdas, lambda_max;
                      gpu=true)
    for (label, coefficients) in
        (("gpu_$(device_label)_dense", dense_coefficients),
         ("gpu_$(device_label)_csr", csr_coefficients))
        max_prediction_difference =
            maximum(norm(A * (coefficients[:, i] - reference[:, i])) /
                    norm(b) for i in eachindex(lambdas))
        println("max_prediction_difference_vs_cpu,", label, ",",
                max_prediction_difference)
    end
    println("done output=", OUT_DIR)
end

main()
