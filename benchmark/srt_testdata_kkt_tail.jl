# Re-solve the low-lambda tail of the SRT SNR30 path on Blackwell GPU CSR.
# Each point is independently checkpointed and can be resumed.
using LinearAlgebra
using SparseArrays
using Prism
using JLD2
using CUDA

const ROOT = "/home/tobias-grasberger/prism/benchmark/results/srt_testdata_20261020"
const PATH_DIR = joinpath(ROOT, "lasso_paths")
const OUT_DIR = joinpath(ROOT, "lasso_path_kkt_tail")
const DATA_DIR = "/home/tobias-grasberger/sparse-reconstruction-toolkit/testdata"
const START_INDEX = parse(Int, get(ENV, "PRISM_KKT_START_INDEX", "21"))
const STOP = KKTStop(atol=0.0, rtol=0.01, check_every=25, maxiter=100_000)
const DEVICE = parse(Int, get(ENV, "PRISM_SRT_GPU_DEVICE", "1"))

function main()
    CUDA.functional() || error("CUDA is not functional")
    CUDA.device!(DEVICE)
    CUDA.allowscalar(false)
    mkpath(OUT_DIR)
    A = JLD2.load(joinpath(DATA_DIR, "A.jld2"))["single_stored_object"]
    b = JLD2.load(joinpath(DATA_DIR, "y_noisy_SNR30.jld2"))["single_stored_object"]
    truth = vec(permutedims(JLD2.load(joinpath(DATA_DIR, "target.jld2"))["single_stored_object"]))
    true_noise_norm = norm(A * truth - b)
    original = JLD2.load(joinpath(PATH_DIR, "cpu_dense", "coefficients.jld2"))
    lambdas = original["lambdas"]
    baseline = original["coefficients"]
    1 <= START_INDEX <= length(lambdas) || error("invalid start index")
    # Use the identical step as the original full path.
    step = initialize(LassoProblem(A, b; constraint=:nonnegative),
                      FISTA(step=:power)).step
    Ag = CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(A))
    bg = CuArray(b)
    solver = initialize(LassoProblem(Ag, bg; constraint=:nonnegative),
                        FISTA(step=step, stop=STOP);
                        x0=CuArray(baseline[:, START_INDEX]))
    println("GPU=", CUDA.name(CUDA.device()), " step=", step,
            " start_index=", START_INDEX, " end_index=", length(lambdas),
            " stop=", STOP)
    flush(stdout)
    rows = String[]
    for i in START_INDEX:length(lambdas)
        checkpoint = joinpath(OUT_DIR, "point_$(lpad(i, 3, '0')).jld2")
        if isfile(checkpoint)
            saved = JLD2.load(checkpoint)
            x = saved["x"]
            copyto!(solver.x, CuArray(x))
            push!(rows, saved["csv_row"])
            println("resume index=", i)
            flush(stdout)
            continue
        end
        lambda = lambdas[i]
        t0 = time_ns()
        result = solve!(solver, lambda; warm_start=true)
        CUDA.synchronize()
        seconds = (time_ns() - t0) / 1e9
        x = Array(result.x)
        x_old = baseline[:, i]
        x_change = norm(x - x_old) / max(norm(x), norm(x_old), eps())
        pred_change = norm(A * (x - x_old)) / norm(b)
        old_residual = norm(A * x_old - b)
        old_objective = 0.5 * old_residual^2 + lambda * sum(abs, x_old)
        rmse = result.residual_norm / sqrt(length(b))
        csv_row = join((i, lambda, result.iterations, result.converged,
                        result.stop_reason, result.kkt_ratio,
                        result.kkt_residual, rmse, result.objective,
                        old_objective, x_change, pred_change,
                        count(!iszero, x_old), count(!iszero, x),
                        seconds), ",")
        JLD2.jldsave(checkpoint; x, lambda, csv_row)
        push!(rows, csv_row)
        println("index=", i, " lambda=", lambda,
                " iterations=", result.iterations,
                " converged=", result.converged,
                " KKT/lambda=", result.kkt_ratio,
                " field_change_vs_default=", x_change,
                " prediction_change_vs_default=", pred_change,
                " seconds=", seconds)
        flush(stdout)
    end
    open(joinpath(OUT_DIR, "truth_diagnostics.csv"), "w") do io
        println(io, "index,lambda,default_relative_field_error,refined_relative_field_error,default_residual_norm,refined_residual_norm,true_noise_norm")
        for i in START_INDEX:length(lambdas)
            x = JLD2.load(joinpath(OUT_DIR, "point_$(lpad(i, 3, '0')).jld2"))["x"]
            old_x = baseline[:, i]
            println(io, join((i, lambdas[i],
                norm(old_x - truth) / norm(truth),
                norm(x - truth) / norm(truth),
                norm(A * old_x - b), norm(A * x - b),
                true_noise_norm), ","))
        end
    end
    open(joinpath(OUT_DIR, "comparison.csv"), "w") do io
        println(io, "index,lambda,iterations,converged,stop_reason,kkt_over_lambda,kkt_residual,rmse,objective,default_objective,relative_field_change_vs_default,relative_prediction_change_vs_default,default_nonzeros,refined_nonzeros,solve_seconds")
        for row in rows
            println(io, row)
        end
    end
end

main()
