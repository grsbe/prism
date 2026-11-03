# Time to Prism's DEFAULT GradientMappingStop on SRT's SNR30 testdata.
# Only Prism is used as a solver. CUDA and JLD2 are benchmark dependencies.
using LinearAlgebra
using SparseArrays
using Prism
using JLD2
using CUDA

const DATA_DIR = get(ENV, "PRISM_SRT_DATA_DIR",
                     "/home/tobias-grasberger/sparse-reconstruction-toolkit/testdata")
const LAMBDA = 2.84e-7
const GPU_DEVICE = parse(Int, get(ENV, "PRISM_SRT_GPU_DEVICE", "1"))
const BLAS_THREADS = parse(Int, get(ENV, "PRISM_SRT_BLAS_THREADS", "32"))
const REPEATS = parse(Int, get(ENV, "PRISM_SRT_REPEATS", "3"))

load_single(name) =
    JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]

function time_to_stop!(solver; gpu=false)
    full_stop = solver.stop
    solver.stop = GradientMappingStop(maxiter=1, warn_optimality_ratio=1e12)
    solve!(solver, LAMBDA; warm_start=false)  # JIT warmup, not timed.
    gpu && CUDA.synchronize()
    solver.stop = full_stop
    times = Float64[]
    results = Any[]
    for run in 1:REPEATS
        t0 = time_ns()
        result = solve!(solver, LAMBDA; warm_start=false)
        gpu && CUDA.synchronize()
        seconds = (time_ns() - t0) / 1e9
        push!(times, seconds)
        push!(results, result)
        println("run,", run, ",seconds,", seconds, ",iterations,",
                result.iterations, ",converged,", result.converged,
                ",stop_reason,", result.stop_reason, ",kkt_over_lambda,",
                result.kkt_ratio, ",residual_norm,", result.residual_norm,
                ",objective,", result.objective)
        flush(stdout)
    end
    return results[end], sort(times)[(REPEATS + 1) ÷ 2], times
end

function report(label, solver; gpu=false)
    println("case,", label)
    flush(stdout)
    result, median_seconds, samples = time_to_stop!(solver; gpu)
    println("summary,", label, ",median_seconds,", median_seconds,
            ",iterations,", result.iterations, ",converged,", result.converged,
            ",kkt_over_lambda,", result.kkt_ratio, ",samples,",
            join(samples, ";"))
    flush(stdout)
    return result
end

function main()
    REPEATS > 0 && isodd(REPEATS) ||
        error("PRISM_SRT_REPEATS must be positive and odd")
    BLAS.set_num_threads(BLAS_THREADS)
    CUDA.functional() || error("CUDA is not functional")
    CUDA.device!(GPU_DEVICE)
    CUDA.allowscalar(false)

    A = load_single("A.jld2")
    b = load_single("y_noisy_SNR30.jld2")
    cpu_problem = LassoProblem(A, b; constraint=:nonnegative)
    cpu = initialize(cpu_problem, FISTA(step=:power))
    step = cpu.step
    Ag = CuArray(A)
    bg = CuArray(b)
    gpu_dense = initialize(
        LassoProblem(Ag, bg; constraint=:nonnegative),
        FISTA(step=step))
    Acsr = CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(A))
    gpu_csr = initialize(
        LassoProblem(Acsr, bg; constraint=:nonnegative),
        FISTA(step=step))
    CUDA.synchronize()

    println("device,", CUDA.name(CUDA.device()), ",matrix,", size(A),
            ",lambda,", LAMBDA, ",step,", step, ",repeats,", REPEATS,
            ",blas_threads,", BLAS.get_num_threads(),
            ",stop,", cpu.stop)
    flush(stdout)

    cpu_result = report("cpu_dense_fista", cpu)
    dense_result = report("gpu_dense_fista", gpu_dense; gpu=true)
    csr_result = report("gpu_csr_fista", gpu_csr; gpu=true)
    for (label, result) in (("gpu_dense_fista", dense_result),
                            ("gpu_csr_fista", csr_result))
        x = Array(result.x)
        println("prediction_difference,", label, ",",
                norm(A * (x - cpu_result.x)) / norm(b))
    end
end

main()
