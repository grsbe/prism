# Real SRT testdata, but ONLY Spasmi solvers. Run in an environment with
# Spasmi, JLD2, and CUDA; no SRT package is imported.
# Example:
# JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=32 julia --project=<benchmark-env> \
#     benchmark/srt_testdata_spasmi.jl
using LinearAlgebra
using SparseArrays
using Spasmi
using JLD2
using CUDA

const DATA_DIR = get(ENV, "SPASMI_SRT_DATA_DIR",
                     "/home/tobias-grasberger/sparse-reconstruction-toolkit/testdata")
const LAMBDA = 2.84e-7
const ITERATIONS = parse(Int, get(ENV, "SPASMI_SRT_ITERATIONS", "40"))
const REPEATS = parse(Int, get(ENV, "SPASMI_SRT_REPEATS", "3"))
const BLAS_THREADS = parse(Int, get(ENV, "SPASMI_SRT_BLAS_THREADS", "32"))
const GPU_DEVICE = parse(Int, get(ENV, "SPASMI_SRT_GPU_DEVICE", "1"))

load_single(name) =
    JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]

function build_timed(f; gpu=false)
    t0 = time_ns()
    value = f()
    gpu && CUDA.synchronize()
    return value, (time_ns() - t0) / 1e9
end

function benchmark_solve(solver; gpu=false)
    run() = solve!(solver, LAMBDA; warm_start=false)
    run()  # Compile and warm up outside timing.
    gpu && CUDA.synchronize()
    times = Float64[]
    result = nothing
    for _ in 1:REPEATS
        t0 = time_ns()
        result = run()
        gpu && CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e9)
    end
    sorted = sort(times)
    return result, sorted[(REPEATS + 1) ÷ 2], times
end

function report(label, setup, result, median_seconds, samples)
    println(join((label, setup, median_seconds, result.iterations,
                  result.converged, result.stop_reason,
                  result.residual_norm, result.kkt_ratio,
                  join(samples, ";")), ","))
end

function main()
    ITERATIONS > 0 || error("SPASMI_SRT_ITERATIONS must be positive")
    REPEATS > 0 && isodd(REPEATS) ||
        error("SPASMI_SRT_REPEATS must be positive and odd")
    BLAS.set_num_threads(BLAS_THREADS)
    CUDA.functional() || error("CUDA is not functional")
    CUDA.device!(GPU_DEVICE)
    CUDA.allowscalar(false)

    A = load_single("A.jld2")
    b = load_single("y_noisy_SNR30.jld2")
    dense_problem = LassoProblem(A, b; constraint=:nonnegative)
    fista_stop = GradientMappingStop(atol=0.0, rtol=0.0,
                                     check_every=ITERATIONS,
                                     maxiter=ITERATIONS,
                                     warn_optimality_ratio=1e12)
    admm_stop = KKTStop(atol=0.0, rtol=0.0,
                        check_every=ITERATIONS,
                        maxiter=ITERATIONS)
    cpu_dense, cpu_dense_setup = build_timed() do
        initialize(dense_problem, FISTA(step=:power, stop=fista_stop))
    end
    step = cpu_dense.step
    As, sparse_conversion = build_timed(() -> sparse(A))
    cpu_sparse, cpu_sparse_setup = build_timed() do
        initialize(LassoProblem(As, b; constraint=:nonnegative),
                   FISTA(step=step, stop=fista_stop))
    end
    Ag, dense_transfer = build_timed(() -> CuArray(A); gpu=true)
    bg, vector_transfer = build_timed(() -> CuArray(b); gpu=true)
    gpu_dense, gpu_dense_setup = build_timed(
        () -> initialize(LassoProblem(Ag, bg; constraint=:nonnegative),
                         FISTA(step=step, stop=fista_stop)); gpu=true)
    Acsr, csr_transfer = build_timed(
        () -> CUDA.CUSPARSE.CuSparseMatrixCSR(As); gpu=true)
    gpu_csr, gpu_csr_setup = build_timed(
        () -> initialize(LassoProblem(Acsr, bg; constraint=:nonnegative),
                         FISTA(step=step, stop=fista_stop)); gpu=true)
    cpu_admm, cpu_admm_setup = build_timed() do
        initialize(dense_problem, ADMM(rho=inv(step), stop=admm_stop))
    end

    println("dataset=", DATA_DIR)
    println("matrix=", size(A), ",eltype=", eltype(A),
            ",nonzeros=", nnz(As), ",density=", nnz(As) / length(A))
    println("device=", CUDA.name(CUDA.device()),
            ",blas_threads=", BLAS.get_num_threads(),
            ",lambda=", LAMBDA, ",step=", step, ",rho=", inv(step),
            ",iterations=", ITERATIONS, ",repeats=", REPEATS)
    println("conversions_seconds: cpu_dense_to_sparse=", sparse_conversion,
            ",cpu_dense_to_gpu_dense=", dense_transfer,
            ",cpu_sparse_to_gpu_csr=", csr_transfer,
            ",cpu_b_to_gpu_b=", vector_transfer)
    println("label,solver_setup_seconds,median_solve_seconds,iterations,converged,stop_reason,residual_norm,kkt_over_lambda,sample_seconds")
    results = Dict{String,Any}()
    for (label, solver, setup, is_gpu) in
        (("cpu_dense_fista", cpu_dense, cpu_dense_setup, false),
         ("cpu_sparse_fista", cpu_sparse, cpu_sparse_setup, false),
         ("gpu_dense_fista", gpu_dense, gpu_dense_setup, true),
         ("gpu_csr_fista", gpu_csr, gpu_csr_setup, true),
         ("cpu_dense_admm", cpu_admm, cpu_admm_setup, false))
        result, median_seconds, samples = benchmark_solve(solver; gpu=is_gpu)
        report(label, setup, result, median_seconds, samples)
        results[label] = result
    end
    x_ref = results["cpu_dense_fista"].x
    for label in ("cpu_sparse_fista", "gpu_dense_fista", "gpu_csr_fista",
                  "cpu_dense_admm")
        x = results[label].x
        x_cpu = x isa CuArray ? Array(x) : x
        println("relative_prediction_difference_vs_cpu_dense_fista,",
                label, ",", norm(A * (x_cpu - x_ref)) / norm(b))
    end
    println("These are fixed-iteration timing results, not convergence claims.")
    println("GPU ADMM was not timed: Spasmi ADMM currently requires CPU dense matrices.")
end

main()
