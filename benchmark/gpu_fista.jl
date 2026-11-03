# Run in an environment containing both Prism and CUDA.jl.
# Example: PRISM_CUDA_DEVICE=1 julia --project=<cuda-env> benchmark/gpu_fista.jl
using LinearAlgebra
using Random
using Prism
using CUDA

CUDA.functional() || error("No functional CUDA device")
CUDA.device!(parse(Int, get(ENV, "PRISM_CUDA_DEVICE", "0")))
CUDA.allowscalar(false)
println("Device: ", CUDA.device())

rng = MersenneTwister(2026)
m = parse(Int, get(ENV, "PRISM_BENCH_ROWS", "512"))
n = parse(Int, get(ENV, "PRISM_BENCH_COLS", "4096"))
iterations = parse(Int, get(ENV, "PRISM_BENCH_ITERATIONS", "100"))
A = randn(rng, Float32, m, n)
b = randn(rng, Float32, m)
lambda = 0.1f0 * norm(adjoint(A) * b, Inf)
stop = KKTStop(atol=0, rtol=0, maxiter=iterations,
               check_every=iterations)

# Same step and data for both devices; construction/JIT/warmup are not timed.
cpu_problem = LassoProblem(A, b)
cpu = initialize(cpu_problem, FISTA(step=:power, stop=stop))
gpu_problem = LassoProblem(CuArray(A), CuArray(b))
gpu = initialize(gpu_problem, FISTA(step=cpu.step, stop=stop))
solve!(cpu, lambda; warm_start=false)
CUDA.@sync solve!(gpu, lambda; warm_start=false)

repeats = parse(Int, get(ENV, "PRISM_BENCH_REPEATS", "5"))
repeats > 0 && isodd(repeats) ||
    throw(ArgumentError("PRISM_BENCH_REPEATS must be positive and odd"))
cpu_times = sort!([@elapsed solve!(cpu, lambda; warm_start=false)
                   for _ in 1:repeats])
gpu_times = sort!([@elapsed CUDA.@sync solve!(gpu, lambda;
                                               warm_start=false)
                   for _ in 1:repeats])
cpu_time = cpu_times[(repeats + 1) ÷ 2]
gpu_time = gpu_times[(repeats + 1) ÷ 2]
cpu_result = solve!(cpu, lambda; warm_start=false)
gpu_result = CUDA.@sync solve!(gpu, lambda; warm_start=false)
prediction_difference = norm(A * cpu_result.x - A * Array(gpu_result.x)) / norm(b)
println("Matrix: ", m, " × ", n, "; iterations: ", iterations)
println("CPU BLAS threads: ", BLAS.get_num_threads())
println("Median of ", repeats, " CPU seconds: ", round(cpu_time; digits=4))
println("Median of ", repeats, " GPU seconds: ", round(gpu_time; digits=4))
println("Median speed ratio CPU/GPU: ", round(cpu_time / gpu_time; digits=2))
println("Relative prediction difference: ", prediction_difference)
println("CPU KKT/λ: ", cpu_result.kkt_ratio,
        "; GPU KKT/λ: ", gpu_result.kkt_ratio)
println("These are fixed-iteration timing results, not convergence claims.")
