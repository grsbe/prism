# Replay the saved nonnegative LASSO lambda grid with the corrected mapping scale.
using CUDA
using JLD2
using LinearAlgebra
using SparseArrays
using Prism

const ROOT = "/home/tobias-grasberger/prism/benchmark/results/srt_testdata_20261020"
const OUT = joinpath(ROOT, "nonnegative_mapping_scale_corrected")
const DATA = "/home/tobias-grasberger/sparse-reconstruction-toolkit/testdata"
const DEVICE = parse(Int, get(ENV, "PRISM_SRT_GPU_DEVICE", "1"))

load_single(name) = JLD2.load(joinpath(DATA, name))["single_stored_object"]

function main()
    CUDA.functional() || error("CUDA unavailable")
    CUDA.device!(DEVICE)
    occursin("Blackwell", CUDA.name(CUDA.device())) || error("Select Blackwell GPU")
    CUDA.allowscalar(false)
    old = JLD2.load(joinpath(ROOT, "lasso_paths/gpu_blackwell_csr/coefficients.jld2"))
    lambdas = old["lambdas"]
    old_coefficients = old["coefficients"]
    dual = JLD2.load(joinpath(ROOT, "lasso_paths_dual_gap_1e4/gpu_blackwell_csr/coefficients.jld2"))
    all(isapprox.(lambdas, dual["lambdas"]; rtol=1e-12)) || error("Lambda grids differ")
    dual_coefficients = dual["coefficients"]
    A = load_single("A.jld2")
    b = load_single("y_noisy_SNR30.jld2")
    truth = vec(permutedims(load_single("target.jld2")))
    noise_norm = load_single("L2_norm_of_noise_SNR30.jld2")
    problem = LassoProblem(CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(A)), CuArray(b);
                           constraint=:nonnegative)
    stop = GradientMappingStop(atol=0, rtol=1e-3, maxiter=10_000)
    solver = initialize(problem, FISTA(step=127.87471066504979, stop=stop))
    mkpath(OUT)
    vectors = Vector{Float64}[]
    rows = Tuple[]
    threshold = 1e-4 * sum(abs2, b)
    open(joinpath(OUT, "path_diagnostics.csv"), "w") do io
        println(io, "index,lambda,lambda_ratio,iterations,converged,rmse,residual_over_noise,relative_field_error,kkt_over_lambda,dual_gap_over_threshold,nonzeros,relative_difference_from_old,relative_difference_from_dual,solve_seconds")
        for (i, lambda) in enumerate(lambdas)
            start = time_ns()
            result = solve!(solver, lambda; warm_start=i > 1)
            CUDA.synchronize()
            seconds = (time_ns() - start) / 1e9
            x = Array(result.x)
            push!(vectors, x)
            gap = lasso_dual_gap(problem, result.x, lambda)
            relative_difference(reference) = norm(x - reference) /
                max(norm(x), norm(reference), eps(Float64))
            row = (i, lambda, lambda / lambdas[1], result.iterations,
                   result.converged, result.residual_norm / sqrt(length(b)),
                   result.residual_norm / noise_norm,
                   norm(x - truth) / norm(truth), result.kkt_ratio,
                   gap / threshold, count(!iszero, x),
                   relative_difference(old_coefficients[:, i]),
                   relative_difference(dual_coefficients[:, i]), seconds)
            push!(rows, row)
            println(io, join(row, ","))
            flush(io)
            println("point=", i, " lambda=", lambda, " iterations=", result.iterations,
                    " converged=", result.converged, " KKT/lambda=", result.kkt_ratio,
                    " difference_vs_old=", row[12])
            flush(stdout)
        end
    end
    JLD2.jldsave(joinpath(OUT, "coefficients.jld2"); lambdas,
                 coefficients=hcat(vectors...), status="complete")
    println("summary points=", length(rows), " converged=", count(r -> r[5], rows),
            " iterations=", sum(r[4] for r in rows),
            " solve_seconds=", sum(r[14] for r in rows))
end

main()
