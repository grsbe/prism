# Calculate identical known-target diagnostics for three saved Blackwell paths.
using JLD2
using LinearAlgebra

const ROOT = "/home/tobias-grasberger/prism/benchmark/results/srt_testdata_20261020"
const OUT = joinpath(ROOT, "nonnegative_mapping_scale_corrected")
const DATA = "/home/tobias-grasberger/sparse-reconstruction-toolkit/testdata"

function main()
    paths = (
        joinpath(ROOT, "lasso_paths/gpu_blackwell_csr/coefficients.jld2"),
        joinpath(OUT, "coefficients.jld2"),
        joinpath(ROOT, "lasso_paths_dual_gap_1e4/gpu_blackwell_csr/coefficients.jld2"),
    )
    saved = map(JLD2.load, paths)
    lambdas = saved[1]["lambdas"]
    all(d -> length(d["lambdas"]) == length(lambdas) &&
             all(isapprox.(d["lambdas"], lambdas; rtol=1e-12)), saved) ||
        error("Saved lambda grids differ")
    truth = vec(permutedims(JLD2.load(joinpath(DATA, "target.jld2"))["single_stored_object"]))
    norm_truth = norm(truth)
    noise_norm = JLD2.load(joinpath(DATA, "L2_norm_of_noise_SNR30.jld2"))["single_stored_object"]
    coefficients = map(d -> d["coefficients"], saved)
    mkpath(OUT)
    open(joinpath(OUT, "field_error_comparison.csv"), "w") do io
        println(io, "index,lambda,noise_norm,old_relative_field_error,corrected_relative_field_error,dual_gap_relative_field_error")
        for (i, lambda) in enumerate(lambdas)
            errors = map(C -> norm(C[:, i] - truth) / norm_truth, coefficients)
            println(io, join((i, lambda, noise_norm, errors...), ","))
        end
    end
end

main()
