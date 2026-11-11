using LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "kernels.jl"))
function grid_moments_1d(mu, sg)
    g = range(0, max(mu, 0) + 12sg; length=2_000_001); d = @. exp(-(g - mu)^2 / (2sg^2)); d ./= sum(d)
    m = sum(g .* d); return m, sqrt(sum(g .^ 2 .* d) - m^2)
end
function grid_moments_2d(A, b, lambda, nv)
    g = range(0, 12; length=1201)
    logd = [-(sum(abs2, A * [u, v] - b) / 2 + lambda * (u + v)) / nv for u in g, v in g]
    d = exp.(logd .- maximum(logd)); d ./= sum(d)
    m1 = sum(d .* g); m2 = sum(d .* g')
    return [m1, m2], [sqrt(sum(d .* g .^ 2) - m1^2), sqrt(sum(d .* (g') .^ 2) - m2^2)]
end
run(seed, A, b, lambda, nv; chains=2, kw...) = reduce(hcat, [_nonnegative_chain(Xoshiro(s), _NonnegativeWork(A, b, lambda, nv), zeros(size(A, 2)); kw...) for s in rand(MersenneTwister(seed), UInt64, chains)])
worst = Dict{String,Float64}()
note(k, v) = (worst[k] = max(get(worst, k, 0.0), v))
for seed in 1:3
    rng = MersenneTwister(seed)
    for (mu, sg) in ((0.7, 0.5), (-0.2, 1.0), (-3.0, 0.4), (-20.0, 1.0), (5.0, 1.0))
        qm, qs = grid_moments_1d(mu, sg); s = [_rand_truncated_normal(rng, mu, sg) for _ in 1:200_000]
        note("truncated normal: relative error of mean", abs(mean(s) - qm) / qm); note("truncated normal: relative error of sd", abs(std(s) - qs) / qs)
    end
    for m in (1, 6)
        r2 = MersenneTwister(10 + m)
        A = randn(r2, m, 2); A[:, 2] = 0.9 .* A[:, 1] .+ 0.3 .* randn(r2, m); b = A * [1.0, 0.0] .+ 0.4 .* randn(r2, m)
        qm, qs = grid_moments_2d(A, b, 0.5, 0.3)
        X = run(seed, A, b, 0.5, 0.3; warmup=1_000, draws=50_000, thin=1)
        note("2D m=$m: abs error of mean", maximum(abs, mean(X; dims=2) - qm)); note("2D m=$m: abs error of sd", maximum(abs, std(X; dims=2) - qs))
    end
    r3 = MersenneTwister(20); A = randn(r3, 30, 8); b = A * max.(randn(r3, 8), 0) + 0.5 .* randn(r3, 30)
    pg = _NonnegativeWork(A, b, 1.0, 0.25); pr = _NonnegativeWork(A, b, 1.0, 0.25, pg.column_squares, false, zeros(0, 0))
    Xg = _nonnegative_chain(Xoshiro(seed), pg, zeros(8); warmup=1_000, draws=50_000, thin=1)
    Xr = _nonnegative_chain(Xoshiro(seed + 50), pr, zeros(8); warmup=1_000, draws=50_000, thin=1)
    note("Gram vs residual form: abs diff of mean", maximum(abs, mean(Xg; dims=2) - mean(Xr; dims=2))); note("Gram vs residual form: abs diff of sd", maximum(abs, std(Xg; dims=2) - std(Xr; dims=2)))
    A0 = randn(r3, 5, 3); A0[:, 2] .= 0; X0 = run(seed, A0, randn(r3, 5), 0.8, 0.4; warmup=500, draws=20_000, thin=1)
    note("zero column: relative error of mean vs noise_variance/lambda", abs(mean(X0[2, :]) - 0.5) / 0.5)
end
foreach(k -> println(rpad(k, 62), round(worst[k], sigdigits=3)), sort(collect(keys(worst))))
