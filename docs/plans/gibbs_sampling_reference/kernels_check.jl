# Run with an environment that has SpecialFunctions, e.g. the Prism project
# after the dependency is added:  julia --project=<prism> kernels_check.jl
using LinearAlgebra, Random, Statistics, SparseArrays, SpecialFunctions
include(joinpath(@__DIR__, "kernels.jl"))
function grid_1d(logdensity, lo, hi)
    g = range(lo, hi; length=2_000_001); l = logdensity.(g); d = exp.(l .- maximum(l)); d ./= sum(d)
    m = sum(g .* d); return m, sqrt(sum(g .^ 2 .* d) - m^2)
end
function grid_2d(A, b, lambda, nv, signed)
    g = range(signed ? -12 : 0, 12; length=1201)
    logd = [-(sum(abs2, A * [u, v] - b) / 2 + lambda * (abs(u) + abs(v))) / nv for u in g, v in g]
    d = exp.(logd .- maximum(logd)); d ./= sum(d)
    m1 = sum(d .* g); m2 = sum(d .* g')
    return [m1, m2], [sqrt(sum(d .* g .^ 2) - m1^2), sqrt(sum(d .* (g') .^ 2) - m2^2)]
end
chains(seed, A, b, lambda, nv, signed; n=2, kw...) = reduce(hcat, [_gibbs_chain(Xoshiro(s), _GibbsWork(A, b, lambda, nv, signed), zeros(size(A, 2)); kw...) for s in rand(MersenneTwister(seed), UInt64, n)])
# independent reference for the signed posterior (Laplace prior as a normal with random variance)
function rinvgauss(rng, mu, shape)
    y = randn(rng)^2; x = mu + mu^2 * y / (2shape) - mu / (2shape) * sqrt(4mu * shape * y + mu^2 * y^2)
    return rand(rng) <= mu / (mu + x) ? x : mu^2 / x
end
function mixture_reference(rng, A, b, lambda, nv; warmup, draws)
    n = size(A, 2); delta = lambda / nv; K = (A' * A) ./ nv; c = (A' * b) ./ nv; w = ones(n); X = zeros(n, draws)
    for it in 1:(warmup + draws)
        F = cholesky(Symmetric(K + Diagonal(1 ./ w))); x = F \ c .+ F.U \ randn(rng, n)
        for i in 1:n; w[i] = 1 / rinvgauss(rng, delta / max(abs(x[i]), 1e-300), delta^2); end
        it > warmup && (X[:, it - warmup] = x)
    end
    return X
end
worst = Dict{String,Float64}(); note(k, v) = (worst[k] = max(get(worst, k, 0.0), v))
for seed in 1:3
    rng = MersenneTwister(seed)
    for (c, s, sg) in ((0.7, 0.3, 0.5), (-0.2, 1.0, 1.0), (0.0, 2.0, 0.4), (30.0, 1.0, 1.0), (-4.0, 6.0, 0.5), (0.5, 40.0, 1.0))
        qm, qs = grid_1d(x -> -(x - c)^2 / (2sg^2) - s * abs(x) / sg^2, -abs(c) - 12sg, abs(c) + 12sg)
        x = [_rand_two_sided_normal(rng, c, s, sg) for _ in 1:200_000]
        note("two-sided normal: abs error of mean / sd", abs(mean(x) - qm) / qs); note("two-sided normal: relative error of sd", abs(std(x) - qs) / qs)
    end
    for signed in (true, false), m in (1, 6)
        r2 = MersenneTwister(10 + m)
        A = randn(r2, m, 2); A[:, 2] = 0.9 .* A[:, 1] .+ 0.3 .* randn(r2, m); b = A * [1.0, 0.0] .+ 0.4 .* randn(r2, m)
        qm, qs = grid_2d(A, b, 0.5, 0.3, signed)
        X = chains(seed, A, b, 0.5, 0.3, signed; warmup=1_000, draws=50_000, thin=1)
        tag = signed ? "signed" : "nonnegative"
        note("2D $tag m=$m: abs error of mean", maximum(abs, mean(X; dims=2) - qm)); note("2D $tag m=$m: abs error of sd", maximum(abs, std(X; dims=2) - qs))
    end
    r3 = MersenneTwister(20); A = randn(r3, 30, 8); b = A * randn(r3, 8) + 0.5 .* randn(r3, 30)
    for signed in (true, false)
        tag = signed ? "signed" : "nonnegative"
        pg = _GibbsWork(A, b, 1.0, 0.25, signed); pr = _GibbsWork(A, b, 1.0, 0.25, signed, pg.column_squares, false, zeros(0, 0)); ps = _GibbsWork(sparse(A), b, 1.0, 0.25, signed)
        Xg = _gibbs_chain(Xoshiro(seed), pg, zeros(8); warmup=1_000, draws=50_000, thin=1)
        Xr = _gibbs_chain(Xoshiro(seed + 50), pr, zeros(8); warmup=1_000, draws=50_000, thin=1)
        Xs = _gibbs_chain(Xoshiro(seed + 90), ps, zeros(8); warmup=1_000, draws=50_000, thin=1)
        note("$tag Gram vs residual form: abs diff of mean", maximum(abs, mean(Xg; dims=2) - mean(Xr; dims=2))); note("$tag Gram vs residual form: abs diff of sd", maximum(abs, std(Xg; dims=2) - std(Xr; dims=2)))
        note("$tag sparse vs dense: abs diff of mean", maximum(abs, mean(Xs; dims=2) - mean(Xr; dims=2))); note("$tag sparse vs dense: abs diff of sd", maximum(abs, std(Xs; dims=2) - std(Xr; dims=2)))
    end
    r4 = MersenneTwister(30); A = randn(r4, 12, 20); xs = zeros(20); xs[[3, 11]] = [2.0, -1.5]; b = A * xs + sqrt(0.3) * randn(r4, 12)
    Xg = chains(seed, A, b, 0.8, 0.3, true; n=4, warmup=1_000, draws=50_000, thin=1); Xm = mixture_reference(rng, A, b, 0.8, 0.3; warmup=2_000, draws=200_000)
    note("signed 20 coefficients vs independent method: abs diff of mean", maximum(abs, mean(Xg; dims=2) - mean(Xm; dims=2))); note("signed 20 coefficients vs independent method: abs diff of sd", maximum(abs, std(Xg; dims=2) - std(Xm; dims=2)))
    A0 = randn(r3, 5, 3); A0[:, 2] .= 0; b0 = randn(r3, 5)
    X0 = chains(seed, A0, b0, 0.8, 0.4, false; warmup=500, draws=20_000, thin=1); note("zero column nonnegative: relative error of mean vs noise_variance/lambda", abs(mean(X0[2, :]) - 0.5) / 0.5)
    X0 = chains(seed, A0, b0, 0.8, 0.4, true; warmup=500, draws=20_000, thin=1); note("zero column signed: abs mean", abs(mean(X0[2, :]))); note("zero column signed: relative error of mean |x| vs noise_variance/lambda", abs(mean(abs, X0[2, :]) - 0.5) / 0.5)
end
foreach(k -> println(rpad(k, 78), round(worst[k], sigdigits=3)), sort(collect(keys(worst))))
