using LinearAlgebra, Random, Statistics
struct W2; A::Matrix{Float64}; b::Vector{Float64}; rate::Float64; nv::Float64; obs::Bool; K::Matrix{Float64}; c::Vector{Float64}; end
W2(A, b, lambda, nv) = (o = size(A,1) <= size(A,2); W2(A, b, (lambda/nv)^2/2, nv, o, o ? zeros(0,0) : (A'A)./nv, o ? zeros(0) : (A'b)./nv))
function lpg(p, theta)
    w = exp.(theta)
    if p.obs
        F = cholesky(Symmetric((p.A .* w') * p.A' + p.nv * I); check=false); issuccess(F) || return -Inf, zero(theta)
        alpha = F \ p.b; z = p.A' * alpha; q = vec(sum(abs2, F.L \ p.A; dims=1))
        logp = -sum(log, diag(F.L)) - dot(p.b, alpha)/2 - p.rate*sum(w) + sum(theta)
    else
        s = sqrt.(w); F = cholesky(Symmetric(s .* p.K .* s' + I); check=false); issuccess(F) || return -Inf, zero(theta)
        t = F \ (s .* p.c); logp = -sum(log, diag(F.L)) + dot(t, s .* p.c)/2 - p.rate*sum(w) + sum(theta)
        X = s .* (F \ (s .* p.K)); q = diag(p.K) .- vec(sum(p.K .* X'; dims=2)); z = p.c .- p.K * (s .* t)
    end
    return logp, w .* (-q ./ 2 .+ z .^ 2 ./ 2 .- p.rate) .+ 1
end
function drawx(rng, p, theta)
    w = exp.(theta)
    if p.obs
        x0 = sqrt.(w) .* randn(rng, length(w)); e = sqrt(p.nv) .* randn(rng, length(p.b))
        F = cholesky(Symmetric((p.A .* w') * p.A' + p.nv * I)); return x0 .+ w .* (p.A' * (F \ (p.b .- p.A * x0 .- e)))
    else
        s = sqrt.(w); F = cholesky(Symmetric(s .* p.K .* s' + I)); return s .* (F \ (s .* p.c)) .+ s .* (F.U \ randn(rng, length(w)))
    end
end
function run(rng, A, b, lambda, nv; warmup, draws, chains, x0=nothing)
    n = size(A, 2); work = W2(A, b, lambda, nv); delta = lambda/nv
    base = isnothing(x0) ? fill(log(2/delta^2), n) : log.(abs.(x0) ./ delta .+ 1/delta^2)
    seeds = rand(rng, UInt64, chains); X = zeros(n, draws, chains); acc = zeros(chains)
    for k in 1:chains
        cr = Xoshiro(seeds[k]); theta = base .+ 0.5 .* randn(cr, n)
        logp, g = lpg(work, theta); ls = log(1e-2); a = 0
        for it in 1:(warmup + draws)
            h = exp(ls); fm = theta .+ (h/2) .* g; prop = fm .+ sqrt(h) .* randn(cr, n)
            pl, pg = lpg(work, prop); bm = prop .+ (h/2) .* pg
            lr = pl - logp - sum(abs2, theta .- bm)/(2h) + sum(abs2, prop .- fm)/(2h)
            pr = isfinite(lr) ? min(1.0, exp(lr)) : 0.0
            if rand(cr) < pr; theta, logp, g = prop, pl, pg; it > warmup && (a += 1); end
            if it <= warmup; ls += (pr - 0.574)/sqrt(it) else X[:, it - warmup, k] = drawx(cr, work, theta) end
        end
        acc[k] = a/draws
    end
    return X, acc
end
function rinvgauss(rng, mu, shape)
    y = randn(rng)^2; x = mu + mu^2*y/(2shape) - mu/(2shape)*sqrt(4mu*shape*y + mu^2*y^2)
    return rand(rng) <= mu/(mu + x) ? x : mu^2/x
end
function gibbs(rng, A, b, lambda, nv; warmup, draws)
    n = size(A, 2); delta = lambda/nv; K = (A'A)./nv; c = (A'b)./nv; w = ones(n); X = zeros(n, draws)
    for it in 1:(warmup + draws)
        F = cholesky(Symmetric(K + Diagonal(1 ./ w))); x = F \ c .+ F.U \ randn(rng, n)
        for i in 1:n; w[i] = 1/rinvgauss(rng, delta/max(abs(x[i]), 1e-300), delta^2); end
        it > warmup && (X[:, it - warmup] = x)
    end
    return X
end
function rhat(x); half = size(x,1) ÷ 2; parts = hcat(x[1:half,:], x[half+1:2half,:]); wi = mean(var(parts; dims=1)); sqrt(((half-1)/half*wi + var(vec(mean(parts; dims=1))))/wi); end
for seed in 1:3, (m, n) in ((12, 20), (40, 8))
    rng = MersenneTwister(seed)
    A = randn(rng, m, n); xs = zeros(n); xs[[2, 5]] = [2.0, -1.5]; b = A*xs + sqrt(0.3)*randn(rng, m)
    t = @elapsed begin
        X, acc = run(rng, A, b, 0.8, 0.3; warmup=5_000, draws=10_000, chains=4)
        G = gibbs(rng, A, b, 0.8, 0.3; warmup=2_000, draws=40_000)
    end
    P = reshape(X, n, :)
    println("seed=$seed m=$m n=$n  mean diff=", round(maximum(abs, mean(P; dims=2) - mean(G; dims=2)), digits=4),
            " std diff=", round(maximum(abs, std(P; dims=2) - std(G; dims=2)), digits=4),
            " max rhat=", round(maximum(rhat(X[i, :, :]) for i in 1:n), digits=3), " acc=", round.(acc, digits=2), " time=", round(t, digits=1), "s")
end
# 1D with the prescribed budget
for seed in 1:3
    X, acc = run(MersenneTwister(seed), reshape([1.3], 1, 1), [0.9], 0.6, 0.5; warmup=2_000, draws=40_000, chains=1)
    println("1D seed=$seed mean=", round(mean(X), digits=4), " var=", round(var(X), digits=4), " acc=", round.(acc, digits=2))
end
