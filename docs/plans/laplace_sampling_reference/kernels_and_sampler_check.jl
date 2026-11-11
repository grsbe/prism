using LinearAlgebra, Random, Statistics

# ---------- kernels (candidate reference code for the plan) ----------
struct Work{T}
    A::Matrix{T}; b::Vector{T}; rate::T; s2::T   # rate = delta^2/2, delta = lambda/s2
end
# observation-space form: C = s2*I + A*W*A'
function logp_grad_obs(p::Work, theta)
    w = exp.(theta)
    AW = p.A .* w'
    C = Symmetric(AW * p.A' + p.s2 * I)
    F = cholesky(C)
    alpha = F \ p.b
    z = p.A' * alpha                       # a_i' C^-1 b
    G = F.L \ p.A                          # m x n
    q = vec(sum(abs2, G; dims=1))          # a_i' C^-1 a_i
    lp = -sum(log, diag(F.L)) - dot(p.b, alpha)/2 - p.rate*sum(w) + sum(theta)
    g = w .* (-q ./ 2 .+ z.^2 ./ 2 .- p.rate) .+ 1
    return lp, g
end
# coefficient-space form (colleague's): B = I + W^1/2 K W^1/2, K = A'A/s2
function logp_grad_coef(p::Work, theta)
    w = exp.(theta); s = sqrt.(w)
    K = (p.A' * p.A) ./ p.s2; c = (p.A' * p.b) ./ p.s2
    F = cholesky(Symmetric(s .* K .* s' + I))
    t = F \ (s .* c)
    lp = -sum(log, diag(F.L)) + dot(t, s .* c)/2 - p.rate*sum(w) + sum(theta)
    X = s .* (F \ (s .* K))
    q = diag(K) .- vec(sum(K .* X'; dims=2))
    z = c .- K * (s .* t)
    g = w .* (-q ./ 2 .+ z.^2 ./ 2 .- p.rate) .+ 1
    return lp, g
end
function draw_x(rng, p::Work, theta)
    w = exp.(theta)
    x0 = sqrt.(w) .* randn(rng, length(w))
    e = sqrt(p.s2) .* randn(rng, length(p.b))
    C = Symmetric((p.A .* w') * p.A' + p.s2 * I)
    return x0 .+ w .* (p.A' * (cholesky(C) \ (p.b .- p.A * x0 .- e)))
end
function mala(rng, p::Work, theta0; warmup, draws, step0=1e-2, target=0.574)
    n = length(theta0); theta = copy(theta0)
    lp, g = logp_grad_obs(p, theta)
    logh = log(step0); acc = 0
    TH = zeros(n, draws); LP = zeros(draws)
    for it in 1:(warmup + draws)
        h = exp(logh)
        mean_f = theta .+ (h/2) .* g
        prop = mean_f .+ sqrt(h) .* randn(rng, n)
        lpp, gp = logp_grad_obs(p, prop)
        mean_b = prop .+ (h/2) .* gp
        logr = lpp - lp - sum(abs2, theta .- mean_b)/(2h) + sum(abs2, prop .- mean_f)/(2h)
        a = isfinite(logr) ? min(1.0, exp(logr)) : 0.0
        if rand(rng) < a
            theta, lp, g = prop, lpp, gp
            it > warmup && (acc += 1)
        end
        if it <= warmup
            logh += (a - target) / sqrt(it)      # Robbins-Monro, frozen after warmup
        else
            TH[:, it - warmup] = theta; LP[it - warmup] = lp
        end
    end
    return TH, LP, acc/draws, exp(logh)
end
# ---------- independent reference: Park-Casella Gibbs ----------
function rinvgauss(rng, mu, lam)
    nu = randn(rng); y = nu^2
    x = mu + mu^2*y/(2lam) - mu/(2lam)*sqrt(4mu*lam*y + mu^2*y^2)
    return rand(rng) <= mu/(mu + x) ? x : mu^2/x
end
function gibbs(rng, p::Work, delta; warmup, draws)
    n = size(p.A, 2); x = zeros(n); w = ones(n)
    K = (p.A' * p.A) ./ p.s2; c = (p.A' * p.b) ./ p.s2
    X = zeros(n, draws)
    for it in 1:(warmup + draws)
        F = cholesky(Symmetric(K + Diagonal(1 ./ w)))
        x = F \ c .+ F.U \ randn(rng, n)
        for i in 1:n
            w[i] = 1 / rinvgauss(rng, delta / max(abs(x[i]), 1e-300), delta^2)
        end
        it > warmup && (X[:, it - warmup] = x)
    end
    return X
end

rng = MersenneTwister(7)
# (a) two forms agree; gradient vs finite differences
m, n = 12, 20
A = randn(rng, m, n); xs = zeros(n); xs[[3, 11]] = [2.0, -1.5]; s2 = 0.3
b = A*xs + sqrt(s2)*randn(rng, m); lambda = 0.8; delta = lambda/s2
p = Work(A, b, delta^2/2, s2)
th = randn(rng, n) .- 1
l1, g1 = logp_grad_obs(p, th); l2, g2 = logp_grad_coef(p, th)
th2 = randn(rng, n)
d1 = logp_grad_obs(p, th2)[1] - l1; d2 = logp_grad_coef(p, th2)[1] - l2
println("forms differ only by a constant: ", abs(d1 - d2), "   grad diff: ", maximum(abs, g1 - g2))
fd = [ (logp_grad_obs(p, th .+ 1e-6 .* (1:n .== i))[1] - logp_grad_obs(p, th .- 1e-6 .* (1:n .== i))[1]) / 2e-6 for i in 1:n]
println("grad vs finite diff: ", maximum(abs, fd - g1))

# (b) 1D analytic check: posterior ∝ exp(-(a x - b)^2/(2 s2) - delta |x|)
a1, b1, s21, lam1 = 1.3, 0.9, 0.5, 0.6; del1 = lam1/s21
p1 = Work(reshape([a1], 1, 1), [b1], del1^2/2, s21)
grid = range(-10, 10; length=400_001)
dens = @. exp(-(a1*grid - b1)^2/(2s21) - del1*abs(grid)); dens ./= sum(dens)
qm, qv = sum(grid .* dens), sum(grid.^2 .* dens)
TH, LP, acc, h = mala(rng, p1, [0.0]; warmup=5_000, draws=200_000)
xs1 = [draw_x(rng, p1, TH[:, k])[1] for k in 1:size(TH, 2)]
println("1D: quad mean=$(round(qm, digits=4)) var=$(round(qv-qm^2, digits=4)) | MALA mean=$(round(mean(xs1), digits=4)) var=$(round(var(xs1), digits=4)) acc=$(round(acc, digits=2))")

# (c) multi-D: MALA+conditional draw vs Gibbs
t0 = time(); TH, LP, acc, h = mala(rng, p, log.(fill(1/(delta^2/2), n)); warmup=20_000, draws=200_000); tm = time() - t0
Xm = reduce(hcat, [draw_x(rng, p, TH[:, k]) for k in 1:20:size(TH, 2)])
Xg = gibbs(rng, p, delta; warmup=5_000, draws=200_000)
println("multi-D acc=$(round(acc, digits=2)) step=$(round(h, sigdigits=3)) time=$(round(tm, digits=1))s")
println("max |mean diff| = ", maximum(abs, mean(Xm; dims=2) - mean(Xg; dims=2)), "   max |std diff| = ", maximum(abs, std(Xm; dims=2) - std(Xg; dims=2)))
println("posterior sd range: ", extrema(std(Xg; dims=2)))
q(X, pr) = [quantile(view(X, i, :), pr) for i in 1:size(X, 1)]
println("max |q05 diff| = ", maximum(abs, q(Xm, .05) - q(Xg, .05)), "  max |q95 diff| = ", maximum(abs, q(Xm, .95) - q(Xg, .95)))
