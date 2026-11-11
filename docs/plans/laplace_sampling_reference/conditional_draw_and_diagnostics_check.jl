using LinearAlgebra, Random, Statistics
rng = MersenneTwister(3)
# coefficient-space conditional draw: check mean and covariance algebra exactly
m, n = 15, 6; A = randn(rng, m, n); b = randn(rng, m); s2 = 0.4; w = exp.(randn(rng, n)); s = sqrt.(w)
K = A'A ./ s2; c = A'b ./ s2
F = cholesky(Symmetric(s .* K .* s' + I))
Sig = inv(K + Diagonal(1 ./ w))
println("coef mean err: ", maximum(abs, s .* (F \ (s .* c)) - Sig * c))
M = s .* inv(F.U)            # draw = mean + M * z
println("coef cov err:  ", maximum(abs, M * M' - Sig))
# observation-space draw: x = x0 + W A' C^-1 (b - A x0 - e) ; linear map check of covariance
C = Symmetric((A .* w') * A' + s2 * I); Ci = inv(C)
G = (w .* A') * Ci
println("obs mean err:  ", maximum(abs, G * b - Sig * c))
L1 = (I - G * A) .* s'; L2 = G .* sqrt(s2)
println("obs cov err:   ", maximum(abs, L1 * L1' + L2 * L2' - Sig))

# diagnostics: x is draws x chains for one scalar quantity
function split_rhat(x::AbstractMatrix)
    half = size(x, 1) ÷ 2
    parts = hcat(x[1:half, :], x[half+1:2half, :])
    within = mean(var(parts; dims=1))
    between = var(vec(mean(parts; dims=1)))
    within > 0 || return NaN
    return sqrt(((half - 1) / half * within + between) / within)
end
function effective_sample_size(x::AbstractMatrix)
    N, M = size(x)
    centered = x .- mean(x; dims=1)
    within = mean(var(x; dims=1))
    between = M > 1 ? var(vec(mean(x; dims=1))) : zero(within)
    total = (N - 1) / N * within + between
    total > 0 || return NaN
    rho(t) = 1 - (within - mean(sum(view(centered, 1:N-t, j) .* view(centered, t+1:N, j)) / N for j in 1:M)) / total
    tau = -1.0; t = 0
    while t + 1 < N                 # Geyer: sum pairs of autocorrelations while positive
        pair = rho(t) + rho(t + 1)
        pair > 0 || break
        tau += 2 * pair; t += 2
    end
    return N * M / max(tau, 1 / log10(N * M))
end
iid = randn(rng, 4000, 4)
println("iid: rhat=", round(split_rhat(iid), digits=3), " ess=", round(effective_sample_size(iid)), " (expect ~16000)")
ar = zeros(4000, 4); for j in 1:4, i in 2:4000; ar[i, j] = 0.9ar[i-1, j] + sqrt(1 - 0.81) * randn(rng); end
println("AR(0.9): rhat=", round(split_rhat(ar), digits=3), " ess=", round(effective_sample_size(ar)), " (expect ~", round(16000 * 0.1 / 1.9), ")")
bad = iid .+ [0 0 0 3]
println("shifted chain: rhat=", round(split_rhat(bad), digits=3))
