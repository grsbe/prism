using Prism, Test, Random, LinearAlgebra, SparseArrays, Statistics

# Exact mean and standard deviation of a 1D density given by its log on a grid.
function grid_moments(logdensity, grid)
    weights = exp.(logdensity .- maximum(logdensity))
    weights ./= sum(weights)
    mu = sum(grid .* weights)
    return mu, sqrt(sum(grid .^ 2 .* weights) - mu^2)
end

@testset "Posterior sampling" begin
    quick = Gibbs(warmup=200, draws=50, chains=2, thin=1)

    @testset "validation" begin
        rng = MersenneTwister(1)
        A, b = randn(rng, 6, 4), randn(rng, 6)
        problem = LassoProblem(A, b)
        @test_throws ArgumentError sample_posterior(
            LassoProblem(A, b; loss=Huber(1.0)), quick, 0.5)
        @test_throws ArgumentError sample_posterior(
            LassoProblem(A, b; penalty=Huber(1.0)), quick, 0.5)
        @test_throws ArgumentError sample_posterior(
            LassoProblem(Float32.(A), Float32.(b)), quick, 0.5)
        @test_throws ArgumentError sample_posterior(problem, quick, 0)
        @test_throws ArgumentError sample_posterior(problem, quick, 0.5;
                                                    noise_variance=0)
        @test_throws DimensionMismatch sample_posterior(problem, quick, 0.5;
                                                        x0=zeros(3))
        @test_throws ArgumentError Gibbs(draws=0)
        @test_throws ArgumentError Gibbs(thin=0)
        @test_throws ArgumentError Gibbs(chains=0)
        @test_throws ArgumentError Gibbs(warmup=-1)
    end

    @testset "one-coefficient draws" begin
        rng = MersenneTwister(2)
        for (center, shift, sigma) in ((0.7, 0.3, 0.5), (-0.2, 1.0, 1.0),
                                       (0.0, 2.0, 0.4), (30.0, 1.0, 1.0),
                                       (-4.0, 6.0, 0.5), (0.5, 40.0, 1.0))
            g = range(-abs(center) - 12sigma, abs(center) + 12sigma;
                      length=2_000_001)
            mu, sd = grid_moments(@.(-(g - center)^2 / (2sigma^2) -
                                     shift * abs(g) / sigma^2), g)
            x = [Prism._rand_two_sided_normal(rng, center, shift, sigma)
                 for _ in 1:200_000]
            @test abs(mean(x) - mu) < 0.03 * sd
            @test std(x) ≈ sd rtol=0.03
        end
        for (mu0, sigma) in ((0.7, 0.5), (-0.2, 1.0), (-3.0, 0.4),
                             (-20.0, 1.0), (5.0, 1.0))
            g = range(0, max(mu0, 0) + 12sigma; length=2_000_001)
            mu, sd = grid_moments(@.(-(g - mu0)^2 / (2sigma^2)), g)
            x = [Prism._rand_truncated_normal(rng, mu0, sigma)
                 for _ in 1:200_000]
            @test mean(x) ≈ mu rtol=0.03
            @test std(x) ≈ sd rtol=0.03
            @test all(>=(0), x)
        end
    end

    @testset "two coefficients against a grid: $constraint, m=$m" for
            constraint in (:signed, :nonnegative), m in (1, 6)
        rng = MersenneTwister(10 + m)
        A = randn(rng, m, 2)
        A[:, 2] = 0.9 .* A[:, 1] .+ 0.3 .* randn(rng, m)
        b = A * [1.0, 0.0] .+ 0.4 .* randn(rng, m)
        lambda, noise_variance = 0.5, 0.3
        g = range(constraint == :signed ? -12 : 0, 12; length=1201)
        logd = [-(sum(abs2, A * [u, v] - b) / 2 +
                  lambda * (abs(u) + abs(v))) / noise_variance
                for u in g, v in g]
        d = exp.(logd .- maximum(logd))
        d ./= sum(d)
        exact_mean = [sum(d .* g), sum(d .* g')]
        exact_std = [sqrt(sum(d .* g .^ 2) - exact_mean[1]^2),
                     sqrt(sum(d .* (g') .^ 2) - exact_mean[2]^2)]
        samples = sample_posterior(
            LassoProblem(A, b; constraint), Gibbs(warmup=1_000, draws=25_000,
                                                  chains=2, thin=1),
            lambda; noise_variance, rng=MersenneTwister(1))
        @test posterior_mean(samples) ≈ exact_mean atol=0.03
        @test posterior_std(samples) ≈ exact_std atol=0.03
        @test maximum(samples.rhat) < 1.05
    end

    @testset "residual form, Gram form and sparse storage agree" begin
        rng = MersenneTwister(20)
        A = randn(rng, 30, 8)
        b = A * randn(rng, 8) + 0.5 .* randn(rng, 30)
        for signed in (true, false)
            gram = Prism._GibbsWork(A, b, 1.0, 0.25, signed)
            @test gram.gram_form
            residual = Prism._GibbsWork(A, b, 1.0, 0.25, signed,
                                        gram.column_squares, false,
                                        zeros(0, 0))
            sparse_work = Prism._GibbsWork(sparse(A), b, 1.0, 0.25, signed)
            @test !sparse_work.gram_form
            runs = [Prism._gibbs_chain(Xoshiro(seed), work, zeros(8);
                                       warmup=1_000, draws=50_000, thin=1)
                    for (seed, work) in ((1, gram), (2, residual),
                                         (3, sparse_work))]
            for other in runs[2:3]
                @test mean(runs[1]; dims=2) ≈ mean(other; dims=2) atol=0.03
                @test std(runs[1]; dims=2) ≈ std(other; dims=2) atol=0.03
            end
        end
    end

    @testset "zero column falls back to the prior" begin
        rng = MersenneTwister(30)
        A = randn(rng, 5, 3)
        A[:, 2] .= 0
        b = randn(rng, 5)
        method = Gibbs(warmup=500, draws=20_000, chains=2, thin=1)
        positive = sample_posterior(LassoProblem(A, b; constraint=:nonnegative),
                                    method, 0.8; noise_variance=0.4,
                                    rng=MersenneTwister(1))
        @test posterior_mean(positive)[2] ≈ 0.5 rtol=0.05
        signed = sample_posterior(LassoProblem(A, b), method, 0.8;
                                  noise_variance=0.4, rng=MersenneTwister(1))
        @test posterior_mean(signed)[2] ≈ 0 atol=0.03
        @test mean(abs, signed.x[2, :, :]) ≈ 0.5 rtol=0.05
    end

    @testset "shapes, signs, reproducibility and conveniences" begin
        rng = MersenneTwister(40)
        A, b = randn(rng, 10, 6), randn(rng, 10)
        # Runs this short are expected to trigger the mixing warning.
        @test_logs (:warn, r"have not mixed") match_mode=:any sample_posterior(
            LassoProblem(A, b), quick, 0.8; rng=MersenneTwister(5))
        @test_logs sample_posterior(
            LassoProblem(A, b), Gibbs(warmup=200, draws=1_000, chains=2, thin=5),
            0.8; rng=MersenneTwister(5))
        silent = Base.CoreLogging.NullLogger()
        Base.CoreLogging.with_logger(silent) do
        for constraint in (:signed, :nonnegative)
            problem = LassoProblem(A, b; constraint)
            samples = sample_posterior(problem, quick, 0.8;
                                       rng=MersenneTwister(5))
            @test size(samples.x) == (6, 50, 2)
            @test length(samples.rhat) == 6
            @test length(samples.ess) == 6
            @test samples.constraint == constraint
            @test samples.lambda == 0.8
            @test (minimum(samples.x) >= 0) == (constraint == :nonnegative)
            interval = credible_interval(samples)
            @test all(interval.lower .<= posterior_quantile(samples, 0.5) .<=
                      interval.upper)
            @test_throws ArgumentError posterior_quantile(samples, 1.5)
            @test_throws ArgumentError credible_interval(samples; level=1)
            @test occursin("PosteriorSamples(6 coefficients", sprint(show, samples))

            again = sample_posterior(problem, quick, 0.8; rng=MersenneTwister(5))
            other = sample_posterior(problem, quick, 0.8; rng=MersenneTwister(6))
            @test again.x == samples.x
            @test other.x != samples.x

            sparse_samples = sample_posterior(
                LassoProblem(sparse(A), b; constraint), quick, 0.8;
                rng=MersenneTwister(5))
            @test size(sparse_samples.x) == (6, 50, 2)

            solution = solve(problem, FISTA(), 0.8)
            @test sample_posterior(problem, quick, solution;
                                   rng=MersenneTwister(1)).lambda == 0.8
            # Without a method argument the default Gibbs() is used.
            @test sample_posterior(problem, solution;
                                   rng=MersenneTwister(1)).lambda == 0.8
            @test sample_posterior(problem, 0.8;
                                   rng=MersenneTwister(1)).lambda == 0.8
        end
        end
    end

    @testset "posterior concentrates on the LASSO solution" begin
        for constraint in (:signed, :nonnegative)
            rng = MersenneTwister(50)
            A = randn(rng, 200, 5)
            b = A * [2.0, 0.0, 1.0, 0.0, 3.0] + 0.1 .* randn(rng, 200)
            problem = LassoProblem(A, b; constraint)
            solution = solve(problem, FISTA(stop=KKTStop(rtol=1e-6)), 0.5)
            samples = sample_posterior(
                problem, Gibbs(warmup=200, draws=2_000, chains=2, thin=1),
                solution; noise_variance=0.01, rng=MersenneTwister(1))
            @test posterior_mean(samples) ≈ solution.x atol=0.1
        end
    end

    @testset "diagnostics" begin
        rng = MersenneTwister(3)
        x = randn(rng, 4000, 4)
        @test Prism._split_rhat(x) ≈ 1 atol=0.01
        @test 10_000 < Prism._effective_sample_size(x) < 17_000
        x[:, 4] .+= 3
        @test Prism._split_rhat(x) > 1.3
    end
end
