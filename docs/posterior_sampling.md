# Posterior sampling: uncertainty at one chosen λ

A LASSO solve returns one coefficient vector per $\lambda$ and says nothing
about how uncertain each coefficient is. `sample_posterior` adds that for
**one fixed $\lambda$**: it draws many coefficient vectors from a probability
distribution whose most probable point is exactly the LASSO solution at that
$\lambda$. The spread of the draws is the uncertainty.

```julia
using Prism
problem = LassoProblem(A, b; constraint=:nonnegative)   # whitened A and b
path = solve_path(problem, FISTA())
chosen = path[12]                                        # you pick one λ
samples = sample_posterior(problem, chosen)

posterior_mean(samples)                  # one value per coefficient
posterior_std(samples)
interval = credible_interval(samples; level=0.9)   # interval.lower, interval.upper
posterior_quantile(samples, 0.5)
samples.x                                # n × draws × chains array of all draws
```

`sample_posterior(problem, lambda)` with a number instead of a `SolveResult`
also works; passing the result additionally starts the sampler at the LASSO
solution. Pass `rng=MersenneTwister(1)` (or any seeded generator) for a
reproducible result.

## What distribution is sampled

For noise variance $\sigma^2$ (`noise_variance`) the draws follow

$$
p(x\mid b)\;\propto\;
\exp\!\Bigl(-\frac{\tfrac12\|Ax-b\|_2^2+\lambda\|x\|_1}{\sigma^2}\Bigr),
$$

on all of $\mathbb R^n$ for `constraint=:signed` and restricted to $x\ge0$
for `constraint=:nonnegative`. The exponent is the LASSO objective, so the
point where this density is largest is the LASSO solution at the same
$\lambda$ and the same constraint, for every $\sigma^2$.

Read as a Bayesian model, this is Gaussian measurement noise with variance
$\sigma^2$ on every entry of $b$, combined with an independent prior on every
coefficient that prefers small values: a Laplace (double-exponential) prior
for signed coefficients and an exponential prior for nonnegative ones, both
with mean magnitude $\sigma^2/\lambda$. A larger $\lambda$ is a tighter prior.
The derivation is in [the mathematics page](mathematics.md#posterior-sampling-at-one-λ).

## What you must supply and what the result means

- **`noise_variance`** is the variance of the noise in each entry of `b`, in
  the units of `b` as passed in. The default `1` is correct only when `A` and
  `b` have been whitened (see "Preparing scientific data" in the mathematics
  page). It does not move the most probable point; it sets the width. A value
  that is too small gives intervals that are too narrow.
- **The intervals are conditional on $\lambda$ and on `noise_variance`.**
  Uncertainty about which $\lambda$ is right is not included. Sampling at two
  nearby $\lambda$ values shows how much the choice matters.
- **Draws are not sparse.** A LASSO solution has exact zeros; a posterior draw
  never does, and neither does the posterior mean. The posterior mean is also
  shrunk differently from the LASSO solution. A coefficient that the LASSO
  sets to zero typically shows up as a distribution concentrated near zero.
- **Near-collinear columns** give wide, strongly correlated distributions for
  the coefficients involved. This is the honest answer: the data cannot tell
  them apart. Sums over such coefficients (computed per draw from `samples.x`)
  can be much better determined than the individual coefficients.

## How it works

The sampler is a **Gibbs sampler**: it visits the coefficients one after
another and replaces each by a fresh random value drawn from its exact
distribution given the current values of all the others. For this posterior
that one-coefficient distribution is known in closed form (a normal
distribution cut off at zero, or two such pieces glued together at zero in the
signed case), so there is no step size, no acceptance rate and nothing to
tune. One pass over all coefficients is a *sweep* and costs about as much as
two matrix-vector products.

`Gibbs(; warmup=1_000, draws=2_000, chains=4, thin=5)` controls the run. Each
of the `chains` independent runs discards `warmup` sweeps, then stores every
`thin`-th sweep until it has `draws` stored draws.

## Check the diagnostics

Successive sweeps are correlated, so 8,000 stored draws are worth fewer than
8,000 independent ones. Two numbers per coefficient tell you whether the run
was long enough:

- `samples.rhat` compares the chains with each other. It should be below
  about 1.05. Larger values mean the chains have not yet settled on the same
  distribution.
- `samples.ess` is the effective sample size: how many independent draws the
  stored ones are worth. A few hundred is enough for a mean and a 90 %
  interval.

`sample_posterior` warns when any coefficient has `rhat > 1.05` or
`ess < 100`. If that happens, increase `thin` or `draws`. Printing `samples`
shows the worst `rhat` and the smallest `ess`.

## Cost and mixing

Measured on the development machine, on one CPU core, for $n=6000$
coefficients. One sweep (signed; nonnegative is about 10 % cheaper):

| m | n | Nonzero fraction of `A` | Dense `Matrix` | `SparseMatrixCSC` |
| --- | --- | --- | --- | --- |
| 1500 | 6000 | 33 % | 3.8 ms | 4.3 ms |
| 1500 | 6000 | 5 % | 4.0 ms | 1.1 ms |
| 6000 | 6000 | 33 % | 16 ms | 16 ms |
| 6000 | 6000 | 5 % | 16 ms | 2.8 ms |

The default `Gibbs()` runs 11,000 sweeps per chain and four chains, so about
3 minutes in total for $m=1500$ and about 12 minutes for $m=6000$ with a dense
matrix. Sparse storage only pays off when clearly fewer than a third of the
entries are nonzero. The chains run one after another on one core.

The sampler updates one coefficient at a time, so coefficients whose columns
of $A$ overlap strongly move slowly together. On a synthetic test with
overlapping plume-shaped rows ($n=6000$), half of the coefficients reached an
effective sample size of about 97 % of the stored draws, but the worst
coefficient reached only 12 % ($m=1500$) and 3 to 13 % ($m=6000$). `rhat`
stayed below 1.03 in all cases. Always check `ess` for the coefficients that
matter.

## Limits

- Least-squares loss and L1 penalty only; the Huber loss and Huber penalty
  are rejected. $\lambda$ must be positive.
- CPU `Float64` only, with a `Matrix` or a `SparseMatrixCSC`. GPU arrays and
  `Float32` are rejected.
- Only `LassoProblem`. Group, block, L1-ball, graph-Laplacian and TV problems
  have no sampler.
- Not included: selecting $\lambda$, estimating `noise_variance`, correlated
  noise (whiten first), and parallel chains.
