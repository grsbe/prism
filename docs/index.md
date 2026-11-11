# Prism documentation

Prism is a Julia toolkit for sparse and spatially smooth reconstruction
from linear observations. It separates the algebraic problem, numerical
method, stopping policy, and multi-$\lambda$ workflow. This separation is
the main design goal: the code should make the modeling decisions visible.

## Read in this order

1. [Mathematical models](mathematics.md): what objective each problem
   type minimizes and what positivity means.
2. [Optimality](optimality.md): what `converged` actually tests,
   including the optional SRT-style gradient-mapping rule.
3. [Performance](performance.md): defaults that save time, when they
   cost accuracy or memory, and how to benchmark a real workload.
4. [Grouped cross-validation](cross_validation.md): ordered folds and
   training-fold lambda scaling.
5. [Posterior sampling](posterior_sampling.md): uncertainty of the LASSO
   coefficients at one chosen λ.
6. [Architecture](architecture.md): where solvers, compatibility calls,
   and future methods belong.

The repository [README](../README.md) is the quick start and current
feature/status summary. These Markdown files can become a Documenter.jl
site without changing their intended page boundaries. No website or
Documenter dependency is installed yet.

## The smallest complete workflow

Choose your background and units first; construct $A$ and $b$; then choose
a problem, solver, stop, and path. For penalized nonnegative LASSO:

```julia
using Prism
problem = LassoProblem(A, b; constraint=:nonnegative)
solver = initialize(problem, FISTA())
results = solve_path!(solver)  # 30 values, λmax down to 1e-4 λmax
# Or: solve_path!(solver, lambdas; stop_on_failure=true)
```

The default path has a fixed logarithmic endpoint and does **not** stop when
the residual or RMSE plateaus. Use `lambda_grid(problem; min_ratio=..., n_lambdas=...)`
to inspect or customize it; an explicit descending `lambdas` sequence is
also accepted. The endpoint controls which models are computed, not which
one is selected. For a physically meaningful estimate, select a λ using
an uncertainty model, external validation, or another stated criterion.
The numerical stop controls how accurately each chosen objective is solved.
