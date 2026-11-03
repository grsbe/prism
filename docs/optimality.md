# What `converged` means

FISTA defaults to a two-stage iterate-change and primal-dual-gap stop.
ADMM defaults to primal/dual feasibility plus a returned-point dual gap.
TV-PDHG retains its own primal–dual/KKT rule. A dual gap bounds the
objective suboptimality of the **returned, owned coefficients**. These are numerical
diagnostics, not evidence that the forward model is correct.
`KKTStop(atol, rtol, check_every, maxiter)` and the older
`GradientMappingStop` remain opt-in alternatives. Exhaustion gives
`converged=false` and `stop_reason=:maxiter`.

## Penalized LASSO

For $f(x)=\tfrac12\|Ax-b\|_2^2+\lambda\|x\|_1$, let
$g=A^\top(Ax-b)$. At each coordinate, the signed KKT violation is

$$
v_j=\begin{cases}
|g_j+\lambda|,&x_j>0,\\
|g_j-\lambda|,&x_j<0,\\
\max(|g_j|-\lambda,0),&x_j=0.
\end{cases}
$$

For nonnegative coefficients, only $x_j>0$ and $x_j=0$ occur; at zero use
$v_j=\max(-g_j-\lambda,0)$. With explicit `KKTStop`, FISTA requires
$\max_j v_j\leq\mathrm{atol}+\mathrm{rtol}\,\lambda$. ADMM requires this
*and* its primal and dual residual tests:

$$
\|x-z\|_2\leq\sqrt n\,\epsilon_{\rm abs}
 +\epsilon_{\rm rel}\max(\|x\|_2,\|z\|_2),\qquad
\rho\|z-z_{\rm old}\|_2\leq\sqrt n\,\epsilon_{\rm abs}
 +\epsilon_{\rm rel}\rho\|u\|_2.
$$

`SolveResult.kkt_ratio` is $\max_jv_j/\lambda$ for $\lambda>0$. At
$\lambda=0$, it is either zero or infinity; use the absolute KKT residual.

## Default FISTA dual-gap rule

For each FISTA problem, `DualGapStop(tol=1e-4)` first checks

$$
\|x_{k+1}-x_k\|_\infty
\leq \mathrm{tol}\,\|x_{k+1}\|_\infty.
$$

Only then does it form a **feasible** dual candidate at the returned
coefficients and require $P-D\leq\mathrm{tol}\,\|b\|_2^2$.
Spasmi minimizes an unnormalized least-squares objective; scikit-learn
divides its objective by $n$, hence reports the equivalent threshold
$\mathrm{tol}\,\|b\|_2^2/n$. A small iterate change alone never declares
convergence. The default always performs at least one FISTA step so that
the update gate is actually evaluated.

For signed penalized LASSO, set $r=Ax-b$ and
$s=\max(1,\|A^\top r\|_\infty/\lambda)$. Then $u=r/s$ obeys
$\|A^\top u\|_\infty\leq\lambda$, and

$$
P=\tfrac12\|r\|_2^2+\lambda\|x\|_1,\qquad
D(u)=-\tfrac12\|u\|_2^2-b^\top u.
$$

For $x\geq0$, replace the bound by
$s=\max(1,\max_j[-(A^\top r)_j]_+/\lambda)$; dual feasibility becomes
$A^\top u\geq-\lambda\mathbf1$. Both produce a valid gap.

For disjoint group LASSO, replace the signed L1 dual constraint by
$\|A_g^\top u\|_2\leq\lambda w_g$ for every group. Under coefficient
nonnegativity it becomes $\|(-A_g^\top u)_+\|_2\leq\lambda w_g$.
Spasmi scales the residual by the largest groupwise dual-bound violation
to form a feasible dual point. The returned-point KKT residual is the
largest groupwise subgradient violation **divided by its weight**. At a
zero signed group it is
$[\|A_g^\top r\|_2-\lambda w_g]_+/w_g$; the nonnegative case uses only
negative gradient components.

The other FISTA families use their own duals:

- **L1 ball:** $u=Ax-b$ needs no rescaling. Subtract the support function
  $R\|A^\top u\|_\infty$ from $D(u)$, or
  $R\max_j[-(A^\top u)_j]_+$ for $x\geq0$.
- **Sparse-plus-ridge block:** scale $u=Ax+Bz-b$ for the sparse block.
  With ridge strength $\gamma>0$, subtract
  $\|B^\top u\|_2^2/(2\gamma)$ from the dual objective, using only
  the negative part of $B^\top u$ if $z\geq0$.
- **Squared Laplacian:** use $u=Ax-b$, $v=\mu Lx$, scale both together
  until $A^\top u+L^\top v$ satisfies the L1 dual constraint, and
  subtract $\|v\|_2^2/(2\mu)$ when $\mu>0$.

At $\lambda=0$, and for a block model with zero ridge strength, these
simple residual-based candidates cannot generally provide a tight dual
certificate. The default then uses an explicitly labelled scaled KKT
fallback. No dual gap is falsely reported in those cases. A gap bound
may be conservative when the residual candidate is poorly aligned, so a
solver can reach `maxiter` despite a small KKT residual.

## Optional FISTA gradient-mapping rule

The old SRT `optimality_reltol` rule did **not** evaluate the KKT
gradient at the returned $x_{k+1}$. In one FISTA step, it formed

$$
g(y_k)=A^\top(Ay_k-b),\qquad
x_{k+1}=\operatorname{prox}_{t\lambda\|\cdot\|_1+I_C}
        (y_k-tg(y_k)),
$$

then checked

$
\left\|\frac{y_k-x_{k+1}}{t}\right\|_\infty
\leq a+r s_k.
$

For signed LASSO, $s_k=\max(\|g(y_k)\|_\infty,\epsilon(t))$,
matching SRT. For nonnegative LASSO with $\lambda>0$, Spasmi uses

$
s_k=\max\left\{\lambda,\epsilon(t),
\max_{j:x_{k+1,j}>0}|g_j(y_k)|,
\max_{j:x_{k+1,j}=0}[-g_j(y_k)]_+\right\}.
$

A positive gradient at a zero coefficient points toward an infeasible
negative update; it cannot justify a looser stopping tolerance. At
$\lambda=0$, the raw-gradient scale is retained.

This is a proximal-gradient mapping at the **extrapolated** $y_k$.
It is a valid optimality residual in the limit, and the gradient
and prox result are already available in the iteration. It is not
identical to the coordinatewise KKT residual at $x_{k+1}$.
The difference is
$g(x_{k+1})-g(y_k)=A^\top A(x_{k+1}-y_k)$, which generally requires
additional matrix-vector products to evaluate.

To use the previous Spasmi/SRT-style rule, pass
`FISTA(stop=GradientMappingStop(atol=a, rtol=r, check_every=1))`.
Its defaults are $a=0$, $r=10^{-3}$, and checking every iteration;
the iteration timing matches SRT's optional rule; the nonnegative scale above intentionally differs. `stop_reason=:gradient_mapping` identifies this stop.
The result still reports exact returned-point `kkt_residual` and
`kkt_ratio`; they may exceed the KKT tolerance despite
`converged=true` under this policy. The same extrapolated-point principle
applies to all FISTA families.
For the constrained L1-ball case the mapping uses the Euclidean norm
and the scale below. ADMM and PDHG retain their own iteration-specific
feasibility checks.
SRT's separate *iterate-change-only* default is not reproduced here;
the new default requires a dual certificate after the update gate.

At exit, all penalized LASSO, block, and Laplacian solvers compute the
returned-point KKT residual and warn when `kkt_ratio > 0.1`, regardless
of their stop rule or convergence status. This threshold is fixed.
A passing dual gap bounds objective error but does not necessarily imply
a small KKT residual relative to $\lambda$. At $\lambda=0$ this ratio is
undefined; inspect `kkt_residual` instead. Constrained L1-ball and TV-PDHG
have no $\lambda$-normalized warning. The constrained FISTA warning uses
its final projected-gradient residual divided by
$\max(\|A^\top b\|_2,1)$; its threshold is controlled by
`GradientMappingStop.warn_optimality_ratio` when that stop is selected.

## L1-ball-constrained least squares

For $C=\{x:\|x\|_1\leq R\}$, optionally intersected with $x\geq0$,
Spasmi evaluates the projected-gradient mapping

$$
r_C(x)=\frac{\|x-\Pi_C(x-tA^\top(Ax-b))\|_2}{t}.
$$

It vanishes exactly at a constrained optimum. The implemented tolerance is
$\mathrm{atol}+\mathrm{rtol}\max(\|A^\top b\|_2,1)$; there is **no** λ to
divide by. Constrained ADMM additionally checks the primal and dual residuals
above. Its default also requires a returned-point constrained dual gap;
explicit `KKTStop` instead uses the projected-gradient test.
`ConstrainedResult.optimality_residual` reports $r_C$.

## Sparse-plus-ridge and Laplacian models

For the block model, define $r=Ax+Bz-b$. The sparse block uses the LASSO KKT
test with gradient $A^\top r$; the smooth block uses
$g_z=B^\top r+\gamma z$. For signed $z$, its violation is $|g_z|$; for
nonnegative $z$, it is $|g_z|$ at $z>0$ and $\max(-g_z,0)$ at zero.
`x_constraint` and `z_constraint` select the signs independently, while
the legacy `constraint` keyword sets both. With explicit `KKTStop`, the
largest block violation must be at most
$\mathrm{atol}+\mathrm{rtol}\,\lambda$.

For squared graph-Laplacian smoothing, the LASSO KKT test instead uses
$g=A^\top(Ax-b)+\gamma L^2x$, where $L$ is the symmetric 2D graph
Laplacian. Its tolerance is the same as penalized LASSO.

## Total variation PDHG

The TV objective is

$$
\tfrac12\|Ax-b\|_2^2+\lambda_1\|x\|_1
 +\lambda_{\rm TV}\operatorname{TV}(x)+I_C(x),
$$

with forward-difference operator $D$ and $C$ either the full space or the
nonnegative orthant. At the returned $x$, Spasmi evaluates four conditions:

1. Coefficient stationarity using $g=A^\top y+D^\top p$ and the signed or
   nonnegative LASSO coordinate rule above, with tolerance
   $\mathrm{atol}+\mathrm{rtol}\,\lambda_1$.
2. Data-dual agreement $\|y-(Ax-b)\|_\infty$, with tolerance
   $\mathrm{atol}+\mathrm{rtol}\max(\|b\|_\infty,1)$.
3. TV complementarity gap
   $\max(\lambda_{\rm TV}\operatorname{TV}(x)-\langle p,Dx\rangle,0)$,
   with tolerance $\mathrm{atol}\max(\|x\|_1,1)
   +\mathrm{rtol}\max(\lambda_{\rm TV}\operatorname{TV}(x),1)$.
4. TV dual feasibility: each isotropic pixel vector $p_j$ must have
   $\|p_j\|_2\leq\lambda_{\rm TV}$ (or each anisotropic component must
   have magnitude at most $\lambda_{\rm TV}$), with tolerance
   $\mathrm{atol}+\mathrm{rtol}\,\lambda_{\rm TV}$.

`TVResult` reports all four residuals separately. A small iterate change by
itself never sets `converged=true`.

## Interpretation

Small numerical residuals do not resolve nonidentifiability: strongly
correlated columns can give nearly identical predictions but different
coefficient maps. Conversely, a scientifically stable aggregate can coexist
with an unfinished solve. Inspect both numerical diagnostics and the physical
quantity of interest over nearby regularization parameters.
