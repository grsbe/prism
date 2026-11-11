# Mathematical models and algorithms

Prism works on an algebraic observation model $b\approx Ax$. Here $A$ is
the sensitivity or design matrix, $b$ the prepared observation vector,
and $x$ the unknown coefficients. The package does not infer units or
construct $A$ from a physical model. If the physical field is $Dx+x_0$,
then coefficient constraints on $x$ are not automatically constraints on
that field.

## Penalized LASSO

The signed problem is

$$
\min_{x\in\mathbb R^n}
F_\lambda(x)
=\frac12\|Ax-b\|_2^2+\lambda\|x\|_1.
$$

The nonnegative variant replaces $\mathbb R^n$ with $\mathbb R_+^n$.
The smooth gradient is $g(x)=A^\top(Ax-b)$, with Lipschitz constant
$L=\|A\|_2^2$. FISTA uses a step $t\leq1/L$ and the proximal update

$$
x_{k+1}
=\operatorname{prox}_{t\lambda\|\cdot\|_1+I_C}
       \bigl(y_k-tg(y_k)\bigr),
\qquad
q_{k+1}=\frac{1+\sqrt{1+4q_k^2}}{2},
\qquad
y_{k+1}=x_{k+1}+\frac{q_k-1}{q_{k+1}}(x_{k+1}-x_k).
$$

For signed coefficients the prox is soft thresholding,
$\operatorname{sign}(v_j)\max(|v_j|-t\lambda,0)$; for nonnegative
coefficients it is $\max(v_j-t\lambda,0)$. This means a coefficient can
be exactly zero, which is the source of sparsity. An estimated
$L$ that is too small can make the step unsafe; the default power
estimate is not a proof that $t\leq1/L$.

The ADMM split $x=z$ instead alternates

$$
\begin{aligned}
x^{k+1}&=(A^\top A+\rho I)^{-1}
          \bigl(A^\top b+\rho(z^k-u^k)\bigr),\\
z^{k+1}&=\operatorname{prox}_{(\lambda/\rho)\|\cdot\|_1+I_C}
          (x^{k+1}+u^k),\\
u^{k+1}&=u^k+x^{k+1}-z^{k+1}.
\end{aligned}
$$

The factorization is cached. For $n>m$, the implementation factors
$AA^\top+\rho I$ and uses the Woodbury identity rather than factoring
an $n\times n$ matrix. $\rho$ is fixed for that initialized solver.

## Huber data loss

The plain, group, and block FISTA problems can replace
$\frac12\|r\|_2^2$, $r=Ax-b$, by the Huber loss
$\sum_i\rho_\delta(r_i)$ with

$$
\rho_\delta(r)=
\begin{cases}
r^2/2, & |r|\le\delta,\\
\delta(|r|-\delta/2), & |r|>\delta,
\end{cases}
\qquad
\psi_\delta(r)=\rho_\delta'(r)=\operatorname{clamp}(r,-\delta,\delta).
$$

The smooth gradient becomes $A^\top\psi_\delta(Ax-b)$. Since $\psi_\delta$
is 1-Lipschitz, the step $t\le1/\|A\|_2^2$ is unchanged. The conjugate is
$\rho_\delta^*(u)=u^2/2$ for $|u|\le\delta$ and $+\infty$ otherwise, so the
dual point $u=\psi_\delta(r)/s$ (with the same feasibility scale $s\ge1$ as
for least squares) stays in its domain and the gap is

$$
\sum_i\rho_\delta(r_i)+\lambda\|x\|_1+\frac{\|u\|^2}{2}+b^\top u .
$$

For least squares, $\psi(r)=r$ recovers the formula above. The dual-gap
tolerance scale is $2\sum_i\ell(b_i)$, which equals $\|b\|^2$ for least
squares; for Huber, outliers do not inflate it quadratically. The path
starts at $\lambda_{\max}=\|A^\top\psi_\delta(b)\|_\infty$ (positive part for
nonnegative coefficients; groupwise norms for group LASSO). ADMM is not
extended, because its $x$-update relies on a least-squares factorization.

## Huber penalty

Plain and block FISTA can replace $\lambda\|x\|_1$ (block: on $x$ only) by
$\lambda\sum_j\rho_\delta(x_j)$. The prox of $\tau\rho_\delta$ is

$$
\operatorname{prox}(v)=
\begin{cases}
v/(1+\tau), & |v|\le\delta(1+\tau),\\
v-\tau\delta\,\operatorname{sign}(v), & \text{otherwise},
\end{cases}
$$

clipped at zero for nonnegative coefficients (exact, since the problem is
separable and one-dimensional). It is zero only for $v=0$, so the penalty
does not select a support. The KKT condition is
$g_j+\lambda\psi_\delta(x_j)=0$, or $g_j\ge0$ at a nonnegative zero.

With $w=-A^\top u$, the conjugate of $\lambda\rho_\delta$ is
$w^2/(2\lambda)$ on $|w|\le\lambda\delta$ (for $x\ge0$, $w_+^2/(2\lambda)$
on $w_+\le\lambda\delta$). The dual point is scaled by
$s=\max(1,\|w\|_\infty/(\lambda\delta))$ (positive part when
nonnegative), and the gap is

$$
P(x)+\frac{\ell^*(u)+\sum_jw_j^2/(2\lambda)}{s^2}+\frac{b^\top u}{s}.
$$

At the optimum $w=\lambda\psi_\delta(x)$, so $s=1$ and the gap vanishes.
For $\delta\to0$, $(\lambda/\delta)\rho_\delta$ tends to $\lambda|x|$ up to a
constant, recovering LASSO. Since no finite $\lambda$ gives $x=0$ (unless
$A^\top\psi(b)=0$), there is no $\lambda_{\max}$; reported `kkt_ratio` uses
$\lambda\delta$, the largest penalty slope, as its scale.

## Disjoint group LASSO

For a partition $\mathcal G$ of the coefficient indices, Prism also solves

$$
\min_{x\in C}\frac12\|Ax-b\|_2^2
+\lambda\sum_{g\in\mathcal G}w_g\|x_g\|_2,
\qquad w_g>0,
$$

where $C=\mathbb R^n$ or $\mathbb R_+^n$. By default
$w_g=\sqrt{|g|}$; custom weights are accepted. With
$v_g=y_g-t[A^\top(Ay-b)]_g$, the group proximal step is

$$
x_g^+=\left(1-\frac{t\lambda w_g}{\|z_g\|_2}\right)_+z_g,
\qquad
z_g=\begin{cases}v_g,&C=\mathbb R^n,\\
(v_g)_+,&C=\mathbb R_+^n.
\end{cases}
$$

The result is zero when $z_g=0$. The nonnegative case clips *before*
group shrinkage. A selected group can contain small nonzero coefficients;
this is group sparsity, not within-group sparsity. Groups must be disjoint
and cover every column. FISTA supports CPU and CUDA arrays; ADMM is not
implemented for this objective.

## L1-ball-constrained least squares

Here the penalty is replaced by a budget $R$:

$$
\min_x\frac12\|Ax-b\|_2^2
\quad\text{subject to}\quad
\|x\|_1\leq R,\quad
x\in C.
$$

$C$ is the full space or the nonnegative orthant. The FISTA prox
becomes projection onto the feasible L1 ball, not soft thresholding
with a particular $\lambda$. Constrained ADMM uses the same split
idea with that projection as its $z$ step. The radius $R$ and a LASSO
$\lambda$ are different controls; their correspondence is
solution-dependent.

## Sparse plus smooth blocks

For $A\in\mathbb R^{m\times n}$ and $B\in\mathbb R^{m\times p}$, the
block model is

$$
\min_{x,z}\frac12\|Ax+Bz-b\|_2^2
+\lambda\|x\|_1+\frac{\gamma}{2}\|z\|_2^2,
$$

with optional coefficient nonnegativity. The $x$ block is sparse,
whereas the ridge penalty keeps $z$ smooth in coefficient magnitude.
A combined design $[A\ B]$ is built once for the reusable CPU FISTA
workspace. This is not a spatial smoothness constraint unless $B$
and the interpretation of $z$ make it one.

## Spatial regularizers

The squared graph-Laplacian model is

$$
\min_{x\in C}\frac12\|Ax-b\|_2^2
+\lambda\|x\|_1+\frac{\gamma}{2}\|Lx\|_2^2,
$$

where $L$ is the 2D nearest-neighbor graph Laplacian. The
$\|Lx\|_2^2$ term penalizes rapid local changes but can spread a
localized source. FISTA includes $\gamma L^\top Lx$ in the smooth
gradient.

The total-variation model is

$$
\min_{x\in C}\frac12\|Ax-b\|_2^2
+\lambda_1\|x\|_1+\lambda_{\rm TV}\operatorname{TV}(x).
$$

With forward spatial differences $Dx$, isotropic TV sums
$\sqrt{(D_xx)_j^2+(D_yx)_j^2}$ over pixels; anisotropic TV sums
the absolute components. PDHG uses a dual variable for the
nonsmooth TV term rather than squaring it. TV tends to preserve
edges while discouraging isolated local jumps.

## Posterior sampling at one λ

`sample_posterior` treats the penalized LASSO objective as a negative
log-density. With noise variance $\sigma^2$ per entry of $b$,

$$
\pi(x)=p(x\mid b)\;\propto\;
\exp\!\Bigl(-\frac{F_\lambda(x)}{\sigma^2}\Bigr)
=\underbrace{\exp\!\Bigl(-\frac{\|Ax-b\|_2^2}{2\sigma^2}\Bigr)}_{\text{Gaussian likelihood}}
\;\underbrace{\prod_{j}\exp\!\Bigl(-\frac{\lambda}{\sigma^2}|x_j|\Bigr)}_{\text{Laplace prior}},
$$

on $\mathbb R^n$, or on $\mathbb R_+^n$ for nonnegative coefficients, where
the prior becomes exponential with mean $\sigma^2/\lambda$. The mode of $\pi$
is the minimizer of $F_\lambda$, the LASSO solution, for every $\sigma^2$;
$\sigma^2$ only scales the width. Everything is conditional on $\lambda$ and
$\sigma^2$; neither is estimated.

**One-coefficient conditional.** Fix all coefficients except $x_i$, let
$a_i$ be column $i$ of $A$, $d_i=\|a_i\|_2^2$, and $r=b-Ax$ the current
residual. As a function of a new value $t$ for $x_i$,

$$
\tfrac12\|r-a_i(t-x_i)\|_2^2=\tfrac{d_i}{2}(t-c_i)^2+\text{const},
\qquad c_i=x_i+\frac{a_i^\top r}{d_i},
$$

so with $s_i^2=\sigma^2/d_i$ and $\tau_i=\lambda/d_i$ the conditional density
is

$$
\pi(t\mid x_{-i})\;\propto\;
\exp\!\Bigl(-\frac{(t-c_i)^2}{2s_i^2}-\frac{\tau_i|t|}{s_i^2}\Bigr).
$$

Completing the square separately for $t\ge0$ and $t\le0$ gives

$$
\pi(t\mid x_{-i})\;\propto\;
\begin{cases}
e^{-\lambda c_i/\sigma^2}\;\mathcal N(t;\,c_i-\tau_i,\,s_i^2), & t\ge0,\\[2pt]
e^{+\lambda c_i/\sigma^2}\;\mathcal N(t;\,c_i+\tau_i,\,s_i^2), & t\le0.
\end{cases}
$$

- *Nonnegative coefficients:* only the first branch exists, so
  $x_i\mid x_{-i}$ is $\mathcal N(c_i-\tau_i,s_i^2)$ truncated to $t\ge0$.
  Its untruncated mean $c_i-\tau_i$ is the argument of the coordinate-descent
  LASSO update $\max(c_i-\tau_i,0)$.
- *Signed coefficients:* a mixture of the two truncated normals. With $\Phi$
  the standard normal distribution function, the positive branch is chosen
  with probability $w_+/(w_++w_-)$,

$$
w_+=e^{-\lambda c_i/\sigma^2}\,\Phi\!\Bigl(\frac{c_i-\tau_i}{s_i}\Bigr),
\qquad
w_-=e^{+\lambda c_i/\sigma^2}\,\Phi\!\Bigl(-\frac{c_i+\tau_i}{s_i}\Bigr).
$$

  The weights are evaluated as logarithms with $\log\Phi(u)=
  \log\operatorname{erfc}(-u/\sqrt2)-\log2$, which stays finite far in the
  tails where $\Phi$ underflows; this is the only use of `SpecialFunctions`.
- *Zero column* ($d_i=0$): the data do not constrain $x_i$, and it is drawn
  from the prior, an exponential with mean $\sigma^2/\lambda$ (with a random
  sign when signed).

**Gibbs sampler.** One sweep draws $x_1,\dots,x_n$ in order, each from its
conditional given the current values of the others. Every such update leaves
$\pi$ invariant, so the sweep does, and the chain converges to $\pi$ from any
start. There is no proposal, step size, or rejection. Truncated normals are
drawn exactly: by rejection from the untruncated normal when at least half
its mass is allowed, and otherwise by Robert's (1995) shifted-exponential
rejection, which stays efficient arbitrarily far in the tail.

The sweep needs $a_i^\top r$ for every $i$. For sparse $A$, and for dense $A$
with $m\le n$, the residual $r$ is stored and updated by
$r\leftarrow r-(t-x_i)a_i$ after each coefficient: $O(\operatorname{nnz}(A))$
per sweep. For dense $A$ with $m>n$, $K=A^\top A$ is formed once and
$A^\top r$ is stored and updated by column $i$ of $K$: $O(n^2)$ per sweep. The
stored vector is recomputed from scratch every 100 sweeps to remove
accumulated rounding.

**Diagnostics.** Draws within a chain are autocorrelated. For each coefficient
Prism reports the split $\hat R$ (each chain is halved; $\hat R$ compares the
between-half variance of the means with the within-half variance and is near
1 when all halves sample the same distribution) and the effective sample size
$N_{\rm eff}=MN/\hat\tau$, where $\hat\tau=-1+2\sum_k(\hat\rho_{2k}+
\hat\rho_{2k+1})$ sums the estimated autocorrelations in pairs until a pair
is no longer positive (Geyer's initial positive sequence).

**Why coordinate-wise.** An alternative writes the Laplace prior as a normal
with exponentially distributed variance, integrates $x$ out, and runs a
gradient-based chain (MALA) on the $n$ log-variances. Each step then factors
a dense $\min(m,n)$-sized matrix, and the construction has no nonnegative
counterpart. On a test with $m=1500$, $n=6000$ that chain cost 0.13 s per
step and produced 0.008 effective draws per second for its worst coefficient,
against 5.8 for the Gibbs sampler, with matching posterior means. The price
of coordinate updates is slow joint movement of coefficients with strongly
overlapping columns; see [posterior sampling](posterior_sampling.md) for
measured costs and how to read the diagnostics.

## Preparing scientific data

For observations $y$, estimated background $c$, prior field $x_0$,
physical dictionary $D$, and covariance whitening operator $W$,
a typical algebraic preparation is

$$
b=W(y-c-Ax_0),\qquad A_{\rm solve}=WAD.
$$

Then Prism solves for coefficients $z$ in
$b\approx A_{\rm solve}z$ and the physical field is $x_0+Dz$.
If the measurement model uses differential observations, apply the
same differencing consistently to both $y$ and $A$. Prism does
not decide whether those modeling choices are justified.

Near-collinear columns can yield almost identical $Ax$ from very
different $x$. A solver reaching numerical optimality does not
make the field identifiable. Compare predictions, aggregates,
and solutions under nearby $\lambda$ values.

See [optimality](optimality.md) for the exact stop tests and
[performance](performance.md) for the cost of each algorithmic choice.
