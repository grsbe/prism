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
