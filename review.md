# Review Notes: Corrections Made from the Initial Version

This note summarizes the main changes made from the initial version of the shock-tube notebook. The purpose is educational: the goal is not only to obtain a working code, but also to clarify which parts of the numerical method and implementation needed correction.

------

## 1. Physics / Numerical Method Corrections

### 1.1 Wave-speed estimate in HLL and HLLC solvers

The shock wave-speed correction factor was incorrect in the original code.

For pressure-based estimates of the HLL/HLLC signal speeds, when the star-region pressure is larger than the initial pressure on one side, the wave is treated as a shock. The correction factor should be

$$
q_K =
\sqrt{
1 + \frac{\gamma+1}{2\gamma}
\left(
\frac{p_*}{p_K} - 1
\right)
}
=
\sqrt{
\frac{(\gamma+1)p_*/p_K + (\gamma-1)}{2\gamma}
}.
$$

Here, $K=L$ or $K=R$.

Therefore, the shock wave speeds should be written as

$$
S_L = u_L - c_L q_L,
\qquad
S_R = u_R + c_R q_R.
$$

In the original code, the expression used

$$
(\gamma+1)p_*/p_K + (\gamma+1),
$$

but the second term should be

$$
\gamma - 1,
$$

not $\gamma + 1$.

So the corrected code uses

```python
np.sqrt(((gamma + 1)*p_star/pK + (gamma - 1))/(2.0*gamma))
```

instead of

```python
np.sqrt(((gamma + 1)*p_star/pK + (gamma + 1))/(2.0*gamma))
```

This correction was applied consistently in the HLL, HLLC, MUSCL, and WENO-related flux calculations.

------

### 1.2 Difference between the pressure-based estimate and the simple Davis estimate

A simpler HLL estimate is

$$
S_L = \min(u_L-c_L,\ u_R-c_R),
\qquad
S_R = \max(u_L+c_L,\ u_R+c_R).
$$

This is robust and simple, but it can be more diffusive.

The pressure-based estimate uses $p_*$ to distinguish shocks from rarefactions. If $p_* > p_K$, the wave is treated as a shock and the signal speed is increased by the factor $q_K$. If $p_* \le p_K$, the wave is treated as a rarefaction and $q_K=1$, giving the usual acoustic speed $u_K \pm c_K$.

This is why the corrected code uses

```python
if p_star <= pK:
    qK = 1.0
else:
    qK = np.sqrt(1.0 + (gamma + 1.0)/(2.0*gamma)*(p_star/pK - 1.0))
```

before computing the outer wave speeds.

------

### 1.3 HLLC flux branch for the right-going region

The HLLC flux must select among four possible regions:

$$
F_{\mathrm{HLLC}} =
\begin{cases}
F_L, & 0 \le S_L, \\
F_L^*, & S_L \le 0 \le S_M, \\
F_R^*, & S_M \le 0 \le S_R, \\
F_R, & S_R \le 0.
\end{cases}
$$

The original implementation incorrectly returned a right-star-state flux in the final branch, where the correct answer is the pure right physical flux $F_R$.

The corrected logical structure is

```python
if SL >= 0.0:
    return FL
elif SM >= 0.0:
    return FL_star
elif SR > 0.0:
    return FR_star
else:
    return FR
```

This is important for cases where all waves move to the left. The error may not be very visible in the standard Sod problem, but it becomes significant for reversed or strongly moving Riemann problems.

------

### 1.4 Contact-wave speed in HLLC

The contact-wave speed $S_M$ was rewritten in a more standard Rankine-Hugoniot form:

$$
S_M =
\frac{
p_R - p_L
+ \rho_L u_L (S_L-u_L)
- \rho_R u_R (S_R-u_R)
}{
\rho_L (S_L-u_L)
- \rho_R (S_R-u_R)
}.
$$

This makes the implementation easier to compare with standard HLLC formulas and reduces the chance of algebraic mistakes in the flux expressions.

------

## 2. Code / Implementation Corrections

### 2.1 Consistent ghost-cell index: `fidx`

The initial notebook mixed different assumptions about the number of ghost cells.

Some parts implicitly assumed

```python
fidx = 2
```

while higher-order reconstruction, especially WENO5, requires three ghost cells on each side:

```python
fidx = 3
```

The corrected version uses `fidx=3` consistently where needed.

The physical cell range is now treated as

```python
ist = fidx
ien = fidx + ngrids
```

and physical-cell arrays are sliced using

```python
array[ist:ien]
```

rather than hard-coded slices such as

```python
array[2:-2]
```

This change is important because hard-coded slices become incorrect as soon as the number of ghost cells changes.

------

### 2.2 Exact solution and plotting used inconsistent array lengths

One runtime error was

```text
ValueError: x and y must have same first dimension
```

For example, the coordinate array had length 64, while the solution array had length 66. This happened because the numerical solution still included ghost cells while the coordinate array represented only physical cells.

The corrected plotting logic uses only the physical region:

```python
x = cel_center[ist:ien]

plt.plot(x, rho[ist:ien])
plt.plot(x, u[ist:ien])
plt.plot(x, p[ist:ien])
```

The exact solution is also plotted over the same physical-cell range:

```python
plt.plot(x, rho_exact[ist:ien], "--")
plt.plot(x, u_exact[ist:ien], "--")
plt.plot(x, p_exact[ist:ien], "--")
```

This makes the numerical and exact solutions comparable on the same grid.

------

### 2.3 `exactsolution()` needed `fidx`

The exact-solution routine creates arrays with ghost cells, so it also needs to know how many ghost cells are used. In the initial version, some calls to `exactsolution()` did not pass `fidx`, which caused errors or inconsistent indexing.

The corrected calls explicitly pass `fidx`:

```python
rho_exact, u_exact, p_exact = exactsolution(
    L=L,
    ngrids=ngrids,
    fidx=fidx,
    tout=tout,
    rhoL=rhoL,
    uL=velL,
    pL=preL,
    rhoR=rhoR,
    uR=velR,
    pR=preR,
)
```

This avoids hidden assumptions about the number of ghost cells.

------

### 2.4 Coordinate definition in the exact solution

The coordinate spacing for the physical solution should be based on the number of physical cells:

```python
dx = L / ngrids
```

not on the total number of cells including ghost cells.

The corrected physical coordinate is

```python
xpos = -0.5*L + (i - fidx + 0.5)*dx
```

for a cell index `i` in the full array including ghost cells.

This makes the exact solution align with the numerical cell centers.

------

### 2.5 WENO reconstruction returned scalars instead of arrays

The original WENO routine overwrote `uL` and `uR` inside the loop and returned only the last reconstructed values. This is incorrect because WENO reconstruction must return left and right interface states at all relevant interfaces.

The corrected version initializes arrays:

```python
uL = np.zeros(ntot)
uR = np.zeros(ntot)
```

and assigns each reconstructed interface value inside the loop:

```python
uL[i] = ...
uR[i] = ...
```

Then the function returns the full arrays:

```python
return uL, uR
```

------

### 2.6 WENO loop bounds were too wide

The WENO stencil accesses values such as

```python
u[i-2], u[i-1], u[i], u[i+1], u[i+2], u[i+3]
```

Therefore the loop range must avoid indexing beyond the allocated array.

The corrected loop uses

```python
for i in range(ist-1, ien):
```

instead of a wider range such as

```python
for i in range(ist-1, ien+1):
```

This prevents out-of-bounds errors for the rightmost interface.

------

### 2.7 Removed duplicated HLLC flux logic in `calculateflux()`

The HLLC flux formula appeared in more than one place. This makes the code harder to maintain, because a bug fixed in one location may remain in another.

The corrected version uses the main `hllc()` function inside `calculateflux()`:

```python
for i in range(ist-1, ien):
    frho[i], fmom[i], fe[i] = hllc(
        rhoL[i], uL[i], pL[i],
        rhoR[i], uR[i], pR[i]
    )
```

This ensures that the first-order HLLC solver and the higher-order reconstruction version use the same Riemann solver.

------

### 2.8 Avoided hard-coded ghost-cell assumptions in animation functions

The animation functions originally used full arrays or hard-coded slices. They were updated to accept `fidx` explicitly and to plot only the physical region.

For example, instead of plotting

```python
plt.plot(cel_center, rho)
```

the corrected version uses

```python
ist = fidx
ien = fidx + ngrids
x = cel_center[ist:ien]

plt.plot(x, rho[ist:ien])
```

This prevents dimension mismatches and makes the visualization independent of the number of ghost cells.

------

## 3. Educational Summary

The most important lesson from these corrections is that the physics and indexing must be consistent throughout the whole code.

From the physics side:

- HLL and HLLC require careful signal-speed estimates.
- Shock and rarefaction waves should be treated differently when using pressure-based wave-speed estimates.
- HLLC has four flux regions, and the final branch must return the pure right-state flux $F_R$, not a star-state flux.

From the implementation side:

- The number of ghost cells should not be hard-coded.
- Physical cells should always be selected using `ist:ien`.
- Exact solutions and numerical solutions must be plotted on the same physical grid.
- Reconstruction routines such as WENO must return arrays of interface states, not a single scalar.
- Duplicated flux formulas should be avoided; the same Riemann solver function should be reused.

These changes make the code more reliable, easier to review, and more useful as a learning tool for finite-volume methods for the Euler equations.
