# Learning Log

Appended after each session: concept covered → key insight in plain language → open questions.

---

## Session 1 — Phase 0 start: splat anatomy + scaffold

**Concept covered:** Splat anatomy (syllabus #1) — what a single 3D Gaussian stores.

**Key insight (plain language):** A splat scene is not surfaces — it's a cloud of millions of fuzzy, semi-transparent, colored ellipsoids, smeared on screen back-to-front like airbrush strokes. Each blob stores only four things:
1. **Position** `μ = (x,y,z)` — the blob's center.
2. **Covariance** — its shape/orientation as an ellipsoid, stored as **3 scale values + a 4-number rotation quaternion** (not a raw matrix, so training keeps it physically valid). The matrix `Σ = R S Sᵀ Rᵀ` is rebuilt on the fly.
3. **Opacity** `α` — how solid vs. see-through (this is what makes back-to-front blending work).
4. **Color via spherical harmonics** — the **DC term** (3 numbers) is the base color; higher coefficients (up to 45 more) add view-dependent sheen/glints.

Why this is "enough": the blobs cooperate like pointillism dots — no single one is right, but a million overlapping soft ellipsoids approximate any shape + light pattern. Gaussians specifically because they're smooth (no aliasing), stay Gaussian when projected to 2D (cheap to render), and are differentiable (trainable).

**Why it matters for our lighting goal:** the DC color has the *capture lighting baked in*, which is exactly why splats aren't relightable by default — Phase 2 has to recover surface normals to fake a relight.

**Plumbing built:** Vite + TS + WebGPU scaffold — device/canvas init, orbit camera, render loop, reference grid + XYZ axes so the camera is visibly working.

**Open questions to revisit:**
- How exactly does `Σ = R S Sᵀ Rᵀ` make an ellipsoid? (Phase 1, [concept]) — **answered in Session 2**
- How do we get a usable surface normal out of a blob that's only described by a covariance? (Phase 2)

---

## Session 2 — Phase 1 start: the 3D covariance

**Concept covered:** 3D covariance from scale + rotation (syllabus #2), plus refreshers on eigenvectors/eigenvalues, spherical harmonics, and `.ply` vs `.splat`.

**Key insight (plain language):** A splat's shape is one 3×3 matrix, `Σ = R S² Rᵀ`.
- Build it as a transform `M = R·S` that maps a unit sphere → the ellipsoid: `S = diag(sx,sy,sz)` scales it (the axis radii), `R` (from the quaternion) rotates it. The covariance is `Σ = M Mᵀ = R S² Rᵀ`.
- **Eigenvectors of `Σ`** = the ellipsoid's axis directions (columns of `R`); **eigenvalues** = `sx², sy², sz²`, so axis radius = √eigenvalue = the scale. (An eigenvector is a direction a matrix only scales, not rotates; the eigenvalue is the scale factor.)
- **Order matters:** scale in the *local* frame first, then rotate (`R` on the outside). Rotate-first would flatten every splat along fixed *world* axes and discard its orientation — and rotating a sphere does nothing, so the rotation would be wasted.
- **Shape from scales:** two big + one tiny = flat disk; one big + two tiny = rod; all equal = sphere.
- **Phase 2 preview:** the shortest axis (smallest eigenvalue) of a flattened, surface-aligned splat is the surface **normal**.

**Also clarified:** SH stores view-dependent *color* (bands = angular detail, DC = base color), and it's the *baked appearance* that makes relighting hard — not a lighting helper. `.ply` keeps the higher SH coeffs (view-dependent highlight); `.splat` drops them (flat color). Neither format carries a usable normal (3DGS writes zeros), which is why Phase 2 computes them from `Σ`.

**Debug viz built:** each splat drawn as its real 3D ellipsoid (vertex shader does `world = R * (spherePos * scale) + center`), synthetic cloud reshaped into surface-tangent flattened disks, `v` toggles ellipsoids vs. flat points.

**Open questions to revisit:**
- How does the 3D covariance get projected to a 2D screen-space ellipse, and why is a Jacobian involved? (Phase 1, EWA — next)
