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

**EWA projection (same session):** The 2D screen covariance is `Σ' = J W Σ Wᵀ Jᵀ`.
- `W` (world→camera) is linear, but the perspective divide (`x/z`, `y/z`) is **nonlinear**, so a Gaussian pushed through it isn't exactly a Gaussian.
- Fix: replace the divide with its **local linear approximation** — the **Jacobian `J`** (matrix of partial derivatives) evaluated at the splat center. The Gaussian is concentrated there, so a linear approx over that small region is good. That makes the whole path linear, and covariance rides through any linear map `A` as `A Σ Aᵀ`.
- Eigen-decompose the 2×2 `Σ'` → ellipse axes; size a billboard to ±3σ; fragment falloff = `exp(-½·dist²) · opacity`.

**Covariance itself:** how a cloud of points spreads — diagonal = per-axis variance, off-diagonal = tilt/correlation; an ellipsoid axis radius = √(eigenvalue) = √variance = the standard deviation (= the stored scale).

**Built:** the real 2D splat pipeline (`splat.wgsl`) with EWA projection + Gaussian falloff + alpha blending, rendered **unsorted** on purpose so the blending artifacts motivate sorting. `v` now cycles splats / ellipsoids / points.

**Depth sorting (same session):** "Over" blending is order-dependent (A-over-B ≠ B-over-A), so splats must be composited **back-to-front**. Opaque geometry sidesteps this with the depth buffer (nearest wins, order irrelevant); transparent splats accumulate many contributions per pixel and need explicit ordering. Sort key = camera-space depth; because moving the camera changes every splat's depth, it must re-sort constantly → the **performance bottleneck** (GPU radix sort is the scaling fix).

**Sorting, done right (counting sort):** A comparison sort (`Array.sort` with a comparator) is `O(n log n)` and pays a JS function call per comparison — the real bottleneck. **Quantize** each depth to a 16-bit bucket (one of 65,536 integer levels) and you can **counting-sort** instead: tally per bucket → prefix-sum to offsets → place each item (`O(n)`, zero comparisons). Quantizing to integer keys is the *precondition* that unlocks counting/radix sort. Radix = counting sort done a few bits at a time, for keys too big for one bucket array. This is exactly what web viewers (antimatter15, GaussianSplats3D) ship; the two architectures are (A) global sort + billboard blend [ours] and (B) tile-based per-tile sort + compute rasterizer [CUDA/gsplat]. Implemented the 16-bit counting sort; it re-sorts + re-uploads the instance buffer only when the camera moves.

**Also learned:** EWA = **Elliptical Weighted Average** (Heckbert 1989 texture filtering → Zwicker et al. surface splatting → 3DGS); the "elliptical" is the screen-space `Σ'` footprint, the "weighted average" is the Gaussian falloff.

**Real data (`.ply` loading):** Wrote a binary `.ply` parser. Unlike `.splat`, a 3DGS `.ply` stores the RAW trainable values, so the parser applies the activations on the way in: **scale→exp, opacity→sigmoid, rotation→normalize**, and **f_dc → 0.5 + C0·f_dc** for base color (`C0` = degree-0 SH basis constant). Loaded `luigi.ply` (14,526 splats, object-centric) — a real capture through the whole pipeline. This file is **DC-only** (no `f_rest`), so it shows base color but no view-dependent SH.

**Open questions to revisit:**
- How is higher-order SH evaluated per view direction to make color view-dependent, and what full-SH scene do we test it on? (Phase 1 — last concept; luigi is DC-only, so we need a fabricated or full-SH example to *see* it.)
