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
- How exactly does `Σ = R S Sᵀ Rᵀ` make an ellipsoid? (Phase 1, [concept])
- How do we get a usable surface normal out of a blob that's only described by a covariance? (Phase 2)
