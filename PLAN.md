# Gaussian Splat Renderer with Dynamic Lighting — Build & Learning Plan

> **Read this first, Claude Code.** This is a *learning project*. I don't need to write the code myself — **you write all of it.** My goal is to come out the other side actually **understanding how 3D Gaussian splatting works**: the representation, the projection math, the sorting, and how lighting gets layered on top. A working, portfolio-worthy renderer is the secondary goal. If I end up with a renderer I can't explain, we've failed, even if it runs. So: build freely, but **your real job is to make sure I understand what you built and why.**

Drop this file in the repo root as `PLAN.md`. Keep a short `CLAUDE.md` for conventions and a `LEARNING_LOG.md` for what I've learned (see §6).

---

## 0. How to work with me (collaboration protocol)

You're writing the renderer. I'm here to understand it. The thing that matters is **depth of explanation**, and it's not uniform — some parts are the actual ideas of splatting and deserve real teaching; others are generic plumbing I can skim. Sort everything you build into one of these:

**Deep-explain — these ARE Gaussian splatting.** Build them, then teach them properly: a primer on *why* first, then a walk through what the code does (line by line for the gnarly bits). Don't let me skip past these.
- What the splat format stores and why that's enough
- Building the 3D covariance from scale + rotation
- Projecting the 3D covariance to a 2D screen-space ellipse (EWA / the Jacobian)
- Evaluating spherical harmonics for color
- Why depth sorting is required and the alpha-compositing order
- Estimating per-splat normals from the covariance
- The shading / lighting math

**Skim — generic plumbing.** Write it, tell me in one line what it does, move on. Don't make me study buffer-binding ceremony as if it's the interesting part.
- Vite + TS + WebGPU setup, device/canvas init
- `.ply` / `.splat` byte parsing
- GPU buffer / bind-group boilerplate
- Tweakpane UI wiring, camera controls, deploy config

**Rules for every session:**
1. **Explain before you build.** Before each new *deep-explain* piece, give me a 3–5 sentence primer on what it does and why. Lead with the "why" (why back-to-front blending, why EWA projection, why splats can't be relit out of the box).
2. **Narrate after you build.** Once a deep-explain piece is written, walk me through it. Use analogies, and feel free to drop ASCII sketches or diagrams in comments when geometry is involved.
3. **Separate signal from noise.** Always tell me whether what you just wrote is a core concept or plumbing, so I know where to spend attention.
4. **Check my understanding at each milestone.** Ask me to explain the key concept back in my own words, or pose one diagnostic question. Don't move on until it lands. If I'm wrong, re-teach a different way — don't just restate.
5. **Build debug visualizations,** not just final output — point cloud → 2D ellipses → normals-as-RGB — so I can *see* each concept working. These are often more instructive than the explanation.
6. **Go at my pace.** If I say "wait, explain that" or ask "why," stop and go deeper before continuing. Never rush past a question to hit a milestone.
7. **Keep `LEARNING_LOG.md`.** After each session, append: concept covered, the key insight in plain language, and any open question to revisit.

---

## 1. The concepts I'm here to learn (the syllabus)

By the end I should be able to explain each of these unprompted. This is the learning spine the phases hang off — when a milestone touches one, make sure it actually lands:

1. **Splat anatomy** — what a 3D Gaussian stores (position, covariance via scale+rotation, opacity, SH color) and why that's enough to represent a scene.
2. **Covariance & the ellipsoid** — how scale + a rotation quaternion compose into a 3×3 covariance matrix, and why that's an ellipsoid.
3. **EWA projection** — how a 3D Gaussian becomes a 2D screen-space ellipse, and why a Jacobian (a local linear approximation of the projection) is involved.
4. **Spherical harmonics** — how view-dependent color is stored and evaluated.
5. **Alpha compositing & sorting** — why transparency forces back-to-front ordering, and why sorting is the performance bottleneck.
6. **Differentiable rendering (conceptually)** — how the *training* side optimizes the blobs. I'm not implementing training, but I should understand how a pile of photos becomes this file.
7. **Normals from covariance** — why the shortest ellipsoid axis approximates a surface normal, and the sign-ambiguity problem.
8. **Shading models** — Lambert / Blinn-Phong / ambient, albedo vs. baked color, and why this relight is approximate.

---

## 2. Stack decisions

| Choice | Pick | Why |
|---|---|---|
| Language | TypeScript | Type safety once buffers/layouts get hairy |
| Bundler | Vite | Instant HMR, zero-config TS |
| GPU API | **WebGPU (WGSL)** | Compute shaders make sort *and* lighting clean; baseline in all major browsers as of 2026 |
| Fallback | WebGL2 (optional, later) | Only if I want max reach |
| Math | `wgpu-matrix` | Lightweight mat4 / quaternion helpers |
| UI controls | `tweakpane` | Fast sliders/toggles for lights |
| Deploy | Vercel or GitHub Pages | Static, free, shareable URL |

**On references:** the clearest implementations to learn from (`kishimisu/Gaussian-Splatting-WebGL`, `antimatter15/splat`) are WebGL. When we hit a deep-explain concept, it's worth pointing me at the matching part of `kishimisu` (it's commented to map back to the original CUDA reference) so I can see a second version of the same idea. `Scthe/gaussian-splatting-webgpu` is the closest WebGPU reference.

---

## 3. The lighting approach (the crux — primer me before Phase 2)

Splats store **view-dependent color** (spherical harmonics) with the capture lighting baked in — that's *why* they're not relightable by default. To light them dynamically you have to recover enough surface info to shade them. Before building this, give me the full conceptual primer; then build it and walk me through it:

1. **Per-splat normal from the covariance.** Surface-aligned splats are flattened; the **shortest axis** approximates the normal. (Make sure I understand *why* before moving on.)
2. **Disambiguate the sign.** The shortest axis is a line, not a direction — flip normals to face the camera or enforce neighbor consistency. Quality gate.
3. **Base color as albedo.** Use the SH DC term as an albedo proxy; optional "delight" slider.
4. **Shade it.** Lambert + Blinn-Phong + ambient, driven by user lights:
   ```wgsl
   let N = normalize(splatNormal);
   let L = normalize(lightPos - worldPos);
   let V = normalize(cameraPos - worldPos);
   let H = normalize(L + V);
   let diff = max(dot(N, L), 0.0);
   let spec = pow(max(dot(N, H), 0.0), shininess);
   let lit  = albedo * (ambient + diff * lightColor) + spec * specColor;
   ```
   Walk me through each line of this — it's small but it's where the lighting actually happens.
5. **Optional: HDRI image-based lighting** for ambient irradiance + cheap reflections.

**Honest framing:** original lighting is baked in, so this is an *approximate, artistic* relight, not physical delighting. That's fine — a draggable sun reshaping a scanned object is the wow factor. Full PBR decomposition is a stretch goal (§5).

---

## 4. Phased build plan (each phase = build tasks + a learning checkpoint)

Don't advance until the previous **learning checkpoint** passes — not just the build milestone. Tags: **[concept]** = deep-explain, **[plumbing]** = skim.

### Phase 0 — Scaffolding & data path
- [ ] Vite + TS + WebGPU init, orbit camera, render loop **[plumbing]**
- [ ] `.ply` / `.splat` parser → typed arrays → GPU buffers **[plumbing, but explain the format = concept]**
- [ ] Render each splat as a flat-colored point
- [ ] **Build milestone:** points on screen.
- [ ] **Learning checkpoint:** I can explain every field stored per Gaussian and why.

### Phase 1 — Core splat rasterization
- [ ] Billboard each Gaussian into an instanced quad **[plumbing]**
- [ ] Build 3D covariance from scale+rotation **[concept]**
- [ ] Project to 2D screen-space covariance (EWA) **[concept]**
- [ ] Fragment shader: Gaussian falloff × opacity → alpha; alpha blend **[concept]**
- [ ] Depth sort back-to-front (CPU + worker first) **[concept — explain the why]**
- [ ] Evaluate SH for view-dependent color **[concept]**
- [ ] **Build milestone:** photorealistic, correctly-blended, orbitable scene.
- [ ] **Learning checkpoint:** I can explain the covariance → 2D-ellipse projection and why sorting is required.

### Phase 2 — Dynamic lighting (the differentiator)
- [ ] Per-splat normal from covariance shortest axis **[concept]**
- [ ] Debug view: normals as RGB **[plumbing — but show me, it's instructive]**
- [ ] Normal sign disambiguation **[concept]**
- [ ] Shading model (Lambert + Blinn-Phong + ambient) **[concept]**
- [ ] Tweakpane lights: draggable point light + directional sun (dir/color/intensity), ambient, baked-vs-relit toggle **[plumbing]**
- [ ] *(Optional)* HDRI image-based lighting
- [ ] **Build milestone:** drag a light, scene responds. **The hero feature.**
- [ ] **Learning checkpoint:** I can explain where normals come from and why this relight is approximate.

### Phase 3 — Polish & portfolio
- [ ] Lighting presets; tone mapping/exposure; optional bloom
- [ ] Turntable recorder for the demo reel
- [ ] Perf pass: GPU radix sort, frustum culling, optional LOD **[concept if I want to learn GPU sorting]**
- [ ] Deploy + write a technical breakdown (baked-vs-relit comparison shots sell it)
- [ ] **Build milestone:** hosted demo + writeup.
- [ ] **Learning checkpoint:** I can give a 5-minute whiteboard explanation of the whole pipeline.

---

## 5. Hard parts / risks

- **Sorting performance** — the perennial bottleneck. CPU-in-a-worker until it hurts; GPU radix sort is the fix. Don't gold-plate early.
- **Normal quality** — covariance normals are noisy on rounder Gaussians and noise shows up as lighting artifacts. Biggest risk for the lighting feature. Object-centric test scenes help a lot.
- **Approximate relight** — can't perfectly remove baked lighting; frame as stylized.
- **WebGPU cold-start** — async shader compile (>200ms first frame). Cache pipelines, warm during load.

---

## 6. Repo conventions & learning artifacts

- `CLAUDE.md`: "TypeScript, WebGPU/WGSL, no 3D framework, wgpu-matrix, Tweakpane, commit per milestone. This is a learning project — you write the code; follow the collaboration protocol in PLAN.md §0 and make sure I understand the **[concept]** parts."
- `LEARNING_LOG.md`: appended after each session — concept covered, key insight in plain language, open questions.
- One phase per branch/PR; commit per milestone.

**Suggested first prompts:**
1. "Give me the §1 primer on splat anatomy, then scaffold a Vite + TS + WebGPU project with an orbit camera and a cleared canvas. Tell me which parts are plumbing vs concept."
2. "Explain the `.splat`/`.ply` Gaussian format field by field, then write the parser. I don't need to write it — just make sure I understand what each field is for."
3. "Render each Gaussian as an instanced billboard, flat-colored from the SH DC term — no projection yet. Then quiz me on what's stored per splat."
4. *(Phase 1)* "Primer me on the 3D covariance and the 2D projection, write the WGSL, then walk me through it line by line. This one's a core concept — go slow."

---

## 7. Test assets

- **Object-centric capture, not a room** — flatter, cleaner normals; far more dramatic relighting.
- **Get a splat:** train with Nerfstudio `splatfacto` (needs an NVIDIA GPU/cloud box), or grab a public `.ply` to start immediately.
- **HDRI:** free `.hdr` from Poly Haven.

---

## 8. References

**Reference renderers**
- `kishimisu/Gaussian-Splatting-WebGL` — WebGL2, commented to map to the original CUDA reference. Best for learning the math.
- `antimatter15/splat` — minimal WebGL, no deps, CPU sort in a worker.
- `Scthe/gaussian-splatting-webgpu` — WebGPU + compute sort, with a companion math blog post.

**Papers**
- Kerbl et al. 2023 — *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (the foundation).
- *GaSLight* (ICCV 2025), *GBake* (SIGGRAPH 2025) — splats + lighting / probe baking.
- *ROGR: Relightable 3D Objects using Generative Relighting* (2025) — for the stretch PBR direction.

**Tracking**
- `Lee-JaeWon/2025-Arxiv-Paper-List-Gaussian-Splatting` — updated daily.
