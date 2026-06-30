# CLAUDE.md — conventions for this repo

**This is a learning project.** I (Claude) write all the code; the human's goal is to *understand* 3D Gaussian splatting, not to type it. Follow the collaboration protocol in `PLAN.md §0`:

- For every chunk, label it **[concept]** (an actual idea of splatting — teach it: *why* first, then walk the code) or **[plumbing]** (boilerplate — one line and move on).
- Explain before building, narrate after, build debug visualizations, and don't pass a **learning checkpoint** until the human can play the idea back.
- After each session, append to `LEARNING_LOG.md`: concept covered, key insight in plain language, open questions.

## Stack
- **TypeScript**, strict mode. No 3D framework (no three.js) — we build the pipeline by hand so the concepts stay visible.
- **WebGPU + WGSL** for rendering and compute. **Vite** dev server / bundler.
- **wgpu-matrix** for mat4 / quaternion math. **Tweakpane** for light UI (added in Phase 2).
- WGSL shaders live in `src/shaders/*.wgsl`, imported as strings via Vite `?raw`.

## Layout
- `src/main.ts` — entry: init, scene wiring, render loop.
- `src/gpu/` — WebGPU device/context/buffer plumbing.
- `src/camera/` — orbit camera.
- `src/scene/` — geometry + (later) splat data.
- `src/shaders/` — `.wgsl` source.

## Workflow
- `npm run dev` — Vite dev server with HMR.
- `npm run build` — type-check (`tsc`) + production bundle.
- One phase per branch; **commit per milestone** with a message naming the milestone.
- Requires a WebGPU-capable browser (Chrome/Edge/Safari 2024+, or Firefox with WebGPU enabled).
