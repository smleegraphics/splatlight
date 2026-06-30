// In-memory representation of a Gaussian cloud — the DECODED form, after all the
// .ply activation functions (exp/sigmoid/normalize) are already applied. Both the
// file parser (splat-parser.ts) and the synthetic generator below produce this
// same SplatCloud, so the renderer never cares where the data came from.
//
// [plumbing] structure + generator. The *meaning* of each field is the concept
// (see the format walkthrough / LEARNING_LOG): position, covariance (scale +
// rotation), opacity, color.

export interface SplatCloud {
  count: number;
  positions: Float32Array; // 3 per splat: x, y, z
  scales: Float32Array; // 3 per splat: LINEAR axis lengths (already exp'd)
  rotations: Float32Array; // 4 per splat: quaternion (w, x, y, z), normalized
  opacities: Float32Array; // 1 per splat: [0, 1]
  colors: Float32Array; // 3 per splat: base RGB from the SH DC term, [0, 1]
}

/**
 * A controllable, object-centric test scene: a thin spherical shell of Gaussians,
 * colored by direction so structure is obvious when you orbit. We only need
 * position + color to render points (Phase 0); scale/rotation/opacity are filled
 * with sane values so the structure is complete for Phase 1+.
 */
export function makeSyntheticCloud(count = 40000, radius = 1.4): SplatCloud {
  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const rotations = new Float32Array(count * 4);
  const opacities = new Float32Array(count);
  const colors = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    // A uniformly-distributed direction on the unit sphere (normalize a 3D
    // Gaussian sample — standard trick for even coverage).
    let x = randn();
    let y = randn();
    let z = randn();
    const inv = 1 / (Math.hypot(x, y, z) || 1);
    x *= inv;
    y *= inv;
    z *= inv;

    const r = radius * (0.97 + 0.03 * Math.random()); // slight shell thickness
    positions[i * 3 + 0] = x * r;
    positions[i * 3 + 1] = y * r;
    positions[i * 3 + 2] = z * r;

    // Color from direction: x,y,z in [-1,1] mapped to RGB in [0,1].
    colors[i * 3 + 0] = 0.5 + 0.5 * x;
    colors[i * 3 + 1] = 0.5 + 0.5 * y;
    colors[i * 3 + 2] = 0.5 + 0.5 * z;

    scales[i * 3 + 0] = 0.01;
    scales[i * 3 + 1] = 0.01;
    scales[i * 3 + 2] = 0.01;
    rotations[i * 4 + 0] = 1; // identity quaternion (w=1, x=y=z=0)
    opacities[i] = 0.9;
  }

  return { count, positions, scales, rotations, opacities, colors };
}

/** Box-Muller: one sample from a standard normal distribution. */
function randn(): number {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Flatten a cloud into one interleaved [x,y,z, r,g,b] array for the point
 * pipeline. (Phase 1 will replace this with per-splat instanced data.)
 */
export function cloudToPointVertices(cloud: SplatCloud): Float32Array {
  const out = new Float32Array(cloud.count * 6);
  for (let i = 0; i < cloud.count; i++) {
    out[i * 6 + 0] = cloud.positions[i * 3 + 0];
    out[i * 6 + 1] = cloud.positions[i * 3 + 1];
    out[i * 6 + 2] = cloud.positions[i * 3 + 2];
    out[i * 6 + 3] = cloud.colors[i * 3 + 0];
    out[i * 6 + 4] = cloud.colors[i * 3 + 1];
    out[i * 6 + 5] = cloud.colors[i * 3 + 2];
  }
  return out;
}
