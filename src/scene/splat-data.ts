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
export function makeSyntheticCloud(count = 12000, radius = 1.4): SplatCloud {
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

    // Flatten each splat into a disk lying tangent to the sphere: two wide axes
    // in the surface, one thin axis (local z) along the outward normal (x,y,z).
    // The thin axis IS the surface normal here — a preview of Phase 2.
    scales[i * 3 + 0] = 0.05; // tangent
    scales[i * 3 + 1] = 0.05; // tangent
    scales[i * 3 + 2] = 0.006; // thin (normal direction)

    // Quaternion (w,x,y,z) rotating local +z (0,0,1) onto the outward dir (x,y,z).
    // axis = cross((0,0,1), d) = (-y, x, 0);  w = 1 + dot((0,0,1), d) = 1 + z.
    let qw = 1 + z;
    let qx = -y;
    let qy = x;
    let qz = 0;
    if (qw < 1e-6) {
      // d points at -z (antiparallel): 180° about x.
      qw = 0;
      qx = 1;
      qy = 0;
      qz = 0;
    }
    const qInv = 1 / Math.hypot(qw, qx, qy, qz);
    rotations[i * 4 + 0] = qw * qInv;
    rotations[i * 4 + 1] = qx * qInv;
    rotations[i * 4 + 2] = qy * qInv;
    rotations[i * 4 + 3] = qz * qInv;

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

/**
 * Pack a cloud into per-instance data for the ellipsoid debug pipeline:
 * 13 floats per splat = center(3) + scale(3) + quaternion(4) + color(3).
 * The quaternion is reordered from stored (w,x,y,z) to (x,y,z,w) so the shader
 * can read `q.w` as the real part.
 */
export function cloudToInstanceData(cloud: SplatCloud): Float32Array {
  const FLOATS = 13;
  const out = new Float32Array(cloud.count * FLOATS);
  for (let i = 0; i < cloud.count; i++) {
    const o = i * FLOATS;
    out[o + 0] = cloud.positions[i * 3 + 0];
    out[o + 1] = cloud.positions[i * 3 + 1];
    out[o + 2] = cloud.positions[i * 3 + 2];
    out[o + 3] = cloud.scales[i * 3 + 0];
    out[o + 4] = cloud.scales[i * 3 + 1];
    out[o + 5] = cloud.scales[i * 3 + 2];
    out[o + 6] = cloud.rotations[i * 4 + 1]; // x
    out[o + 7] = cloud.rotations[i * 4 + 2]; // y
    out[o + 8] = cloud.rotations[i * 4 + 3]; // z
    out[o + 9] = cloud.rotations[i * 4 + 0]; // w
    out[o + 10] = cloud.colors[i * 3 + 0];
    out[o + 11] = cloud.colors[i * 3 + 1];
    out[o + 12] = cloud.colors[i * 3 + 2];
  }
  return out;
}
