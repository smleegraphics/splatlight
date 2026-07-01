// [plumbing, but the format is the concept] Parser for a 3D Gaussian Splatting
// `.ply` (binary little-endian). Unlike `.splat`, this stores the RAW trainable
// values, so we apply the activation functions on the way in:
//   scale  → exp()          (stored as log)
//   opacity→ sigmoid()      (stored pre-sigmoid)
//   rot    → normalize()    (quaternion, w,x,y,z = rot_0..3)
//   f_dc   → 0.5 + C0·f_dc  (SH degree-0 basis → base RGB)
// Higher-order SH (f_rest_*) is present in full files but ignored here for now;
// luigi.ply happens to be DC-only.

import type { SplatCloud } from './splat-data';

const SH_C0 = 0.28209479177387814; // degree-0 spherical harmonic basis constant

const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));

export function parsePly(buffer: ArrayBuffer): SplatCloud {
  const bytes = new Uint8Array(buffer);

  // --- header (ASCII) ---
  const headerText = new TextDecoder('ascii').decode(
    bytes.subarray(0, Math.min(bytes.length, 20000)),
  );
  const endMarker = headerText.indexOf('end_header');
  if (endMarker < 0) throw new Error('Not a PLY file (no end_header).');
  const dataStart = headerText.indexOf('\n', endMarker) + 1;

  let count = 0;
  const props: string[] = [];
  for (const raw of headerText.slice(0, endMarker).split('\n')) {
    const line = raw.trim();
    if (line.startsWith('element vertex')) {
      count = parseInt(line.split(/\s+/)[2], 10);
    } else if (line.startsWith('property')) {
      const parts = line.split(/\s+/);
      if (parts[1] !== 'float') {
        throw new Error(`Unsupported PLY property type "${parts[1]}" (expected float).`);
      }
      props.push(parts[parts.length - 1]);
    }
  }

  const stride = props.length * 4; // all float32
  const idx: Record<string, number> = {};
  props.forEach((p, i) => (idx[p] = i));
  const need = (name: string): number => {
    if (!(name in idx)) throw new Error(`PLY missing required property "${name}".`);
    return idx[name];
  };

  // Resolve column indices once.
  const iX = need('x');
  const iY = need('y');
  const iZ = need('z');
  const iDc0 = need('f_dc_0');
  const iDc1 = need('f_dc_1');
  const iDc2 = need('f_dc_2');
  const iOp = need('opacity');
  const iS0 = need('scale_0');
  const iS1 = need('scale_1');
  const iS2 = need('scale_2');
  const iR0 = need('rot_0');
  const iR1 = need('rot_1');
  const iR2 = need('rot_2');
  const iR3 = need('rot_3');

  const view = new DataView(buffer, dataStart);
  const f = (v: number, col: number): number => view.getFloat32(v * stride + col * 4, true);

  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const rotations = new Float32Array(count * 4);
  const opacities = new Float32Array(count);
  const colors = new Float32Array(count * 3);

  for (let v = 0; v < count; v++) {
    positions[v * 3 + 0] = f(v, iX);
    positions[v * 3 + 1] = f(v, iY);
    positions[v * 3 + 2] = f(v, iZ);

    // SH DC term → base color.
    colors[v * 3 + 0] = clamp01(0.5 + SH_C0 * f(v, iDc0));
    colors[v * 3 + 1] = clamp01(0.5 + SH_C0 * f(v, iDc1));
    colors[v * 3 + 2] = clamp01(0.5 + SH_C0 * f(v, iDc2));

    opacities[v] = sigmoid(f(v, iOp));

    scales[v * 3 + 0] = Math.exp(f(v, iS0));
    scales[v * 3 + 1] = Math.exp(f(v, iS1));
    scales[v * 3 + 2] = Math.exp(f(v, iS2));

    // Quaternion (w, x, y, z) = rot_0..3, normalized.
    let qw = f(v, iR0);
    let qx = f(v, iR1);
    let qy = f(v, iR2);
    let qz = f(v, iR3);
    const inv = 1 / (Math.hypot(qw, qx, qy, qz) || 1);
    rotations[v * 4 + 0] = qw * inv;
    rotations[v * 4 + 1] = qx * inv;
    rotations[v * 4 + 2] = qy * inv;
    rotations[v * 4 + 3] = qz * inv;
  }

  return { count, positions, scales, rotations, opacities, colors };
}
