// [plumbing] Parser for the antimatter15 ".splat" format: a flat array of
// 32-byte records, no header. Each record:
//   offset  0 : position  float32 x3  (12 bytes)
//   offset 12 : scale     float32 x3  (12 bytes, already linear / exp'd)
//   offset 24 : color     uint8   x4  (4 bytes; r,g,b base color, a = opacity)
//   offset 28 : rotation  uint8   x4  (4 bytes; quaternion packed to bytes)
// Produces the decoded SplatCloud from splat-data.ts.

import type { SplatCloud } from './splat-data';

const BYTES_PER_SPLAT = 32;
const FLOATS_PER_SPLAT = BYTES_PER_SPLAT / 4; // 8

export function parseSplat(buffer: ArrayBuffer): SplatCloud {
  const count = Math.floor(buffer.byteLength / BYTES_PER_SPLAT);
  const f32 = new Float32Array(buffer); // view for the float fields (pos, scale)
  const u8 = new Uint8Array(buffer); // view for the byte fields (color, rotation)

  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const rotations = new Float32Array(count * 4);
  const opacities = new Float32Array(count);
  const colors = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    const f = i * FLOATS_PER_SPLAT;
    const b = i * BYTES_PER_SPLAT;

    positions[i * 3 + 0] = f32[f + 0];
    positions[i * 3 + 1] = f32[f + 1];
    positions[i * 3 + 2] = f32[f + 2];

    scales[i * 3 + 0] = f32[f + 3];
    scales[i * 3 + 1] = f32[f + 4];
    scales[i * 3 + 2] = f32[f + 5];

    // Color bytes (24..26) → [0,1]; 4th byte (27) is opacity.
    colors[i * 3 + 0] = u8[b + 24] / 255;
    colors[i * 3 + 1] = u8[b + 25] / 255;
    colors[i * 3 + 2] = u8[b + 26] / 255;
    opacities[i] = u8[b + 27] / 255;

    // Rotation bytes (28..31): each packed as (q+1)*128, so decode (byte-128)/128.
    // Order is (w, x, y, z); renormalize to be safe after the byte round-trip.
    let qw = (u8[b + 28] - 128) / 128;
    let qx = (u8[b + 29] - 128) / 128;
    let qy = (u8[b + 30] - 128) / 128;
    let qz = (u8[b + 31] - 128) / 128;
    const inv = 1 / (Math.hypot(qw, qx, qy, qz) || 1);
    rotations[i * 4 + 0] = qw * inv;
    rotations[i * 4 + 1] = qx * inv;
    rotations[i * 4 + 2] = qy * inv;
    rotations[i * 4 + 3] = qz * inv;
  }

  return { count, positions, scales, rotations, opacities, colors };
}
