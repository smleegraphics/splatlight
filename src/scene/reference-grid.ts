// [plumbing] Generates a ground grid on the XZ plane plus colored X/Y/Z axis
// lines. This isn't part of the splat pipeline — it's a spatial reference so we
// can SEE that the orbit camera works and have a coordinate frame to reason about
// when real splats arrive. Each vertex is 6 floats: position(xyz) + color(rgb).

export interface LineGeometry {
  /** Interleaved [x,y,z, r,g,b] per vertex. */
  vertices: Float32Array;
  vertexCount: number;
}

export function makeReferenceGrid(halfExtent = 5, step = 1): LineGeometry {
  const verts: number[] = [];
  const grid: [number, number, number] = [0.22, 0.24, 0.28];

  const push = (
    x: number,
    y: number,
    z: number,
    c: [number, number, number],
  ): void => {
    verts.push(x, y, z, c[0], c[1], c[2]);
  };

  // Grid lines parallel to X and Z (skip the two center lines; the axes cover them).
  for (let i = -halfExtent; i <= halfExtent; i += step) {
    if (i !== 0) {
      push(-halfExtent, 0, i, grid);
      push(halfExtent, 0, i, grid);
      push(i, 0, -halfExtent, grid);
      push(i, 0, halfExtent, grid);
    }
  }

  // Axes through the origin: X red, Y green (up), Z blue.
  const L = halfExtent;
  push(0, 0, 0, [0.85, 0.25, 0.25]);
  push(L, 0, 0, [0.85, 0.25, 0.25]);
  push(0, 0, 0, [0.3, 0.8, 0.35]);
  push(0, L, 0, [0.3, 0.8, 0.35]);
  push(0, 0, 0, [0.3, 0.45, 0.9]);
  push(0, 0, L, [0.3, 0.45, 0.9]);

  return { vertices: new Float32Array(verts), vertexCount: verts.length / 6 };
}
