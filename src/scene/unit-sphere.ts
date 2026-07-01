// [plumbing] A low-poly unit sphere (UV sphere). We draw ONE of these per splat,
// instanced, and the ellipsoid shader scales + rotates it into each splat's
// covariance shape. Positions on a unit sphere double as surface normals.

export interface Mesh {
  positions: Float32Array; // 3 per vertex, on the unit sphere
  indices: Uint16Array; // triangle list
  indexCount: number;
}

export function makeUnitSphere(latBands = 8, lonBands = 12): Mesh {
  const positions: number[] = [];
  const indices: number[] = [];

  for (let lat = 0; lat <= latBands; lat++) {
    const theta = (lat / latBands) * Math.PI; // 0..π from north pole
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);
    for (let lon = 0; lon <= lonBands; lon++) {
      const phi = (lon / lonBands) * 2 * Math.PI;
      // Unit-sphere point; also its own outward normal.
      positions.push(sinT * Math.cos(phi), cosT, sinT * Math.sin(phi));
    }
  }

  const stride = lonBands + 1;
  for (let lat = 0; lat < latBands; lat++) {
    for (let lon = 0; lon < lonBands; lon++) {
      const a = lat * stride + lon;
      const b = a + stride;
      // Two triangles per quad, wound so the outward face is front.
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }

  return {
    positions: new Float32Array(positions),
    indices: new Uint16Array(indices),
    indexCount: indices.length,
  };
}
