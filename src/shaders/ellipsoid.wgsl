// [concept made visible] Debug view: draw each splat as its real 3D ellipsoid.
// The vertex stage rebuilds the covariance shape directly — it takes a unit-sphere
// vertex and applies M = R * S (rotation * scale), which is exactly the transform
// whose covariance is Σ = M Mᵀ = R S² Rᵀ. Seeing the oriented ellipsoids IS seeing
// the covariance.
//
// The lighting here is a FIXED headlight, only so the shapes read as 3D. It is NOT
// the Phase 2 relighting feature.

struct Camera {
  viewProj : mat4x4f,
};
@group(0) @binding(0) var<uniform> camera : Camera;

// Build a 3x3 rotation matrix from a unit quaternion (x, y, z, w).
fn quatToMat3(q : vec4f) -> mat3x3f {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  let xx = x * x; let yy = y * y; let zz = z * z;
  let xy = x * y; let xz = x * z; let yz = y * z;
  let wx = w * x; let wy = w * y; let wz = w * z;
  // Columns of R (WGSL matrices are column-major).
  return mat3x3f(
    vec3f(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz),       2.0 * (xz - wy)),
    vec3f(2.0 * (xy - wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
    vec3f(2.0 * (xz + wy),       2.0 * (yz - wx),       1.0 - 2.0 * (xx + yy)),
  );
}

struct VSOut {
  @builtin(position) clip   : vec4f,
  @location(0)       color  : vec3f,
  @location(1)       normal : vec3f, // world-space, for the debug headlight
};

@vertex
fn vs(
  // Per-vertex: unit sphere position (also its normal).
  @location(0) spherePos : vec3f,
  // Per-instance: this splat's covariance + color.
  @location(1) center : vec3f,
  @location(2) scale  : vec3f,
  @location(3) quat   : vec4f,
  @location(4) color  : vec3f,
) -> VSOut {
  let R = quatToMat3(quat);

  // M = R * S applied to the sphere point → the ellipsoid, then translate.
  let world = R * (spherePos * scale) + center;

  // Correct ellipsoid normal = R * (n / scale) (inverse-transpose of R*S on a
  // unit-sphere normal). Good enough for the debug shade.
  let normal = normalize(R * (spherePos / scale));

  var out : VSOut;
  out.clip = camera.viewProj * vec4f(world, 1.0);
  out.color = color;
  out.normal = normal;
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  let L = normalize(vec3f(0.4, 0.8, 0.5)); // fixed debug light direction
  let diffuse = max(dot(normalize(in.normal), L), 0.0);
  let shade = 0.35 + 0.65 * diffuse; // ambient + diffuse, debug only
  return vec4f(in.color * shade, 1.0);
}
