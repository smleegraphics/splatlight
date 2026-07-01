// [concept] The real splat: project each 3D Gaussian to a 2D screen ellipse (EWA)
// and draw it as a soft, alpha-blended billboard.
//
// Vertex stage builds Σ' = J W Σ Wᵀ Jᵀ:
//   Σ = R S² Rᵀ   the 3D covariance (world space)
//   W             world→camera rotation (from the view matrix)
//   J             Jacobian of the perspective divide at the splat center
//                 (the local linear approximation of the nonlinear projection)
// then sizes + orients a quad to the resulting 2×2 ellipse. Fragment stage
// evaluates the 2D Gaussian falloff × opacity → alpha.

struct Camera {
  viewProj   : mat4x4f,
  view       : mat4x4f,
  focal      : vec2f, // pixels
  viewport   : vec2f, // pixels (width, height)
  renderMode : f32,   // 0 = SH color, 1 = normals-as-RGB debug
};
@group(0) @binding(0) var<uniform> camera : Camera;

// 3x3 rotation matrix from a unit quaternion (x, y, z, w).
fn quatToMat3(q : vec4f) -> mat3x3f {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  let xx = x * x; let yy = y * y; let zz = z * z;
  let xy = x * y; let xz = x * z; let yz = y * z;
  let wx = w * x; let wy = w * y; let wz = w * z;
  return mat3x3f(
    vec3f(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz),       2.0 * (xz - wy)),
    vec3f(2.0 * (xy - wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
    vec3f(2.0 * (xz + wy),       2.0 * (yz - wx),       1.0 - 2.0 * (xx + yy)),
  );
}

struct VSOut {
  @builtin(position) clip    : vec4f,
  @location(0)       color   : vec3f,
  @location(1)       opacity : f32,
  @location(2)       vSigma  : vec2f, // position within the ellipse, in σ units
  @location(3)       normal  : vec3f, // world-space surface normal (shortest axis)
};

@vertex
fn vs(
  @location(0) corner  : vec2f, // quad corner in [-1, 1]
  @location(1) center  : vec3f,
  @location(2) scale   : vec3f,
  @location(3) quat    : vec4f,
  @location(4) color   : vec3f,
  @location(5) opacity : f32,
) -> VSOut {
  var out : VSOut;

  // --- 3D covariance Σ = (R S)(R S)ᵀ = R S² Rᵀ, in world space ---
  let R = quatToMat3(quat);
  let RS = mat3x3f(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
  let Sigma = RS * transpose(RS);

  // --- splat center in camera space; W = world→camera rotation ---
  let pCam = (camera.view * vec4f(center, 1.0)).xyz;
  let W = mat3x3f(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);

  // Cull splats at/behind the camera (the projection blows up there).
  if (pCam.z > -0.0001) {
    out.clip = vec4f(2.0, 2.0, 2.0, 1.0); // off-screen, gets clipped
    return out;
  }

  // --- Jacobian of the perspective divide, evaluated at the center ---
  let z = pCam.z;
  let J = mat3x3f(
    vec3f(camera.focal.x / z, 0.0, 0.0),
    vec3f(0.0, camera.focal.y / z, 0.0),
    vec3f(-camera.focal.x * pCam.x / (z * z), -camera.focal.y * pCam.y / (z * z), 0.0),
  );

  // --- 2D screen covariance Σ' = J W Σ Wᵀ Jᵀ (read the top-left 2×2) ---
  let T = J * W;
  let cov = T * Sigma * transpose(T);
  let a = cov[0].x + 0.3; // low-pass dilation: keep tiny splats ≥ ~1px
  let b = cov[0].y;
  let c = cov[1].y + 0.3;

  // --- eigen-decompose [[a,b],[b,c]] → principal axes of the ellipse ---
  let mid = 0.5 * (a + c);
  let disc = sqrt(max(mid * mid - (a * c - b * b), 0.0));
  let lambda1 = mid + disc;
  let lambda2 = max(mid - disc, 0.0);
  let sigma1 = sqrt(lambda1); // pixels, along major axis
  let sigma2 = sqrt(lambda2); // pixels, along minor axis
  var e1 : vec2f;
  if (abs(b) < 1e-6) {
    e1 = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), a >= c);
  } else {
    e1 = normalize(vec2f(lambda1 - c, b));
  }
  let e2 = vec2f(-e1.y, e1.x);

  // --- place the quad corner ±Kσ along each axis (pixels → clip) ---
  let K = 3.0; // cover ±3σ (99.7% of the Gaussian)
  let offsetPx = corner.x * (K * sigma1) * e1 + corner.y * (K * sigma2) * e2;
  let ndcOffset = 2.0 * offsetPx / camera.viewport;

  var clip = camera.viewProj * vec4f(center, 1.0);
  clip.x += ndcOffset.x * clip.w; // shift in NDC = (offset) * w before the divide
  clip.y += ndcOffset.y * clip.w;

  // Surface normal ≈ the covariance's shortest axis = R's column with the
  // smallest scale. R is a rotation, so its columns are already unit length.
  var normal = R[2];
  var minScale = scale.z;
  if (scale.x < minScale) { minScale = scale.x; normal = R[0]; }
  if (scale.y < minScale) { minScale = scale.y; normal = R[1]; }

  out.clip = clip;
  out.color = color;
  out.opacity = opacity;
  out.vSigma = corner * K; // in σ units → falloff below
  out.normal = normal;
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  // 2D Gaussian falloff: exp(-½ · (distance from center in σ units)²).
  let power = -0.5 * dot(in.vSigma, in.vSigma);
  let alpha = in.opacity * exp(power);
  if (alpha < 1.0 / 255.0) {
    discard;
  }
  var rgb = in.color;
  if (camera.renderMode > 0.5) {
    rgb = normalize(in.normal) * 0.5 + vec3f(0.5); // normal (x,y,z) → RGB
  }
  return vec4f(rgb * alpha, alpha); // premultiplied alpha
}
