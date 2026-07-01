// [concept] The real splat: project each 3D Gaussian to a 2D screen ellipse (EWA),
// shade it, and draw it as a soft, alpha-blended billboard.
//
// Vertex stage:
//   • Σ' = J W Σ Wᵀ Jᵀ  — 3D covariance → 2D screen ellipse (EWA projection).
//   • normal = shortest covariance axis, oriented outward (sign disambiguation).
//   • shade with Lambert + Blinn-Phong + ambient, blended baked↔relit.
// Fragment stage: 2D Gaussian falloff × opacity → alpha.

struct Camera {
  viewProj     : mat4x4f,
  view         : mat4x4f,
  objectCenter : vec3f, // scene centroid — used to orient normals outward
  renderMode   : f32,   // 0 = shaded/baked color, 1 = normals-as-RGB debug
  focal        : vec2f, // pixels
  viewport     : vec2f, // pixels (width, height)
};
@group(0) @binding(0) var<uniform> camera : Camera;

struct Lighting {
  lightPos   : vec3f,
  intensity  : f32,
  lightColor : vec3f,
  ambient    : f32,
  specColor  : vec3f,
  shininess  : f32,
  relight    : f32, // 0 = baked color, 1 = fully relit
};
@group(0) @binding(1) var<uniform> lighting : Lighting;

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

// Lambert + Blinn-Phong + ambient. `worldPos` is the splat center (flat per-splat).
fn shade(N : vec3f, worldPos : vec3f, albedo : vec3f, camPos : vec3f) -> vec3f {
  let L = normalize(lighting.lightPos - worldPos); // surface → light
  let V = normalize(camPos - worldPos);            // surface → eye
  let H = normalize(L + V);                         // halfway vector
  let diff = max(dot(N, L), 0.0);                   // Lambert
  let spec = pow(max(dot(N, H), 0.0), lighting.shininess); // Blinn-Phong
  let diffuse = diff * lighting.intensity * lighting.lightColor; // vec3
  return albedo * (vec3f(lighting.ambient) + diffuse) + spec * lighting.specColor;
}

struct VSOut {
  @builtin(position) clip    : vec4f,
  @location(0)       color   : vec3f,
  @location(1)       opacity : f32,
  @location(2)       vSigma  : vec2f, // position within the ellipse, in σ units
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
  let sigma1 = sqrt(lambda1);
  let sigma2 = sqrt(lambda2);
  var e1 : vec2f;
  if (abs(b) < 1e-6) {
    e1 = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), a >= c);
  } else {
    e1 = normalize(vec2f(lambda1 - c, b));
  }
  let e2 = vec2f(-e1.y, e1.x);

  // --- place the quad corner ±Kσ along each axis (pixels → clip) ---
  let K = 3.0;
  let offsetPx = corner.x * (K * sigma1) * e1 + corner.y * (K * sigma2) * e2;
  let ndcOffset = 2.0 * offsetPx / camera.viewport;
  var clip = camera.viewProj * vec4f(center, 1.0);
  clip.x += ndcOffset.x * clip.w;
  clip.y += ndcOffset.y * clip.w;

  // --- surface normal = shortest covariance axis, oriented outward ---
  var normal = normalize(R[2]);
  var minScale = scale.z;
  if (scale.x < minScale) { minScale = scale.x; normal = normalize(R[0]); }
  if (scale.y < minScale) { minScale = scale.y; normal = normalize(R[1]); }
  if (dot(normal, center - camera.objectCenter) < 0.0) { normal = -normal; }

  // --- display color: normals debug, or (baked ↔ relit) shading ---
  var displayColor = color;
  if (camera.renderMode > 0.5) {
    displayColor = normal * 0.5 + vec3f(0.5);
  } else {
    let camPos = -(transpose(W) * camera.view[3].xyz); // camera world position
    let lit = shade(normal, center, color, camPos);
    displayColor = mix(color, lit, lighting.relight);
  }

  out.clip = clip;
  out.color = displayColor;
  out.opacity = opacity;
  out.vSigma = corner * K;
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
  return vec4f(in.color * alpha, alpha); // premultiplied alpha
}
