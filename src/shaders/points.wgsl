// [plumbing] Flat-colored point shader for Phase 0. Same camera transform as the
// grid: place each splat's center on screen and paint it its base (SH DC) color.
// One pixel per splat — no ellipse, no falloff, no blending yet. Those are Phase 1,
// and this file is where that growth will happen.

struct Camera {
  viewProj : mat4x4f,
};

@group(0) @binding(0) var<uniform> camera : Camera;

struct VSOut {
  @builtin(position) clip  : vec4f,
  @location(0)       color : vec3f,
};

@vertex
fn vs(
  @location(0) position : vec3f,
  @location(1) color    : vec3f,
) -> VSOut {
  var out : VSOut;
  out.clip = camera.viewProj * vec4f(position, 1.0);
  out.color = color;
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
