// [plumbing] Minimal line shader: transform each vertex by the camera's
// view-projection matrix and pass its color straight through. Used only by the
// reference grid/axes — the splat shaders come later.

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
