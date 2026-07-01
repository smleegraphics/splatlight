// [plumbing] Orbit camera. Holds the viewpoint as spherical coordinates around a
// target (azimuth, elevation, distance), turns mouse drags into changes to those,
// and each frame produces a view-projection matrix for the shaders. No splatting
// math here — just standard camera bookkeeping.

import { mat4, vec3 } from 'wgpu-matrix';
import type { Mat4, Vec3 } from 'wgpu-matrix';

const WORLD_UP: Vec3 = vec3.create(0, 1, 0);
const HALF_PI = Math.PI / 2;
const EPS = 0.001;

export class OrbitCamera {
  // Orientation as spherical coordinates around `target`.
  azimuth = 0.7; // radians, around the Y axis
  elevation = 0.5; // radians, above the XZ plane
  distance = 4.0;
  target: Vec3 = vec3.create(0, 0, 0);

  // Projection params.
  fovY = (50 * Math.PI) / 180;
  near = 0.01;
  far = 100;

  // Computed each update().
  eye: Vec3 = vec3.create();
  private view: Mat4 = mat4.identity();
  private proj: Mat4 = mat4.identity();
  private viewProj: Mat4 = mat4.identity();

  private dragging: 'orbit' | 'pan' | null = null;
  private lastX = 0;
  private lastY = 0;

  attach(el: HTMLElement): void {
    el.addEventListener('contextmenu', (e) => e.preventDefault());

    el.addEventListener('pointerdown', (e) => {
      this.dragging = e.button === 2 ? 'pan' : 'orbit';
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      el.setPointerCapture(e.pointerId);
    });

    el.addEventListener('pointerup', (e) => {
      this.dragging = null;
      el.releasePointerCapture(e.pointerId);
    });

    el.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastX;
      const dy = e.clientY - this.lastY;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      if (this.dragging === 'orbit') this.orbit(dx, dy);
      else this.pan(dx, dy);
    });

    el.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        // Multiplicative zoom feels linear to the eye and never crosses 0.
        this.distance *= Math.exp(e.deltaY * 0.001);
        this.distance = clamp(this.distance, 0.2, 50);
      },
      { passive: false },
    );
  }

  private orbit(dx: number, dy: number): void {
    this.azimuth -= dx * 0.005;
    this.elevation = clamp(this.elevation + dy * 0.005, -HALF_PI + EPS, HALF_PI - EPS);
  }

  private pan(dx: number, dy: number): void {
    // Move the target across the camera's local right/up plane. Scaling by
    // distance keeps the pan speed feeling constant at any zoom level.
    const forward = vec3.normalize(vec3.subtract(this.target, this.eye));
    const right = vec3.normalize(vec3.cross(forward, WORLD_UP));
    const up = vec3.cross(right, forward);
    const k = this.distance * 0.0015;
    vec3.addScaled(this.target, right, -dx * k, this.target);
    vec3.addScaled(this.target, up, dy * k, this.target);
  }

  /** The world→camera matrix from the last update() (needed for EWA projection). */
  get viewMatrix(): Mat4 {
    return this.view;
  }

  /** Recompute and return the view-projection matrix for the given aspect ratio. */
  update(aspect: number): Mat4 {
    const ce = Math.cos(this.elevation);
    const se = Math.sin(this.elevation);
    const ca = Math.cos(this.azimuth);
    const sa = Math.sin(this.azimuth);

    vec3.set(
      this.target[0] + this.distance * ce * sa,
      this.target[1] + this.distance * se,
      this.target[2] + this.distance * ce * ca,
      this.eye,
    );

    mat4.lookAt(this.eye, this.target, WORLD_UP, this.view);
    mat4.perspective(this.fovY, aspect, this.near, this.far, this.proj);
    mat4.multiply(this.proj, this.view, this.viewProj);
    return this.viewProj;
  }
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}
