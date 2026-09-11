// [plumbing] Entry point: init WebGPU, build the pipelines once, load a splat
// scene (default .ply, or one you drag-and-drop / pick), and run the render loop.

import { initWebGPU } from './gpu/context';
import { createBufferWithData, writeBuffer } from './gpu/buffers';
import { OrbitCamera } from './camera/orbit-camera';
import { makeReferenceGrid } from './scene/reference-grid';
import {
  makeSyntheticCloud,
  cloudToPointVertices,
  cloudToInstanceData,
  cloudToSplatInstances,
} from './scene/splat-data';
import { makeUnitSphere } from './scene/unit-sphere';
import { parsePly } from './scene/ply-parser';
import type { SplatCloud } from './scene/splat-data';
import lineShaderSrc from './shaders/line.wgsl?raw';
import pointsShaderSrc from './shaders/points.wgsl?raw';
import ellipsoidShaderSrc from './shaders/ellipsoid.wgsl?raw';
import splatShaderSrc from './shaders/splat.wgsl?raw';
import { Pane } from 'tweakpane';

// pos(3) + color(3) interleaved, all f32 — shared by the grid and point pipelines.
const POS_COLOR_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: 6 * 4,
  attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position
    { shaderLocation: 1, offset: 12, format: 'float32x3' }, // color
  ],
};

// Ellipsoid debug view: unit-sphere mesh (per vertex) + splat data (per instance).
const SPHERE_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: 3 * 4,
  attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }], // sphere pos
};
const INSTANCE_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: 13 * 4,
  stepMode: 'instance',
  attributes: [
    { shaderLocation: 1, offset: 0, format: 'float32x3' }, // center
    { shaderLocation: 2, offset: 12, format: 'float32x3' }, // scale
    { shaderLocation: 3, offset: 24, format: 'float32x4' }, // quaternion (x,y,z,w)
    { shaderLocation: 4, offset: 40, format: 'float32x3' }, // color
  ],
};

// 2D splat billboard: a static quad (per vertex) + splat data incl. opacity (per instance).
const QUAD_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: 2 * 4,
  attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }], // corner in [-1,1]
};
const SPLAT_INSTANCE_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: 14 * 4,
  stepMode: 'instance',
  attributes: [
    { shaderLocation: 1, offset: 0, format: 'float32x3' }, // center
    { shaderLocation: 2, offset: 12, format: 'float32x3' }, // scale
    { shaderLocation: 3, offset: 24, format: 'float32x4' }, // quaternion (x,y,z,w)
    { shaderLocation: 4, offset: 40, format: 'float32x3' }, // color
    { shaderLocation: 5, offset: 52, format: 'float32' }, // opacity
  ],
};

const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const SPLAT_FLOATS = 14;
const NUM_BUCKETS = 65536; // 16-bit depth quantization

/** Fetch + parse the default .ply scene; fall back to the synthetic sphere. */
async function loadCloud(): Promise<SplatCloud> {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}luigi.ply`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cloud = parsePly(await res.arrayBuffer());
    console.log(`Loaded luigi.ply: ${cloud.count} splats`);
    return cloud;
  } catch (err) {
    console.warn('Could not load luigi.ply — falling back to synthetic cloud.', err);
    return makeSyntheticCloud();
  }
}

/** Frame the camera to the cloud; return its centroid + bounding radius. */
function fitCameraToCloud(
  camera: OrbitCamera,
  cloud: SplatCloud,
): { center: [number, number, number]; radius: number } {
  const n = cloud.count;
  if (n === 0) return { center: [0, 0, 0], radius: 1 };
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let i = 0; i < n; i++) {
    cx += cloud.positions[i * 3];
    cy += cloud.positions[i * 3 + 1];
    cz += cloud.positions[i * 3 + 2];
  }
  cx /= n;
  cy /= n;
  cz /= n;
  let r = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.hypot(
      cloud.positions[i * 3] - cx,
      cloud.positions[i * 3 + 1] - cy,
      cloud.positions[i * 3 + 2] - cz,
    );
    if (d > r) r = d;
  }
  r = r > 0 ? r : 1;
  camera.target[0] = cx;
  camera.target[1] = cy;
  camera.target[2] = cz;
  camera.distance = r * 2.2;
  camera.near = Math.max(0.001, r * 0.002);
  camera.far = Math.max(camera.far, r * 20);
  return { center: [cx, cy, cz], radius: r };
}

/** Parse a `#rrggbb` (or `#rgb`) hex color to RGB floats in [0,1]. */
function hexToRgb(hex: string): [number, number, number] {
  let h = hex.replace('#', '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  const n = parseInt(h, 16);
  if (Number.isNaN(n)) return [1, 1, 1];
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/** GPU/CPU state that depends on the loaded cloud (rebuilt on load). */
interface Scene {
  cloud: SplatCloud;
  center: [number, number, number];
  radius: number;
  pointBuffer: GPUBuffer;
  instanceBuffer: GPUBuffer; // ellipsoid view
  splatInstanceBuffer: GPUBuffer; // 2D splats (reordered by the sort)
  splatInstances: Float32Array;
  order: Uint32Array;
  depths: Float32Array;
  buckets: Uint16Array;
  sortedInstances: Float32Array;
}

async function main(): Promise<void> {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  const { device, context, format } = await initWebGPU(canvas);

  const camera = new OrbitCamera();
  camera.attach(canvas);

  // --- cloud-independent resources (built once) ---

  const grid = makeReferenceGrid();
  const vertexBuffer = createBufferWithData(device, grid.vertices, GPUBufferUsage.VERTEX, 'grid-vertices');

  // Camera uniform: viewProj(16) + view(16) + objectCenter(3) + renderMode(1) +
  // focal(2) + viewport(2) = 40 floats.
  const cameraData = new Float32Array(40);
  const cameraBuffer = device.createBuffer({
    size: cameraData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const lineModule = device.createShaderModule({ code: lineShaderSrc });
  const gridPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: lineModule, entryPoint: 'vs', buffers: [POS_COLOR_LAYOUT] },
    fragment: { module: lineModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'line-list' },
    depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
  });
  const gridBindGroup = device.createBindGroup({
    layout: gridPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: cameraBuffer } }],
  });

  const pointsModule = device.createShaderModule({ code: pointsShaderSrc });
  const pointsPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: pointsModule, entryPoint: 'vs', buffers: [POS_COLOR_LAYOUT] },
    fragment: { module: pointsModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'point-list' },
    depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
  });
  const pointsBindGroup = device.createBindGroup({
    layout: pointsPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: cameraBuffer } }],
  });

  const sphere = makeUnitSphere();
  const sphereVertexBuffer = createBufferWithData(device, sphere.positions, GPUBufferUsage.VERTEX, 'unit-sphere-verts');
  const sphereIndexBuffer = createBufferWithData(device, sphere.indices, GPUBufferUsage.INDEX, 'unit-sphere-indices');
  const ellipsoidModule = device.createShaderModule({ code: ellipsoidShaderSrc });
  const ellipsoidPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: ellipsoidModule, entryPoint: 'vs', buffers: [SPHERE_LAYOUT, INSTANCE_LAYOUT] },
    fragment: { module: ellipsoidModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
  });
  const ellipsoidBindGroup = device.createBindGroup({
    layout: ellipsoidPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: cameraBuffer } }],
  });

  const quadBuffer = createBufferWithData(
    device,
    new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]),
    GPUBufferUsage.VERTEX,
    'splat-quad',
  );
  const splatModule = device.createShaderModule({ code: splatShaderSrc });
  const splatPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: splatModule, entryPoint: 'vs', buffers: [QUAD_LAYOUT, SPLAT_INSTANCE_LAYOUT] },
    fragment: {
      module: splatModule,
      entryPoint: 'fs',
      targets: [
        {
          format,
          blend: {
            // Premultiplied-alpha "over" compositing.
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' },
  });
  const lightingData = new Float32Array(16);
  const lightingBuffer = device.createBuffer({
    size: lightingData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const splatBindGroup = device.createBindGroup({
    layout: splatPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: { buffer: lightingBuffer } },
    ],
  });

  const counts = new Uint32Array(NUM_BUCKETS); // reused sort scratch

  // Build all cloud-dependent state (rebuilt on load / drag-drop).
  const buildScene = (cloud: SplatCloud): Scene => {
    const { center, radius } = fitCameraToCloud(camera, cloud);
    const splatInstances = cloudToSplatInstances(cloud);
    return {
      cloud,
      center,
      radius,
      pointBuffer: createBufferWithData(device, cloudToPointVertices(cloud), GPUBufferUsage.VERTEX, 'splat-points'),
      instanceBuffer: createBufferWithData(device, cloudToInstanceData(cloud), GPUBufferUsage.VERTEX, 'splat-instances'),
      splatInstanceBuffer: createBufferWithData(device, splatInstances, GPUBufferUsage.VERTEX, 'splat-instances-2d'),
      splatInstances,
      order: new Uint32Array(cloud.count),
      depths: new Float32Array(cloud.count),
      buckets: new Uint16Array(cloud.count),
      sortedInstances: new Float32Array(splatInstances.length),
    };
  };

  let scene = buildScene(await loadCloud());
  let lastSortKey = '';

  const loadIntoScene = (cloud: SplatCloud): void => {
    scene.pointBuffer.destroy();
    scene.instanceBuffer.destroy();
    scene.splatInstanceBuffer.destroy();
    scene = buildScene(cloud);
    lastSortKey = '';
  };

  // Load a .ply by dragging it onto the window, or via the panel button below.
  const loadFile = async (file: File): Promise<void> => {
    try {
      loadIntoScene(parsePly(await file.arrayBuffer()));
      console.log(`Loaded ${file.name}: ${scene.cloud.count} splats`);
    } catch (err) {
      console.error(`Failed to load ${file.name}:`, err);
    }
  };
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e: DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) void loadFile(file);
  });
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = '.ply';
  fileInput.style.display = 'none';
  document.body.appendChild(fileInput);
  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    if (file) void loadFile(file);
    fileInput.value = '';
  });

  // --- lighting controls (light position is in units of the scene radius) ---
  const lightParams = {
    relight: 0.7,
    intensity: 1.0,
    ambient: 0.25,
    shininess: 24,
    specular: 0.25,
    color: '#ffffff',
    normalMode: 0, // 0 = object (centroid), 1 = scene (camera-facing)
    offX: 1.5,
    offY: 1.5,
    offZ: 1.0,
  };
  const packLighting = (): void => {
    const [lr, lg, lb] = hexToRgb(lightParams.color);
    const [cx, cy, cz] = scene.center;
    const r = scene.radius;
    lightingData[0] = cx + lightParams.offX * r;
    lightingData[1] = cy + lightParams.offY * r;
    lightingData[2] = cz + lightParams.offZ * r;
    lightingData[3] = lightParams.intensity;
    lightingData[4] = lr;
    lightingData[5] = lg;
    lightingData[6] = lb;
    lightingData[7] = lightParams.ambient;
    lightingData[8] = lightParams.specular;
    lightingData[9] = lightParams.specular;
    lightingData[10] = lightParams.specular;
    lightingData[11] = lightParams.shininess;
    lightingData[12] = lightParams.relight;
    lightingData[13] = lightParams.normalMode;
    writeBuffer(device, lightingBuffer, lightingData);
  };

  const pane = new Pane({ title: 'lighting' });
  pane.addButton({ title: 'load .ply…' }).on('click', () => fileInput.click());
  pane.addBinding(lightParams, 'relight', { min: 0, max: 1, step: 0.01 });
  pane.addBinding(lightParams, 'intensity', { min: 0, max: 3, step: 0.01 });
  pane.addBinding(lightParams, 'ambient', { min: 0, max: 1, step: 0.01 });
  pane.addBinding(lightParams, 'shininess', { min: 1, max: 128, step: 1 });
  pane.addBinding(lightParams, 'specular', { min: 0, max: 1, step: 0.01 });
  pane.addBinding(lightParams, 'color');
  pane.addBinding(lightParams, 'normalMode', {
    label: 'normals',
    options: { 'object (centroid)': 0, 'scene (camera-facing)': 1 },
  });
  const lightFolder = pane.addFolder({ title: 'light position (× radius)' });
  lightFolder.addBinding(lightParams, 'offX', { min: -4, max: 4, step: 0.05 });
  lightFolder.addBinding(lightParams, 'offY', { min: -4, max: 4, step: 0.05 });
  lightFolder.addBinding(lightParams, 'offZ', { min: -4, max: 4, step: 0.05 });

  // View cycle: 'v' rotates through splats / normals / ellipsoids / points.
  const VIEWS = ['splats', 'normals', 'ellipsoids', 'points'] as const;
  let viewIndex = 0;
  window.addEventListener('keydown', (e) => {
    if (e.key === 'v' || e.key === 'V') viewIndex = (viewIndex + 1) % VIEWS.length;
  });

  // [concept] Depth sort: back-to-front via a 16-bit counting sort; reorder the
  // instance buffer and re-upload. Re-sorts only when the camera moved.
  const sortSplats = (): void => {
    const key = `${camera.azimuth}|${camera.elevation}|${camera.distance}|${camera.target[0]}|${camera.target[1]}|${camera.target[2]}`;
    if (key === lastSortKey) return;
    lastSortKey = key;

    const { cloud, splatInstances, order, depths, buckets, sortedInstances, splatInstanceBuffer } = scene;

    const m = camera.viewMatrix;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < cloud.count; i++) {
      const p = i * 3;
      const z = m[2] * cloud.positions[p] + m[6] * cloud.positions[p + 1] + m[10] * cloud.positions[p + 2] + m[14];
      depths[i] = z;
      if (z < min) min = z;
      if (z > max) max = z;
    }
    const scale = max > min ? (NUM_BUCKETS - 1) / (max - min) : 0;
    for (let i = 0; i < cloud.count; i++) {
      buckets[i] = Math.min(NUM_BUCKETS - 1, ((depths[i] - min) * scale) | 0);
    }
    counts.fill(0);
    for (let i = 0; i < cloud.count; i++) counts[buckets[i]]++;
    let running = 0;
    for (let b = 0; b < NUM_BUCKETS; b++) {
      const c = counts[b];
      counts[b] = running;
      running += c;
    }
    for (let i = 0; i < cloud.count; i++) order[counts[buckets[i]]++] = i;
    for (let i = 0; i < cloud.count; i++) {
      const src = order[i] * SPLAT_FLOATS;
      const dst = i * SPLAT_FLOATS;
      for (let k = 0; k < SPLAT_FLOATS; k++) sortedInstances[dst + k] = splatInstances[src + k];
    }
    writeBuffer(device, splatInstanceBuffer, sortedInstances);
  };

  // Depth texture is recreated whenever the canvas backing size changes.
  let depthTexture: GPUTexture | null = null;
  const ensureSize = (): void => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (depthTexture && canvas.width === w && canvas.height === h) return;
    canvas.width = w;
    canvas.height = h;
    depthTexture?.destroy();
    depthTexture = device.createTexture({
      size: [w, h],
      format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  };

  const frame = (): void => {
    ensureSize();
    const view = VIEWS[viewIndex];
    const viewProj = camera.update(canvas.width / canvas.height);
    const focal = (0.5 * canvas.height) / Math.tan(0.5 * camera.fovY);
    cameraData.set(viewProj, 0);
    cameraData.set(camera.viewMatrix, 16);
    cameraData[32] = scene.center[0];
    cameraData[33] = scene.center[1];
    cameraData[34] = scene.center[2];
    cameraData[35] = view === 'normals' ? 1 : 0; // renderMode
    cameraData[36] = focal;
    cameraData[37] = focal;
    cameraData[38] = canvas.width;
    cameraData[39] = canvas.height;
    writeBuffer(device, cameraBuffer, cameraData);

    const drawSplats = view === 'splats' || view === 'normals';
    if (drawSplats) {
      sortSplats();
      packLighting();
    }

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.043, g: 0.051, b: 0.063, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthTexture!.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    pass.setPipeline(gridPipeline);
    pass.setBindGroup(0, gridBindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(grid.vertexCount);

    if (drawSplats) {
      pass.setPipeline(splatPipeline);
      pass.setBindGroup(0, splatBindGroup);
      pass.setVertexBuffer(0, quadBuffer);
      pass.setVertexBuffer(1, scene.splatInstanceBuffer);
      pass.draw(6, scene.cloud.count);
    } else if (view === 'ellipsoids') {
      pass.setPipeline(ellipsoidPipeline);
      pass.setBindGroup(0, ellipsoidBindGroup);
      pass.setVertexBuffer(0, sphereVertexBuffer);
      pass.setVertexBuffer(1, scene.instanceBuffer);
      pass.setIndexBuffer(sphereIndexBuffer, 'uint16');
      pass.drawIndexed(sphere.indexCount, scene.cloud.count);
    } else {
      pass.setPipeline(pointsPipeline);
      pass.setBindGroup(0, pointsBindGroup);
      pass.setVertexBuffer(0, scene.pointBuffer);
      pass.draw(scene.cloud.count);
    }
    pass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  };

  requestAnimationFrame(frame);
}

main().catch((err: unknown) => {
  console.error(err);
  const el = document.getElementById('error');
  if (el) {
    el.style.display = 'grid';
    el.textContent = err instanceof Error ? err.message : String(err);
  }
});
