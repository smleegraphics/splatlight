// [plumbing] Entry point: initialise WebGPU, build the line pipeline for the
// reference grid, and run the render loop driven by the orbit camera. As phases
// progress, the splat pipeline gets added alongside this grid.

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

/** Point the camera at the cloud's centroid and back off to frame its extent. */
function fitCameraToCloud(camera: OrbitCamera, cloud: SplatCloud): void {
  const n = cloud.count;
  if (n === 0) return;
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
  camera.target[0] = cx;
  camera.target[1] = cy;
  camera.target[2] = cz;
  camera.distance = r * 2.2;
  camera.near = Math.max(0.001, r * 0.002);
  camera.far = Math.max(camera.far, r * 20);
}

async function main(): Promise<void> {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  const { device, context, format } = await initWebGPU(canvas);

  const camera = new OrbitCamera();
  camera.attach(canvas);

  // Reference grid geometry → vertex buffer.
  const grid = makeReferenceGrid();
  const vertexBuffer = createBufferWithData(
    device,
    grid.vertices,
    GPUBufferUsage.VERTEX,
    'grid-vertices',
  );

  // Camera uniform: viewProj(16) + view(16) + focal(2) + viewport(2) + renderMode(1)
  // + padding = 40 floats.
  const cameraData = new Float32Array(40);
  const cameraBuffer = device.createBuffer({
    size: cameraData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const shaderModule = device.createShaderModule({ code: lineShaderSrc });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: shaderModule, entryPoint: 'vs', buffers: [POS_COLOR_LAYOUT] },
    fragment: { module: shaderModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'line-list' },
    depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: cameraBuffer } }],
  });

  // Load a real captured scene (falls back to the synthetic sphere on failure),
  // then frame the camera to it.
  const cloud = await loadCloud();
  fitCameraToCloud(camera, cloud);
  const pointBuffer = createBufferWithData(
    device,
    cloudToPointVertices(cloud),
    GPUBufferUsage.VERTEX,
    'splat-points',
  );
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

  // Ellipsoid debug view: draw each splat as its 3D covariance shape (M = R*S on a
  // unit sphere), instanced. Toggle against the flat points with 'v'.
  const sphere = makeUnitSphere();
  const sphereVertexBuffer = createBufferWithData(
    device,
    sphere.positions,
    GPUBufferUsage.VERTEX,
    'unit-sphere-verts',
  );
  const sphereIndexBuffer = createBufferWithData(
    device,
    sphere.indices,
    GPUBufferUsage.INDEX,
    'unit-sphere-indices',
  );
  const instanceBuffer = createBufferWithData(
    device,
    cloudToInstanceData(cloud),
    GPUBufferUsage.VERTEX,
    'splat-instances',
  );
  const ellipsoidModule = device.createShaderModule({ code: ellipsoidShaderSrc });
  const ellipsoidPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: ellipsoidModule,
      entryPoint: 'vs',
      buffers: [SPHERE_LAYOUT, INSTANCE_LAYOUT],
    },
    fragment: { module: ellipsoidModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
  });
  const ellipsoidBindGroup = device.createBindGroup({
    layout: ellipsoidPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: cameraBuffer } }],
  });

  // 2D splat billboards: project each Gaussian to a screen ellipse (EWA) and
  // alpha-blend it. A static quad (per vertex) + splat instance data.
  const quadBuffer = createBufferWithData(
    device,
    new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]),
    GPUBufferUsage.VERTEX,
    'splat-quad',
  );
  const splatInstances = cloudToSplatInstances(cloud);
  const splatInstanceBuffer = createBufferWithData(
    device,
    splatInstances,
    GPUBufferUsage.VERTEX,
    'splat-instances-2d',
  );
  const splatModule = device.createShaderModule({ code: splatShaderSrc });
  const splatPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: splatModule,
      entryPoint: 'vs',
      buffers: [QUAD_LAYOUT, SPLAT_INSTANCE_LAYOUT],
    },
    fragment: {
      module: splatModule,
      entryPoint: 'fs',
      targets: [
        {
          format,
          // Premultiplied-alpha "over" compositing.
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-list' },
    // Transparent: test against the grid, but don't write depth (so splats blend).
    depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' },
  });
  const splatBindGroup = device.createBindGroup({
    layout: splatPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: cameraBuffer } }],
  });

  // View cycle: 'v' rotates through the 2D splats, the 3D ellipsoids, and points.
  const VIEWS = ['splats', 'normals', 'ellipsoids', 'points'] as const;
  let viewIndex = 0;
  window.addEventListener('keydown', (e) => {
    if (e.key === 'v' || e.key === 'V') viewIndex = (viewIndex + 1) % VIEWS.length;
  });

  // [concept] Depth sort: "over" blending is order-dependent, so splats must be
  // drawn back-to-front. We quantize each splat's camera-space depth to a 16-bit
  // bucket and counting-sort — O(n) with no per-comparison function calls — then
  // rewrite the instance buffer in that order. Only re-sorts when the camera moved.
  const SPLAT_FLOATS = 14;
  const NUM_BUCKETS = 65536; // 16-bit depth quantization
  const order = new Uint32Array(cloud.count);
  const depths = new Float32Array(cloud.count);
  const buckets = new Uint16Array(cloud.count);
  const counts = new Uint32Array(NUM_BUCKETS);
  const sortedInstances = new Float32Array(splatInstances.length);
  let lastSortKey = '';

  const sortSplats = (): void => {
    const key = `${camera.azimuth}|${camera.elevation}|${camera.distance}|${camera.target[0]}|${camera.target[1]}|${camera.target[2]}`;
    if (key === lastSortKey) return;
    lastSortKey = key;

    // 1. Camera-space z per splat (row 2 of the view matrix), tracking the range.
    const m = camera.viewMatrix;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < cloud.count; i++) {
      const p = i * 3;
      const z =
        m[2] * cloud.positions[p] +
        m[6] * cloud.positions[p + 1] +
        m[10] * cloud.positions[p + 2] +
        m[14];
      depths[i] = z;
      if (z < min) min = z;
      if (z > max) max = z;
    }

    // 2. Quantize depth → 16-bit bucket. Front is z<0, so smaller z (smaller
    //    bucket) = farther = drawn first = back-to-front.
    const scale = max > min ? (NUM_BUCKETS - 1) / (max - min) : 0;
    for (let i = 0; i < cloud.count; i++) {
      buckets[i] = Math.min(NUM_BUCKETS - 1, ((depths[i] - min) * scale) | 0);
    }

    // 3. Counting sort into `order` — no comparisons.
    counts.fill(0);
    for (let i = 0; i < cloud.count; i++) counts[buckets[i]]++; // tally per bucket
    let running = 0; // prefix sum → each bucket's start offset
    for (let b = 0; b < NUM_BUCKETS; b++) {
      const c = counts[b];
      counts[b] = running;
      running += c;
    }
    for (let i = 0; i < cloud.count; i++) {
      order[counts[buckets[i]]++] = i; // place splat i, advance its bucket's slot
    }

    // 4. Gather the instance data into sorted order and upload.
    for (let i = 0; i < cloud.count; i++) {
      const src = order[i] * SPLAT_FLOATS;
      const dst = i * SPLAT_FLOATS;
      for (let k = 0; k < SPLAT_FLOATS; k++) {
        sortedInstances[dst + k] = splatInstances[src + k];
      }
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
    cameraData[32] = focal;
    cameraData[33] = focal;
    cameraData[34] = canvas.width;
    cameraData[35] = canvas.height;
    cameraData[36] = view === 'normals' ? 1 : 0; // renderMode
    writeBuffer(device, cameraBuffer, cameraData);

    const drawSplats = view === 'splats' || view === 'normals';
    if (drawSplats) sortSplats();

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

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(grid.vertexCount);

    if (drawSplats) {
      pass.setPipeline(splatPipeline);
      pass.setBindGroup(0, splatBindGroup);
      pass.setVertexBuffer(0, quadBuffer);
      pass.setVertexBuffer(1, splatInstanceBuffer);
      pass.draw(6, cloud.count);
    } else if (view === 'ellipsoids') {
      pass.setPipeline(ellipsoidPipeline);
      pass.setBindGroup(0, ellipsoidBindGroup);
      pass.setVertexBuffer(0, sphereVertexBuffer);
      pass.setVertexBuffer(1, instanceBuffer);
      pass.setIndexBuffer(sphereIndexBuffer, 'uint16');
      pass.drawIndexed(sphere.indexCount, cloud.count);
    } else {
      pass.setPipeline(pointsPipeline);
      pass.setBindGroup(0, pointsBindGroup);
      pass.setVertexBuffer(0, pointBuffer);
      pass.draw(cloud.count);
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
