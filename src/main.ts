// [plumbing] Entry point: initialise WebGPU, build the line pipeline for the
// reference grid, and run the render loop driven by the orbit camera. As phases
// progress, the splat pipeline gets added alongside this grid.

import { initWebGPU } from './gpu/context';
import { createBufferWithData, writeBuffer } from './gpu/buffers';
import { OrbitCamera } from './camera/orbit-camera';
import { makeReferenceGrid } from './scene/reference-grid';
import { makeSyntheticCloud, cloudToPointVertices, cloudToInstanceData } from './scene/splat-data';
import { makeUnitSphere } from './scene/unit-sphere';
import lineShaderSrc from './shaders/line.wgsl?raw';
import pointsShaderSrc from './shaders/points.wgsl?raw';
import ellipsoidShaderSrc from './shaders/ellipsoid.wgsl?raw';

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

const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';

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

  // Camera uniform: a single mat4x4f = 16 floats = 64 bytes.
  const cameraBuffer = device.createBuffer({
    size: 64,
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

  // Splat cloud → one flat-colored point per Gaussian. Synthetic for now; swap in
  // parseSplat(fileBuffer) here to load a real .splat. Same camera, point topology.
  const cloud = makeSyntheticCloud();
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

  // View toggle: 'v' switches between the ellipsoid debug view and flat points.
  let showEllipsoids = true;
  window.addEventListener('keydown', (e) => {
    if (e.key === 'v' || e.key === 'V') showEllipsoids = !showEllipsoids;
  });

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
    const viewProj = camera.update(canvas.width / canvas.height);
    writeBuffer(device, cameraBuffer, viewProj);

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

    if (showEllipsoids) {
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
