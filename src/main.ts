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

  // Camera uniform: viewProj(16) + view(16) + focal(2) + viewport(2) = 36 floats.
  const cameraData = new Float32Array(36);
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

  // 2D splat billboards: project each Gaussian to a screen ellipse (EWA) and
  // alpha-blend it. A static quad (per vertex) + splat instance data.
  const quadBuffer = createBufferWithData(
    device,
    new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]),
    GPUBufferUsage.VERTEX,
    'splat-quad',
  );
  const splatInstanceBuffer = createBufferWithData(
    device,
    cloudToSplatInstances(cloud),
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
  const VIEWS = ['splats', 'ellipsoids', 'points'] as const;
  let viewIndex = 0;
  window.addEventListener('keydown', (e) => {
    if (e.key === 'v' || e.key === 'V') viewIndex = (viewIndex + 1) % VIEWS.length;
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
    const focal = (0.5 * canvas.height) / Math.tan(0.5 * camera.fovY);
    cameraData.set(viewProj, 0);
    cameraData.set(camera.viewMatrix, 16);
    cameraData[32] = focal;
    cameraData[33] = focal;
    cameraData[34] = canvas.width;
    cameraData[35] = canvas.height;
    writeBuffer(device, cameraBuffer, cameraData);

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

    const view = VIEWS[viewIndex];
    if (view === 'splats') {
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
