// [plumbing] Small GPU buffer helpers used across the project.

/**
 * Thin wrapper over `device.queue.writeBuffer`.
 *
 * Why this exists: TypeScript 5.7+ types a bare `Float32Array` as
 * `Float32Array<ArrayBufferLike>`, but `@webgpu/types` wants an ArrayBuffer-backed
 * view (`ArrayBufferView<ArrayBuffer>`). Every array we hand to the GPU
 * (wgpu-matrix matrices, parsed splat data) is ArrayBuffer-backed at runtime, so
 * we assert that fact once, here, instead of at every call site.
 */
export function writeBuffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  data: ArrayBufferView,
  offset = 0,
): void {
  device.queue.writeBuffer(buffer, offset, data as unknown as GPUAllowSharedBufferSource);
}

/** Create a buffer and immediately upload `data` into it. */
export function createBufferWithData(
  device: GPUDevice,
  data: ArrayBufferView,
  usage: GPUBufferUsageFlags,
  label?: string,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  writeBuffer(device, buffer, data);
  return buffer;
}
