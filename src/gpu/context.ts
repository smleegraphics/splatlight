// [plumbing] WebGPU device + canvas init. Nothing splatting-specific here:
// ask the browser for a GPU adapter, get a logical device from it, and wire the
// canvas up as a surface we can render into.

export interface GpuContext {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  /** Preferred swap-chain texture format for this platform (usually bgra8unorm). */
  format: GPUTextureFormat;
}

export async function initWebGPU(canvas: HTMLCanvasElement): Promise<GpuContext> {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not available in this browser. Try Chrome/Edge 113+ or Safari 18+.');
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('No suitable GPU adapter found.');
  }

  const device = await adapter.requestDevice();
  // Surface lost (driver reset, tab backgrounded too long, etc.) — log it so we
  // are not mystified by a frozen canvas later.
  device.lost.then((info) => {
    console.error(`WebGPU device lost: ${info.reason} — ${info.message}`);
  });

  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new Error('Failed to get a WebGPU canvas context.');
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  return { device, canvas, context, format };
}
