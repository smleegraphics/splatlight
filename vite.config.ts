import { defineConfig } from 'vite';

// `base: './'` makes built asset paths relative, which keeps a static deploy
// (GitHub Pages / Vercel) working from any sub-path. Everything else is default.
export default defineConfig({
  base: './',
  server: { port: 5173, open: true },
});
