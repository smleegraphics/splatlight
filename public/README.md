# Splat test assets

Splat files (`*.ply`, `*.splat`) live here and are served at the site root
(e.g. `/luigi.ply`). They are **git-ignored** — don't commit third-party captures.
If none is present, the app falls back to a synthetic sphere.

## Default test scene

`luigi.ply` — a small, object-centric 3DGS capture (~14.5K Gaussians, DC-only, no
higher-order SH), from the public [`dylanebert/3dgs`](https://huggingface.co/datasets/dylanebert/3dgs) dataset. Fetch it with:

```sh
curl -sL "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/luigi/luigi.ply" \
  -o public/luigi.ply
```

## Using your own scene

Drop any 3DGS `.ply` in here and point `loadCloud()` in `src/main.ts` at it.
Full-SH `.ply` files (with `f_rest_*` properties) are needed to see
view-dependent color.
