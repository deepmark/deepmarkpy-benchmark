# Spike S2: can the torch-2.7 services share one process?

Run 2026-10-19 against `v3.0.0` (`feat/gpu-support`). Each service's `constraints*.txt` is the frozen environment its image was captured with. Co-hosting engines in one process (plan §4.4, option O3) only keeps each service's numerics if one environment satisfies all of their pins.

## Result

Across the ten services on the torch-2.7.1 base, 130 packages are pinned and **19 are pinned at different versions**. The ones that can change numbers:

| Package | Versions |
|---|---|
| numpy | 1.26.4 (AWARE), 2.2.6 (Perth), 2.4.6 (all others) |
| librosa | 0.9.2 (AWARE), 0.11.0 (all others) |
| numba | 0.59.0 (AWARE), 0.66.0 (all others) |
| soundfile | 0.12.1 (AWARE), 0.13.1 (Perth), 0.14.0 (all others) |
| huggingface-hub | 0.24.7 (BigVGAN runtime stage), 1.25.1 (all others, including BigVGAN's builder stage) |
| protobuf | 3.19.6 (DAC), 7.35.1 (BigVGAN builder stage only) |

The other 13 are the serving stack and tooling: fastapi, pydantic, uvicorn, starlette, matplotlib, pyyaml, scikit-learn and so on.

**Seven services have no conflicting pin at all:** AudioSeal, WavMark, VAE, EnCodec, DAC, Diffusion and SE1. Their union is a consistent environment in which no package version changes for any of them. Adding BigVGAN's runtime stage introduces exactly one conflict, huggingface-hub 0.24.7 against 1.25.1. That matters only for loading weights, but it is untested.

## Conclusion

- **Full group: NO-GO** as frozen. AWARE and Perth would change numpy/librosa.
- **Core seven: feasible with no version change.** Expected numerics are identical, and the gate is EXACT: the co-hosted engine must match its own service image bit for bit, on CPU and GPU.
- BigVGAN can join only after a load test with huggingface-hub 1.25.1, gated EXACT.
- XCodec2 (torch 2.4.1), SE2 (2.13.0), SilentCipher and TimbreWM (2.0.0) are separate runtimes by construction.

Since S1 showed CUDA IPC works across containers in one pod, co-hosting is **not needed for zero-copy handoff**. What it would still add:
- kernels overlapping across engines, where separate processes time-slice;
- one CUDA context instead of seven.

MPS (S3) may give the first without merging environments. Whether either is worth it is decided once the GPU lane is measured as the bottleneck (Phase 4 and later).
