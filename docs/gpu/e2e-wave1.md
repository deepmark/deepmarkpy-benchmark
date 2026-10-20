# First end-to-end runs: the benchmark on a GPU deployment

Run 2026-10-19 on MIG 2 (`2g.35gb`), branch `feat/gpu-support`, using the real CLI path:

    deepmark-services up --config cfg.json [--gpu <MIG UUID>]
    deepmark-benchmark --config cfg.json

The config ran AudioSeal against all 47 attacks on the 20 smoke files, seed 1234.

## What ran where

`deepmark-services` placed 5 services on the GPU, each by its shipped-engine PASS on this GPU model: AudioSeal, DAC, BigVGAN (float64), SE1 and SE2. Everything else ran on the CPU:

| Services | Why on CPU |
|---|---|
| EnCodec, XCodec2 | Parity FAIL, so CPU by decision |
| Diffusion, VAE | Stochastic, not yet verified |
| NetworkTransmission, Opus | No GPU path |

Confirmed independently:
- NVML showed exactly 5 processes holding memory on MIG 2 (8.1 GB in total): AudioSeal 444 MiB, DAC 2.0 GB, BigVGAN 2.6 GB, SE1 452 MiB, SE2 2.5 GB. Their PIDs are those of the five GPU-placed containers.
- `run_metadata.json` records the deployment: GPU UUID and name, and per service the device, image, precision and verdict files.

## Speed

| Deployment | Wall, 20 files | Per file |
|---|---|---|
| All CPU | ~23.5 min | ~70 s |
| 5 services on GPU | ~11 min | ~33 s |

That is 2.1×, matching the study. Nothing runs in parallel yet. The rest is host-side work (metrics, native attacks, NetworkTransmission's real-time pacing), which is what the pipelined executor (Phase 4) addresses.

## Numbers: CPU deployment vs GPU deployment (per file, 47 attacks)

- **35 attacks: identical per-file accuracy on all 20 files.** These include the two deterministic GPU services, DAC and BigVGAN, and every detection, made by AudioSeal's detector on the GPU. Quality metrics move by |ΔPESQ| ~1e-5 to 1e-4. The watermarked audio itself: |ΔPESQ| ≤ 4.9e-5.
- **Discontinuous attacks**, whose input, the GPU-watermarked audio, differs by ~1e-5:
  - Codec2 at 700/1200/2400: 9–10 files changed, means ±0.6 pp, as in the study;
  - Opus: 2 files changed, mean unchanged;
  - XCodec2, on the CPU: 1 file changed, −0.9 pp.
- **Server-side stochastic attacks** differ on every run, CPU or GPU: Diffusion, VAE and NetworkTransmission (−0.6 to +0.9 pp).
- **SE1 and SE2 moved −4.7 and −5.0 pp.** Both add noise from an unseeded server-side generator. Measured with two runs per device (AudioSeal + SE1 + SE2, same seed):

  | Service | Same device, run to run | CPU vs GPU |
  |---|---|---|
  | SE1 | CPU/CPU +3.4 pp (14/20 files); GPU/GPU −2.5 pp (17/20) | −2.5 to +3.4 pp |
  | SE2 | CPU/CPU −3.8 pp (17/20); GPU/GPU −0.6 pp (13/20) | +0.3 to +4.7 pp |

  The CPU-vs-GPU differences are inside the run-to-run spread. With the noise seeded, the parity gate measured SE1 and SE2 on GPU equal to CPU to ~1e-6. Per-request seeds (plan Phase 3, decision D2) make both deterministic, so they can be compared file by file.

Raw results and the comparison script are kept on the development host.
