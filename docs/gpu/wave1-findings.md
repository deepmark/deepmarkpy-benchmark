# Phase 0 and wave 1 on MIG 2 (2g.35gb): findings

Run 2026-10-19. Branch `feat/gpu-support` on the `v3.0.0` baseline; upstream `main` cannot build its images (dependabot pins).

Slice: `<MIG UUID>`, H200 NVL `2g.35gb`. TF32 is off and TorchScript GPU fusion is off. CPU references run on 8 threads.

Parity corpus (31 cases, hash `725e8866…`):
- 20 speech clips;
- 6 edge lengths, including 63,999/64,000/64,001 samples;
- silence, −60 dBFS, clipped, noise and a sweep.

Attacks are fed the CPU AudioSeal embed of the corpus, so they all see the same input.

## Verdicts

| Service | Verdict | Worst numbers vs CPU | Engine time CPU → GPU |
|---|---|---|---|
| AudioSeal embed | **PASS** NUMERIC | max \|d\| 9.0e-6; relative L2 2.3e-6 (speech), 2.2e-5 (watermark alone); 0 bit flips | 116 → 34 ms |
| AudioSeal detect | **PASS** NUMERIC | 62 cases, 0 bit flips, \|Δconf\| within 1e-4 | — |
| SE1 | **PASS** NUMERIC | max \|d\| 4.7e-6; relative L2 2.5e-6 | 167 ms → (GPU) |
| SE2 | **PASS** NUMERIC | max \|d\| 2.3e-5; relative L2 9.1e-7 | 11.8 s → (GPU) |
| EnCodec | FAIL as specified; proven tie-breaks | see below | 193 ms → (GPU) |
| DAC | FAIL as specified; proven tie-breaks | see below | 5.0 s → (GPU) |
| BigVGAN | FAIL NUMERIC (max \|d\| 1.4e-4 > 1e-4) | GPU float32 is less accurate than CPU float32 | 8.9 s → 0.53 s (f32), 0.85 s (f64) |
| XCodec2 | FAIL NUMERIC | the CPU float32 reference itself is 2.4e-4 from exact | — |

## What the failures are

**Codebook codecs: EnCodec and DAC.** `parity/codec_probe.py` recorded every codebook decision on CPU and GPU, and decoded the GPU's codes on the CPU.
- Hooked runs equal the parity runs bit for bit.
- **EnCodec:** 12 of 12,869 codec frames (0.09%) choose a different code.
- **DAC:** 2 of 14,789 (0.014%).
- **Every first-differing choice is causally explained.** The CPU's preference for its code is within the effect of the observed quantiser-input difference, or within float32 rounding of |x|² − 2x·e + |e|².
- In one EnCodec case the margin is negative: there the GPU's code is the nearer one in exact arithmetic.
- **Given the same codes**, GPU and CPU decode to max \|d\| 4.5e-6.
- The audio "flip frames" span several 20 ms frames per codec frame, because EnCodec's LSTM decoder spreads one code over 4–17 audio frames.
- **Two clauses of the approved QUANTIZED tier are ill-posed:**
  - The cap is "≤ 1% of audio frames", but EnCodec reaches 4.8% from a single code.
  - The bound is "|ΔPESQ| ≤ 5e-3 per case". A proven tie-break on the −60 dBFS clip moves PESQ by 0.064, because PESQ normalises level, so 1% of a quiet signal counts fully.

**Near-silent references.** DAC's "quiet" (−74 dBFS RMS) and "silence" (−78 dBFS, the watermark alone) cases fail relative L2 at 4.7e-5 and 8.2e-5. Their absolute differences are 4e-8 (−148 dBFS). At those levels a relative bound of 3e-5 demands −160 dBFS agreement, which is below float32 resolution of the networks' internal activations.

**BigVGAN.** Measured against a float64 run of the same weights on the same mel input:
- CPU float32 error: max 2.4e-5 to 4.8e-5 (relative ~4e-6).
- GPU float32 error: max up to 1.4e-4 (relative up to 2.1e-5), 1.4–6× the CPU's.
- Disabling cuDNN does not help (error equal or larger, 3.3× slower), so this is GPU float32 arithmetic, not algorithm choice.
- **In float64 on the GPU**, BigVGAN matches the float64 reference to ~1e-13, at 0.85 s per call: 1.6× GPU float32, ~10× faster than CPU float32. Its difference from the CPU reference is then exactly the CPU's own float32 error (max 4.8e-5, relative ≤ 1.1e-5), which **passes NUMERIC**.

**XCodec2.** Its FSQ quantiser makes **no** different decision (0 of 8,608 frames), so its differences are not tie-breaks. They arise in the Vocos decoder backbone: backbone output max \|d\| 3.5e-3 (relative 1.8e-5). On the worst clip, measured against a float64 decoder:
- CPU float32: max 6.8e-4, relative 2.4e-4;
- GPU float32: max 8.5e-4, relative 3.0e-4;
- GPU float64: ~1e-12.

The decoder is ill-conditioned: float32 on either device is ~1e-4 from exact. No GPU configuration can agree with the CPU float32 reference to 1e-4 on that clip, because the reference does not hold that accuracy itself. Two decoded bits flip on the 30 s clip; the watermark there is near chance after XCodec2.

## Infrastructure findings

- **DAC was non-repeatable on GPU.** TorchScript's profiling executor switched its scripted `snake` activation to a fused kernel after the first call, so the first call differed by 1.8e-6. Strict numerics now disable GPU fusion. Every call then agrees, and the DAC CPU reference is EXACT-unchanged.
- **CPU results already vary by host.** EnCodec on 16 vs 8 CPU threads differs at median relative 4e-7 (max \|d\| 2e-6). Production CPU services use all cores, so published CPU numbers depend on the machine at this level.
- **XCodec2's "random encoder" (study §4.1) was a false alarm** (commit `4d0bc5a`).
- Upstream: a pending dependabot set breaks every image build. See the session notes.
