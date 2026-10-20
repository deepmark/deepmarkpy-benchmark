# GPU implementation plan

> **Status and decisions, 2026-10-19 (end of Phase 0 / wave 1).** Details: `wave1-findings.md`, `spike-cuda-ipc.md`, `spike-cohosting.md`.
>
> - **Baseline:** the `v3.0.0` tag, on branch `feat/gpu-support`. Upstream `main` cannot build images because of dependabot pins. Rebase once that is fixed; verdicts are keyed by image ID.
> - **Placement after wave 1** (changes to §2 below):
>   - **GPU:** AudioSeal (embed and detect), SE1, SE2 all PASS. BigVGAN goes to GPU **in float64**: user decision; exact, and ~10× faster than CPU float32.
>   - **CPU (user decision):** EnCodec, because it fails the approved QUANTIZED rule as written, and XCodec2, because its float32 decoder is ill-conditioned on any device.
>   - **Pending:** DAC. Its two code flips meet the approved near-tie rule; the verdict follows from the probe run.
> - **Verdicts are local, not committed** (2026-10-20). They count only for the image IDs and GPU model they were measured on, so each machine runs the gate itself. They live in `build/verdicts/`; §3.4's "committed with the change" is superseded.
> - **Tolerance amendments** (to §3.2): relative checks pass when RMS error ≤ max(bound × reference RMS, 1e-6), a −120 dBFS floor. The QUANTIZED rule stays as approved, now fed by `codec_probe.py` evidence.
> - **Numerics** (add to §3.2): strict mode also disables TorchScript GPU fusion. Otherwise DAC's first GPU call differed from later ones.
> - **Corrections:**
>   - D3 (XCodec2's "random encoder") was a false alarm. Only the revision pin and a load check were added.
>   - D-Replay landed as `fftconvolve` with the IR in float64: 1e-15 from direct convolution.
> - **Spikes:**
>   - S1 CUDA IPC: GO, provided the containers are in one pod with shared IPC and PID namespaces; works across torch versions.
>   - S2 co-hosting: feasible for 7 services with no version change.
>   - S3 MPS: deferred. A host-level daemon on a shared machine needs a private pipe directory, so it needs the user's go-ahead.

Proposal, 2026-10-19. Builds on `results/RESULTS-docker.md` (phase 2 study, 2026-10-01), `results/raw/parity/`, and the code at `origin/main` `fee91af`.

It covers:
- which models and attacks move to the GPU;
- how each move is proven numerically correct before it's used;
- how the GPU and CPU parts connect (transport, CUDA IPC, transfers);
- how work is scheduled;
- what changes in the code, and in what order.

---

## 0. Summary

1. **On the GPU:** AudioSeal plus the 8 AI attacks first: DAC, BigVGAN, XCodec2, EnCodec, SE1, SE2, Diffusion, VAE. Then SilentCipher and TimbreWM. WavMark, Perth and NISQA go after they've been measured. AWARE is optional and detect-only. **Everything else stays on the CPU:** 37 native attacks, NetworkTransmission, Opus, PESQ/ViSQOL/STOI.
2. **No component runs on the GPU in a benchmark without a recorded parity PASS** against its CPU image on a fixed corpus. The placement layer enforces this, not convention. Pure refactors (transport, scheduling, caching) must be **bit-exact** (max |Δ| = 0).
3. **"Error < 1e-5" needs a precise definition.** Section 3.2 gives one. As *relative* L2 error (≈100 dB) it is met in the median by every deterministic service with TF32 off. As a worst-case *absolute* bound it would reject AudioSeal (3.1e-5), BigVGAN (3.1e-5) and XCodec2 (3.9e-5), even though their decoded bits and metrics are identical to CPU. I propose max |Δ| ≤ 1e-4, relative L2 ≤ 1e-5 (median) / 3e-5 (worst), and zero downstream decision flips.
4. **The GPU is no longer the bottleneck. CPU work is.** After the GPU move, 80% of the per-file time is host-side: metrics 48%, native attacks 31% (study §6). The largest gain comes from a **pipelined executor** that runs CPU work concurrently with GPU work, across files.
   - Projection, to be re-measured in Phase 0: ~4–5 s per file, so **~15–25 min for the full 245-file run**.
   - For comparison: 2.3 h with GPU but sequential, 5.0 h on CPU.
5. **Transfers are cheap; serialization is not.**
   - A clip is 0.3–0.6 MB. PCIe moves that in ~30 µs, and the whole file's ~75 crossings take ~3 ms.
   - Today each crossing goes through base64 + JSON + HTTP + pydantic, and costs ~10–15 ms. AudioSeal detect is 29 ms round trip against 13.6 ms of engine time.
   - So "efficient CPU↔GPU transfer" means: a binary transport, batching small calls, and overlapping decode with compute.
   - **CUDA IPC** removes only the remaining ~30 µs per hop. It is designed here as a pluggable transport, gated on a feasibility spike and a measured benefit.
6. **Prerequisite: per-task seeding.** Results must not depend on execution order, concurrency or device. This changes per-file random draws compared with v2.0.0, so it is a decision for you (§8). Legacy sequential mode stays available to reproduce published numbers.

---

## 1. Where things stand (changes since the study)

| Fact | Consequence |
|---|---|
| MIG was repartitioned. MIG 2 is now **`2g.35gb`, `<MIG UUID>`**: 32 SMs, ~33 GB. It was 1g.18gb, 16 SMs, 16 GB. | All study speed numbers need re-baselining. The memory budget roughly doubles: all GPU candidates (~15–16 GB) fit with room for batching. |
| The study's own tooling (not in this repository) hard-coded the old slice's UUID. | A device is now named in one place: `--gpu` (index or UUID), default `DEEPMARK_GPU_DEVICE`. |
| The CDI spec (`/var/run/cdi/nvidia.yaml`, regenerated Oct 7) already lists the new UUID. | `--device nvidia.com/gpu=MIG-50bf…` works under rootless Podman with no admin action. |
| The checkout is at `7c1c7be`, 2 behind `origin/main` `fee91af`. `e5ae099` makes the base image **`torch==2.7.1+cpu`**, adds cache volumes and `mem_limit`s, adds a **NISQA service**, and fixes WaveletAttack's read-only input. | Rebase first. Every GPU service now needs an explicit GPU image variant; the study's "production images already carry CUDA" is no longer true. |
| Contract fixtures (`tests/fixtures/contracts/`) are SHA-256 hashes of CPU response bodies. | GPU outputs can never match them. They remain the **CPU** contract, and the GPU gets its own parity verdicts (§3). |
| `tests/test_cpu_only_builds.py` and `CLAUDE.md` say "CPU only". | That policy becomes "CPU by default, GPU as verified opt-in variants" (§8, D7). |
| There is no CI (`.github/` is absent). | Gates run locally on MIG 2 through one command (§3.4). |
| Host: 64 threads, load average ~20 from other users. | CPU parallelism defaults to 8 workers, configurable. |

---

## 2. What runs where

Speed figures are from study §5.2 (1g slice; to be re-measured on 2g). Parity figures are from §11 with TF32 off.

| Component | Device | Speedup vs CPU (client) | Parity, TF32 off (worst clip) | Needed to get there | Gate tier (§3.2) |
|---|---|---|---|---|---|
| **AudioSeal** embed/detect | **GPU, wave 1** | detect 64→13.6 ms; ~55 calls per file | 0 / 16,000 bit flips; Δconf 6.3e-5; max\|Δ\| 3.1e-5; 104 dB | Device placement, `inference_mode`, MIG cuDNN fallback (`gpu_engines.py:21-73`) | NUMERIC |
| **DAC** | GPU, wave 1 | 5,255→311 ms (17×) | 46 dB worst (VQ near-tie), 123 dB median | GPU image | QUANTIZED |
| **BigVGAN** | GPU, wave 1 | 10.1 s→0.9 s (11×) | 98.5 dB, max\|Δ\| 3.1e-5 | GPU image | NUMERIC |
| **XCodec2** | GPU, wave 1 | 2.1 s→0.27 s (8×) | 97.7 dB | GPU image; explicit input `.to(device)` (`inference.py:34` doesn't move the input); **decision D3** (random encoder) first | NUMERIC |
| **SE2** | GPU, wave 1 | 17.4 s→1.34 s (13×) | 126 dB (seeded) | cu130 torch image; ClearVoice MIG device pick | NUMERIC |
| **EnCodec** | GPU, wave 1 | 182→45 ms | 61 dB worst (VQ), 120 dB median | GPU image | QUANTIZED |
| **SE1** | GPU, wave 1 | 193→51 ms | 131 dB (seeded) | `.cpu()` before `.numpy()` | NUMERIC |
| **Diffusion** | GPU, wave 2 | 7.0 s→2.4 s | Not repeatable today (OS-entropy seed per request) | Request seed (D6); CPU `torch.Generator` so noise matches across devices | NUMERIC if the CPU generator holds, else STOCHASTIC |
| **VAE (RAVE)** | GPU, wave 2 | 96→25 ms | Distributional only (TorchScript draws on the device RNG) | Reset streaming state per request (D5) | STOCHASTIC |
| **SilentCipher** | GPU, wave 3 | 784/1,366→101/166 ms | Messages identical over 6 rounds; full parity not yet run | `+cu118` image (cu117 has no sm_90) | NUMERIC |
| **TimbreWM** | GPU, wave 3 | 552/454→62/55 ms | Not yet run | `+cu118` image | NUMERIC |
| WavMark, Perth | Measure first | Not tested | — | GPU image | NUMERIC |
| NISQA (opt-in metric, now a service upstream) | Measure first | — | — | GPU image | NUMERIC on scores |
| AWARE | **detect on GPU only (optional)**; embed stays CPU | detect 21→3.3 ms; embed *slower* on GPU (4.9→5.5 s) | — | Engine must control its device (the package self-places) | NUMERIC |
| NetworkTransmission | **CPU** | Real-time paced: 5.6 s/file is waiting, not compute | — | Scale with **replicas** (each container has its own netem namespace) | — |
| Opus, 37 native attacks | **CPU** | ≤ 0.5 s each except Replacement (2.15 s) | Goldens are bit-exact on CPU | Run in parallel on CPU (§4.6). GPU ports would break goldens for ≤ 4 s/file total. | — |
| PESQ / ViSQOL / STOI | **CPU** | Reference C implementations; no faithful GPU equivalent | — | Parallel CPU pool | — |

**VRAM on MIG 2 (33 GB):** waves 1–2 measured 11.0–11.2 GB together on the old slice. SilentCipher + TimbreWM + AWARE add ~3 GB; WavMark ~0.8 GB. That leaves ~17 GB for batching workspaces and a second replica.

---

## 3. Verification: the gate before every change

### 3.1 Four gates, one per kind of change

| Gate | Applies to | Pass criterion |
|---|---|---|
| **T: Transport** | wire format, shared memory, IPC, keep-alive, caching resamplers, reusing ClearVoice | **Bit-exact**: max \|Δ\| = 0 on every output |
| **D: Device** | moving one (service, op) from CPU to GPU, or changing batch shape on GPU | Tier tolerances below, measured against the CPU image on the parity corpus |
| **S: Scheduling** | executor, ordering, concurrency, sharding, replicas | **Bit-exact** results JSON against the sequential executor with the same per-task seeds, for every deterministic attack |
| **E: End to end** | after each wave, and before any default flips | 20-file run (then 245): deterministic attacks have **identical per-file accuracy**; chaotic/stochastic attacks stay within the CPU-vs-CPU spread (below) |

Isolation rule: gate D feeds **the same CPU-produced input** to the component under test, as the study did, so exactly one variable changes. Gate E then checks the composition.

### 3.2 Tolerances (what "< 1e-5" means here)

Measured, GPU with TF32 off against CPU (study §11, `raw/parity/waveform.md`):

| Service | max \|Δ\| | worst rel. L2 (= 10^(−SNR/20)) | median rel. L2 | Decoded bits | max \|ΔPESQ\| |
|---|---|---|---|---|---|
| AudioSeal embed | 3.1e-5 | 6.2e-6 | 1.8e-7 | identical | 2e-5 |
| BigVGAN | 3.1e-5 | 1.2e-5 | 9.4e-6 | identical | 6e-5 |
| XCodec2 | 3.9e-5 | 1.3e-5 | 5.1e-6 | identical | 1e-4 |
| SE1 / SE2 | 2.1e-7 / 4.5e-7 | 2.8e-7 / 4.8e-7 | — | identical | 4e-5 / 2.5e-4 |
| EnCodec / DAC | 4.3e-3 / 1.4e-2 | 9.2e-4 / 5.1e-3 (one VQ flip) | ~1e-6 | identical | 9e-5 / 4e-3 |

Proposed tiers (stored in `parity/tolerances.yaml`, per service and op, reviewed like goldens):

- **EXACT** (gates T and S): max \|Δ\| = 0.
- **NUMERIC** (deterministic float paths). All of the following:
  - max \|Δ\| ≤ **1e-4**;
  - rel. L2 ≤ **1e-5** median and ≤ **3e-5** worst clip;
  - identical output length, no NaN/Inf;
  - **0 decoded-bit flips**, using the CPU detector on the output;
  - for models: \|Δconfidence\| ≤ 1e-4, and the watermark residual decodes identically;
  - \|ΔPESQ\| ≤ 1e-3, \|ΔViSQOL\| ≤ 1e-3, \|ΔSTOI\| ≤ 1e-4;
  - **GPU run twice → bit-identical** (repeatability).
- **QUANTIZED** (EnCodec, DAC). NUMERIC on every 20 ms frame except "flip frames". Flip frames must be ≤ 1% of frames, and each must be a **codebook near-tie**: an instrumented CPU run shows the best/second-best code-distance margin < 1e-4 relative. This proves the difference is a tie-break, not a bug. Downstream bits identical, \|ΔPESQ\| ≤ 5e-3.
- **STOCHASTIC** (VAE; Diffusion if the CPU generator doesn't hold). Over ≥ 50 clips, GPU-vs-CPU differences in bit accuracy, PESQ, ViSQOL and STOI must be indistinguishable from CPU-seed-A vs CPU-seed-B. Two-sample KS p > 0.05, and the mean bit-accuracy difference must lie within the CPU–CPU 95% interval. This formalises how the study read §11.

**Why not 1e-5 absolute.**
- float32 inference through ~50 conv layers, with different kernel algorithms on GPU (cuDNN) and CPU (oneDNN), lands at 1e-5–1e-4 absolute even when every decision downstream is identical.
- 1e-4 is −80 dBFS. The decision-level checks (bits, confidence, metrics) are what protect the benchmark's numbers.
- If you want 1e-5 absolute anyway, AudioSeal, BigVGAN and XCodec2 fail it today. No GPU precision setting fixes that, because the CPU reference is itself float32 and sits ~1e-5 from the exact value. Even fp64 on the GPU would be "different from CPU" by about that much.

Numerics settings, applied in code (`serving/device.py`) **and** by env (`NVIDIA_TF32_OVERRIDE=0`):
- `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`;
- `cudnn.benchmark = False`, `cudnn.deterministic = True`;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

The study shows that PyTorch's default (cuDNN TF32 on) alone causes bit flips (13 / 16,000) and drops the codecs to 20–40 dB.

### 3.3 Parity corpus

- 20 smoke clips (real speech, the study's list).
- Synthetic edge cases:
  - lengths 16,000; 57,143 (TimeStretch output); 63,999 / 64,000 / 64,001 (the MIG cuDNN boundary, §5.3); 80,000; 160,000; 480,000;
  - silence, −60 dBFS, full-scale clipping, white noise, a sine sweep.
- Models get fixed watermarks (`default_rng(1234)`). Attacks get the CPU-watermarked clip.
- References: CPU outputs from the CPU image, keyed by **image digest + torch version + git rev**. Stored outside git, on the machine that runs the gate, with a hash manifest in git.

### 3.4 The per-step procedure ("one change → one gate → one commit")

```
deepmark-parity capture  --service S --op OP --image <cpu image>                 # once per CPU image digest
deepmark-parity run      --service S --op OP --image <gpu image> --gpu $DEEPMARK_GPU_DEVICE --repeat 2
deepmark-parity verdict  --service S --op OP --tier NUMERIC                      # writes build/verdicts/S.OP.<digest>.json
deepmark-parity e2e      --runs A B --classes parity/attack_classes.yaml          # gate E
```

The verdict JSON records:
- all metrics per case, and the tier and thresholds used;
- PASS/FAIL;
- image digest, torch/cuDNN/driver versions, MIG profile and UUID, git rev.

It's committed with the change, and the commit message quotes PASS plus the worst-case numbers.

### 3.5 Enforcement

The placement layer (§5, `execution/placement.py`) reads `build/verdicts/`. A (service, op) may be placed on GPU only if a PASS verdict exists for the **digest of the image actually running**; the service reports its digest and device at `/capabilities`. An unverified GPU path fails the run at startup, naming the missing verdict. `run_metadata.json` records the verdict IDs used.

---

## 4. Architecture: connecting the dots

### 4.1 Per-file data flow and where crossings happen

```
x (clean, host) ──► EMBED (model, GPU) ──► w ──┬─► 8 GPU attacks ─► y_a ─┬─► DETECT (GPU) ─► bits/conf
                                               │                          └─► METRICS (CPU, needs x and y_a)
                                               └─► 41 CPU attacks ─► y_a ─┬─► DETECT (GPU)
                                                                          └─► METRICS (CPU)
extra: SameModel / Collusion / Collusion2 / CrossModel issue embeds (CrossModel also detects on a 2nd model)
```

With the model and AI attacks on GPU, the **minimum** number of host↔device crossings per file is fixed by placement:

| Item | Crossings |
|---|---|
| x → GPU | 1 |
| w → host | 1 |
| w → each GPU attack process | 8 (0 if co-hosted or IPC) |
| each GPU y_a → host | 8 |
| each GPU y_a → detector | 8 (0 if co-hosted or IPC) |
| each CPU y_a → detector | 41 |
| extra embeds | ~8 |
| **Total** | **~75 × 0.3–0.6 MB ≈ 40 MB/file** |

That's 2–4 ms of PCIe per file.

### 4.2 Where the time actually goes in a "transfer"

| Per hop | Today | Binary over HTTP | Shared memory | CUDA IPC |
|---|---|---|---|---|
| Copy | ~30 µs (pageable) | same | same | 0 host copies |
| Serialization | float64 → base64 (853 KB) → JSON → pydantic `max_length` check → decode, both directions | raw bytes | none | none |
| Control | new TCP connection per call (no `Session`), rootlessport proxy | keep-alive | UDS message | UDS message |
| Observed | AudioSeal detect: 29 ms round trip vs 13.6 ms engine (§6) | expected ~2–4 ms | expected <1 ms | expected <1 ms |

The serving side makes this worse. Every `app.py` endpoint is `async def` calling a synchronous engine, with a single uvicorn worker, so **the GPU sits idle while a request is decoded and encoded**. For short engines such as AudioSeal detect, that roughly halves GPU-lane utilisation.

### 4.3 Transport tiers

One `ServiceClient` (host) and one `app_factory` (service) provide every tier. A service advertises its tiers at `GET /capabilities` and the client picks the best tier both sides support. Every tier must pass **gate T** (bit-exact), because each one carries the same float64 bytes.

| Tier | What it is | Status |
|---|---|---|
| **T0** JSON + base64-float64 | Today's wire, kept for compatibility. Contract fixtures stay valid on CPU. | Kept |
| **T1** binary | `application/octet-stream` body (float64 LE). Params in a small JSON header. Keep-alive session. Server-side request queue plus engine worker thread, so decode overlaps compute. | **Core** (Phase 5) |
| **T2** shared memory | Control over HTTP-on-Unix-domain-socket (no ports, no rootlessport). Audio in fixed-size slots of a tmpfs directory mounted into each container. Host and containers `np.memmap` the slot. | Optional, **O1**. Adopt if T1 overhead is still > 10% of engine time on the bottleneck lane. |
| **T3** CUDA IPC | GPU→GPU handoff between service processes (below). | Optional, **O2**. Spike plus measured benefit. |

**CUDA IPC design (T3).** It's built to be independent of torch version, because the GPU services span torch 2.0 (cu118), 2.4 (cu121), 2.7 (cu126) and 2.13 (cu130), and torch's own `_share_cuda_` reductions aren't stable across them.

- **Slots.** Each GPU service process allocates a small **ring of exchange slots** with `cudaMalloc` (ctypes on its own libcudart), outside torch's caching allocator. For example, 32 slots × 4 MB.
- **Handshake.** At startup it exports `cudaIpcGetMemHandle` for each slot. Peers call `cudaIpcOpenMemHandle` **once** and cache the mapping, because opening costs milliseconds while reusing a mapping costs nothing.
- **Handoff.**
  1. The producer D2D-copies its output into a free slot.
  2. It syncs its stream (or records an IPC event).
  3. It sends `{slot, generation, n, dtype}` over the control channel.
  4. The consumer wraps the mapped pointer with `__cuda_array_interface__`, giving a torch tensor with zero copies.
  5. A release message returns the slot.
- **Host copy.** Still made once, asynchronously, for metrics. CPU attacks and metrics need it anyway.
- **Container requirements.** All GPU containers in **one Podman pod sharing the IPC and PID namespaces**, on the same MIG GPU instance. Whether legacy CUDA IPC works within a MIG instance under rootless Podman is exactly what spike **S1** establishes. Cross-instance IPC is not supported by MIG, and we only use MIG 2 anyway.
- **Expected gain.** ~16 avoided crossings per file × ~30 µs ≈ 0.5 ms/file. That's why it's optional. It becomes interesting only if a future pipeline keeps intermediate tensors on the GPU and never needs them on the host.

### 4.4 GPU concurrency (the lever once the GPU lane is the bottleneck)

Separate processes on one MIG instance **time-slice**: kernels from different contexts never overlap. Small kernels (AudioSeal 13 ms, EnCodec, VAE) leave most of the 32 SMs idle. Options, in order of preference:

1. **Batching with deterministic composition** (core, Phase 6). The detect inputs of one file whose producers are host-native attacks are batched together, bucketed by exact length. Composition depends only on the file, never on timing, so results stay reproducible. Gate D runs batched GPU against single-item CPU, and the GPU repeat must be EXACT.
2. **Co-hosting** (optional, **O3**). One engine-host process per runtime group runs several engines on separate CUDA streams. Kernels then overlap, the handoff within the group is in-process with zero copies, and N−1 CUDA contexts (~0.3–0.5 GB each) are saved. The torch-2.7 group is AudioSeal, EnCodec, DAC, Diffusion, BigVGAN, SE1, VAE (+ WavMark, Perth, AWARE). There's a known pin conflict: `neural_vocoder` pins `huggingface_hub==0.24.7`, while the base has 0.33.2. Spike **S2** resolves the union.
3. **MPS on MIG 2** (optional). Concurrency without merging processes. The daemon must be confined to our UUID, and its pipe directory shared into the containers. Spike **S3**.
4. **Replicas** help only when the GPU isn't saturated. They're always right for NetworkTransmission, which is a sleeping CPU service.

### 4.5 Determinism model (prerequisite for anything concurrent)

Today every random draw comes from one process-global stream, seeded once (`run.py:233-238`). A result therefore depends on its position in the stream: on file order (unsorted `os.listdir`), on the attack set, and on everything that ran before it. The new model:

- **Seeds.** `seed(task) = SeedSequence(run_seed, spawn_key=(stable_file_id, stable_task_id, purpose))`, where `stable_file_id` is a hash of the relative path and `stable_task_id` is a hash of the display name (e.g. `Codec2VocoderAttack_700`).
- **Host attacks.** They run in worker processes, and the worker sets `random.seed` / `np.random.seed` from the task seed just before `apply`. **Plugin code is unchanged.** `generate_watermark()` likewise gets a per-file seed.
- **Services.** Requests carry an optional `seed`. VAE, SE1, SE2 and Diffusion seed per request; without the field they behave as today.
- **Hidden state:**
  - **Codec2**: libcodec2's global PRNG (`codec2_vocoder/attack.py:24-31`). Each call runs in a fresh process forked from a clean template, so it always starts from the same PRNG state (D4).
  - **VAE**: RAVE streaming buffers. Snapshot at load, restore before each request (D5).
  - **NetworkTransmission**: kernel netem randomness can't be seeded and stays stochastic. That's already documented.
- **Legacy mode.** `--seeding legacy` (sequential executor only) reproduces v2.0.0 bit-for-bit.

### 4.6 Scheduling: the algorithm

Three layers: two decided before the run (placement, admission window), one at run time (ordering).

**(a) Placement optimiser (static, before the run).**
- Variables: a device for each (plugin, op) in the GPU-eligible set (≤ ~16 binary choices). An op is eligible only if it has a PASS verdict and fits the VRAM budget.
- Objective: minimise the throughput bound
  `T* = max over resources R of load_R / capacity_R`.
- Resources: the GPU lane, the CPU pool (W workers), each single-concurrency service, and NetworkTransmission replicas.
- `load_R` includes compute time from the cost table (measured, `run_profile.json`) **plus the transfer and serialization cost of every edge that crosses devices** under that placement.
- Search: exhaustive over 2^16 is trivial, so the result is optimal for the cost model. VRAM is a knapsack constraint, binding only on small slices.
- Output: the device map, the predicted per-file time, and the predicted bottleneck. `--plan_only` prints these without running anything.

**(b) Admission window.** At most K files are in flight. K is the smallest value that keeps the bottleneck lane saturated in a discrete-event simulation over the cost table, so memory is bounded at about K × 40 MB.

**(c) Run-time ordering: a residency-aware list scheduler.** It's event-driven and single-threaded, with no I/O, so it can be unit-tested against a simulated clock.

- **Priority.** Upward rank (HEFT): the longest remaining cost path to the file's completion, transfers included. Ties go to the earliest file. This starts long poles first (NetworkTransmission, Diffusion, SE2) and keeps file latency bounded.
- **Locality.** When a buffer becomes resident on device X, its pending consumers on X are boosted (consume while hot). The buffer is released when its refcount reaches 0.
- **Copies.**
  - The host copy is made once, eagerly and asynchronously, if any host consumer exists.
  - The device copy is made once per consuming GPU process.
  - Every buffer therefore crosses each device boundary at most once, so the transfer count equals the analytical minimum of §4.1. That's a unit-test assertion, not an aspiration.
- **Static batch groups** (§4.4.1) are dispatched when every member is ready. They're all fast host attacks.
- **Order-sensitive tasks**, if any remain after §4.5, go to an ordered lane in canonical order.
- **Per-service concurrency limits** are enforced, because single-worker services queue anyway. For example, NetworkTransmission adds and removes a netem qdisc on `lo` on every call, so it's 1 per replica.

**Why this and not an exact solver.** For N near-identical files on a few resources, makespan ≥ fill + N × T* + drain. A list scheduler with the right priorities, plus enough files in flight, gets within fill/drain of that bound. An exact solver (CP-SAT) adds a dependency and minutes of solve time for no measurable gain. The run summary reports **efficiency = lower bound / actual**. If it falls below 90%, the cost model or priorities get fixed, which is measurable.

### 4.7 Projected throughput (re-measured in Phase 0 and after each phase)

Per-file costs, from the study's 1g-slice run `gpu2`:
- **GPU lane:** ~6.2 s engine time. Expected ~4–5 s on 2g. Minus SE2 rebuild (0.3 s) and detect batching (~0.5 s) gives ~3.5–4.5 s.
- **CPU:** metrics 16.4 + native ~5 = ~21.5 CPU-s. With W = 8 that's ~2.7 s.
- **NetworkTransmission:** 5.6 s per replica, so 2 replicas give 2.8 s.

Bottleneck: GPU at ~4–5 s/file. **245 files: ~15–25 min.** Today's references: 2.3 h (GPU, sequential, 1g) and 5.0 h (CPU).

---

## 5. Code changes, by module

New host-side package **`src/deepmarkpy/execution/`**:

| Module | Responsibility |
|---|---|
| `graph.py` | `Task`/`TaskKey`/`TaskKind` (EMBED, ATTACK, DETECT, METRICS, WM_QUALITY, PERSIST). `build_file_graph()` encodes today's semantics of `Benchmark.run`: crop-before-attack, CrossModel's extra detect, ZeroBitCollusion's clean input, Codec2 expansion, save_audio. |
| `seeding.py` | Per-task seeds (§4.5) and a `seeded_globals()` context manager. |
| `placement.py` | Device map from CLI/config, the optimiser (§4.6a), and **verdict enforcement** (§3.5). |
| `costs.py` | Cost table: load and save `run_profile.json`, defaults, transfer-cost model. |
| `scheduler.py` | Pure scheduling logic (§4.6b, c), plus a discrete-event simulator used by `--plan_only` and the tests. |
| `lanes.py` | `CpuLane`: process pool with per-task seeding and `threadpoolctl` limits of 1 per worker (the §4.4 Replay lesson). `ServiceLane`: threads, keep-alive, concurrency = replicas, batch dispatch. `IsolatedLane`: fork per call, for Codec2. |
| `buffers.py` | `AudioRef`, with residency (host / slot / device-slot) and refcounts. One store implementation per transport tier. |
| `sequential.py` | Today's loop, moved verbatim, as `--executor sequential`. This is the reference for gate S. |
| `results.py` | Assembles **the same results schema** as today. Calls `on_file_complete` per file (files may finish out of order; reports are sorted by path). |

Changes to existing host code:

| File | Change |
|---|---|
| `benchmark.py` | `run()` builds the graph and delegates to the chosen executor. Signature, validation-before-run and result schema are unchanged (`test_run_completeness.py`). `apply_attack` stays and is called by the workers. `no_attacks` and `detection_reliability` stay sequential at first. |
| `run.py` | New flags: `--executor {sequential,pipelined}`, `--seeding {legacy,per_task}`, `--workers`, `--files_in_flight`, `--device_map`, `--plan_only`, `--profile_out`, `--on_task_error {abort,record}`. Defaults don't change behaviour until Phase 7 (D2). `run_metadata.json` gains executor, seeding, device map, image digests and verdict IDs. |
| `core/wire.py` | `decode_audio` returns a writable array (`.copy()`). `encode_audio_bytes`/`decode_audio_bytes` (raw float64 LE). Encoding names. |
| `core/transport.py` (new) | `ServiceClient`: one HTTP client for all 16 plugin clients, which today each call `requests.post` themselves. Keep-alive, timeouts, tier negotiation, errors that name the plugin. `BaseModel._make_request` delegates to it. |
| `core/inference.py` | Optional `detect_batch`/`apply_batch` with a loop default. Stays import-light (`test_engine_contract.py:69-80`). |
| `core/base_model.py` | `detect_batch()` default loop, overridden by HTTP clients when the service supports batching. |

New service-side package **`src/deepmarkpy/serving/`**, shipped in the package that images install:

| Module | Responsibility |
|---|---|
| `device.py` | `resolve_device()` from `DEEPMARK_DEVICE` (`cpu` default, `cuda`, `auto`). `configure_numerics()` (§3.2). `mig_cudnn_fallback()`, promoted from `gpu_engines.py:21-32`. Placement logging. torch is imported lazily. |
| `app_factory.py` | `make_app(engine, kind)` registers the existing T0 routes **byte-for-byte unchanged**, plus T1 binary routes, `/capabilities` (tiers, batch, seed, device, image digest), and `/health`. A request queue plus engine worker thread lets decode/encode overlap compute. Each `app.py` shrinks to a few lines, migrated one service at a time with CPU contract fixtures as the gate. |
| `ipc.py` (O2 only) | Slot ring, handle export/open, control messages (§4.3). |

Per-plugin engine changes, each its own commit with its own gate:

| Plugin | Change |
|---|---|
| `models/audio_seal/inference.py` | Honour `device`, `inference_mode`, `.cpu()`, cuDNN fallback. `detect_batch`. |
| `attacks/speech_enhancement_1/inference.py` | Honour `device`; `.cpu()` before `.numpy()`. Request seed. |
| `attacks/speech_enhancement_2/inference.py` | Device pick that works under MIG. Prefer passing the device to ClearVoice; otherwise the `get_free_gpu` patch. Request seed. Optional ClearVoice reuse (gate T: EXACT or don't ship). |
| `attacks/speech_tokenization/inference.py` | Move the input to the device. D3 fix if approved. |
| `attacks/descript_audio_codec/inference.py`, `attacks/encodec/inference.py` | Cache `torchaudio.Resample` per (sr_in, sr_out, device); gate T. |
| `attacks/diffusion/inference.py` | Request seed; CPU `torch.Generator`. |
| `attacks/vae/inference.py` | Restore streaming state per request. Request seed. |
| `models/{silent_cipher,timbrewm,wavmark,perth,aware}` | Honour `device`; AWARE gets split placement. |
| `attacks/codec2_vocoder/attack.py` | Nothing. Isolation is done by the lane. |
| `attacks/replay/attack.py` | `threadpoolctl` limit (EXACT). `fftconvolve` only if D-Replay is approved (changes outputs by ~1e-13). |

Build and deploy:

| Item | Change |
|---|---|
| Base images | `Dockerfile.base` stays `+cpu` (upstream). New `Dockerfile.base.cu126` → `ml-services-base:cu126`, with its own `constraints.base.cu126.txt`. |
| Service Dockerfiles | `ARG BASE_IMAGE` (and `ARG TORCH_INDEX` for SE2 cu130, SilentCipher/TimbreWM cu118, XCodec2 cu121). One Dockerfile per service, with variants by build argument instead of copies. |
| `docker-compose.gpu.yml` (repo root) | Override for GPU-placed services only. CDI `devices: [nvidia.com/gpu=${DEEPMARK_GPU_DEVICE}]` (works for Podman and CDI-enabled Docker), `DEEPMARK_DEVICE=cuda`, `NVIDIA_TF32_OVERRIDE=0`, image tags `:cu126` etc. NetworkTransmission replicas. |
| `scripts/gpu_slice_guard.py` | Refuses to start unless `DEEPMARK_GPU_DEVICE` exists, is the expected MIG device, has no foreign processes on its GPU instance, and has enough free memory. Replaces the three stale UUID references. |
| `tools/parity/` | `deepmark-parity`, promoted from `parity_apply.py`, `parity_compare.py`, `compare_runs.py`, `replay_*.py`. Adds thresholds and verdicts. `parity/tolerances.yaml`, `parity/attack_classes.yaml`, `build/verdicts/` (local, not committed). |

Tests:

| File | Purpose |
|---|---|
| `test_seeding.py` | Seeds are stable across processes and independent of order. |
| `test_graph.py` | The graph reproduces today's per-file semantics. |
| `test_scheduler.py` | Simulated clock. Dependencies respected. **Transfers == analytical minimum**. Window bound holds. Batch composition is deterministic. Efficiency ≥ 0.9 on synthetic workloads. |
| `test_executor_equivalence.py` | Sequential vs pipelined, `per_task` seeding, native attacks plus a fake in-process model, with W ∈ {1, 2, 8} and shuffled ready order. Results JSON **identical**. |
| `test_wire_format.py` / `test_transport.py` | Extended: binary round trip exact, writable decode, T0 and T1 routes byte-identical on a FastAPI `TestClient`. |
| `test_cpu_only_builds.py` → `test_build_variants.py` | CPU files and images must resolve `+cpu`. GPU pins allowed only in `*.cu*.txt` and `*cu*` Dockerfiles. Also checks the resolved build, not just the suffix (study §9.5). |
| `test_service_client_contract.py`, `test_wire_format.py` substring checks | Updated to assert use of `ServiceClient` and `app_factory` instead of the literal patterns. |
| Docs | `CLAUDE.md` / `README.md` / `docs/CONSUMING.md`: GPU usage, gates, flags, the CPU-default policy. |

Unchanged on purpose: `test_discovery_lock.py` (no new plugin classes), native goldens (native attacks don't move), CPU contract fixtures (T0 routes stay byte-identical).

---

## 6. Implementation order

Each step lists the gate that must pass before the next step starts. All GPU work is on **MIG 2 only**, via `DEEPMARK_GPU_DEVICE`. Effort is rough engineer-days.

**Phase 0: Ground truth (3–4 d)**
1. Branch from `origin/main` `fee91af` and carry the study scripts over. Gate: `pytest` passes on the client env, and the known rubberband failures (3) are documented.
2. `DEEPMARK_GPU_DEVICE=MIG-50bf…` as the single source of truth, plus `gpu_slice_guard.py`. Fix the three stale UUID references.
3. Rebuild CPU images on main, and build `ml-services-base:cu126` plus GPU variants. Check placement inside each container (the `serve_swap.py` placement log).
4. **Re-baseline on 2g.35gb:** CPU 20-file run; per-service engine times on GPU (TF32 off and on); slice memory with all services resident.
5. `tools/parity` with tolerances and verdicts. Capture CPU references for every candidate on the corpus (§3.3).
6. Spikes, each producing a short go/no-go note:
   - **S1** CUDA IPC between two rootless Podman containers in one pod on MIG 2, including across torch versions (2.7 ↔ 2.0);
   - **S2** pip resolution of the torch-2.7 group union;
   - **S3** MPS confined to MIG 2.

**Phase 1: Correctness prerequisites (2 d)**
1. Writable `decode_audio`. Gate: EXACT, goldens and contracts unchanged.
2. Every task failure names the attack (`--on_task_error`). Gate: tests.
3. Replay `threadpoolctl` limit. Gate: EXACT.
4. D3 (XCodec2) and D-Replay (`fftconvolve`), if approved. Each gets its own golden or reference re-capture, called out in the commit.

**Phase 2: Device layer and GPU migration (8–10 d)**
1. `serving/device.py` and the `DEEPMARK_DEVICE` plumbing. Gate: on CPU, every contract fixture is **SHA-identical** (CPU behaviour untouched).
2. **Wave 1**, one service per step, in this order: AudioSeal detect → AudioSeal embed → DAC → BigVGAN → SE2 → XCodec2 → EnCodec → SE1.
   - Each step: gate D at its tier, plus GPU repeat EXACT. AudioSeal also includes the short-input sweep (cuDNN MIG fallback) and the 1,000-input detect replay from the study (0 flips).
3. Gate E after wave 1: 20-file run with the sequential executor, CPU vs GPU. Deterministic attacks have identical per-file accuracy, as in study `gpu2`.
4. **Wave 2:** Diffusion (with the D6 seed), VAE (D5). Gate D at NUMERIC or STOCHASTIC, then gate E.
5. **Wave 3:** SilentCipher, TimbreWM (cu118 images); WavMark, Perth and NISQA if the measurements justify it; AWARE detect (optional). Gate D, and gate E for each model as the model under test.
6. `test_build_variants.py`, plus a docs update of the CPU-default policy.

**Phase 3: Determinism (2–3 d)**
1. `execution/seeding.py`; `--seeding per_task` in the sequential executor. Gate: same seed twice gives an EXACT results JSON, and `--seeding legacy` stays EXACT against pre-change output.
2. Service `seed` field (VAE, SE1, SE2, Diffusion). Gate: absent field gives an unchanged response (CPU contract SHA), and present seed twice gives EXACT.
3. Codec2 isolation (D4). Gate: Codec2 results independent of preceding calls (permutation test).

**Phase 4: Pipelined executor (5–7 d)**
1. `graph.py`, `scheduler.py` (with simulator) and `results.py`, unit-tested with fakes.
2. `lanes.py`, the pipelined executor and NetworkTransmission replicas.
3. **Gate S:** pipelined vs sequential, both `per_task`, on 20 files on CPU and then on GPU. Results JSON EXACT for every deterministic attack. Stochastic attacks (NT, and VAE if STOCHASTIC) excluded and listed.
4. Measure: wall time, per-lane utilisation, scheduler efficiency against the lower bound. Fix the cost table or priorities if efficiency is below 0.9.

**Phase 5: Transport T1 (3 d)**
1. `ServiceClient` (keep-alive) and migrating the 16 clients. Gate T: EXACT.
2. `app_factory` with T1 routes, request queue and worker thread, one service at a time. Gate T: EXACT, and CPU T0 contract SHA unchanged.
3. Measure per-call overhead per service. Decide on O1 (shared memory).

**Phase 6: GPU-lane efficiency (3–4 d)**
1. Detect batching for the model under test with deterministic groups. Gate D (batched GPU vs single CPU), plus GPU repeat EXACT.
2. Resampler caching (DAC, EnCodec) and ClearVoice reuse (SE2). Gate T: EXACT, or don't ship.
3. Re-measure, and re-run the placement optimiser with the new costs.

**Phase 7: Validation and rollout (2–3 d)**
1. Full 245-file runs: CPU-legacy vs GPU-pipelined. Gate E with a written comparison like study §6.
2. Decide on default flags and the version bump (D2).
3. Docs, and a `make gpu-gate` / `make gpu-run` entry point.

**Optional tracks** (start only on a go from the spike and the measurements):
- **O1** shared-memory transport;
- **O2** CUDA IPC;
- **O3** co-hosting or MPS;
- **O4** AWARE split placement.

Each passes gate T (EXACT) or gate D like everything else.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| The MIG layout changes again, or someone else's job lands on MIG 2 (it happened during §11) | Slice guard at startup. The UUID lives in one variable. Speed measurements are flagged as invalid if foreign processes appear on our GPU instance during a run (NVML sampling). |
| BigVGAN's median rel. L2 was 9.4e-6, just under the 1e-5 bound; the 2g slice may pick other cuDNN algorithms and cross it | If it fails, find the layer and algorithm responsible before touching the threshold. A tolerance only changes by an explicit, reviewed edit to `tolerances.yaml`. |
| The cuDNN/NVML assert behaves differently on 2g.35gb | Re-run the short-input sweep in Phase 0. The fallback is generic and logged, and the gate corpus includes the boundary lengths. |
| Batching or replicas break reproducibility | Deterministic batch composition only; gate S/E plus GPU repeat EXACT. |
| A rebuilt image silently changes numerics (pip drift) | Verdicts are keyed by image digest; the placement layer refuses a mismatch. |
| Pipelining hides a failure | `--on_task_error abort` stays the default. In `record` mode the run exits non-zero with a named list of failures, in keeping with "stop runs from quietly shrinking". |
| Host CPU contention with other users | `--workers` default 8, per-worker BLAS/OMP threads 1. |
| Upstream pin conflicts block co-hosting | It's optional (O3). The core plan keeps one container per service. |

---

## 8. Decisions needed

| # | Decision | Recommendation |
|---|---|---|
| D1 | Tolerance definition (§3.2) | Tiers as proposed: 1e-4 absolute, 1e-5 / 3e-5 relative L2, zero decision flips; EXACT for refactors |
| D2 | Per-task seeding changes per-file random draws compared with v2.0.0 | Adopt it for new runs, and bump the minor version with a "not per-file comparable" note like v1/v2. Keep `--seeding legacy` (sequential) to reproduce published numbers. |
| D3 | XCodec2's random encoder (study §4.1) | Fix the key remap **before** the GPU baseline, so we don't validate a broken model. It moves published numbers. |
| D4 | Codec2 PRNG: reset per call (fork isolation) or keep the history-dependent legacy behaviour | Isolation: results become a function of (input, params, seed) |
| D5 | VAE streaming state: reset per request | Reset; otherwise any reordering changes VAE outputs |
| D6 | Diffusion: request seed instead of OS entropy | Yes; it also makes the GPU vs CPU comparison possible element by element |
| D7 | "CPU only" policy | "CPU by default; GPU via verified variants", enforced by `test_build_variants.py` and the verdict registry |
| D8 | CUDA IPC / MPS / co-hosting | Decide after spikes S1–S3 and Phase 5/6 measurements, against the criteria in §4.3–4.4 |
| D-Replay | `fftconvolve` in Replay (output changes ~1e-13; goldens move) | Yes, as a deliberate golden update; `threadpoolctl` alone is the EXACT fallback |
