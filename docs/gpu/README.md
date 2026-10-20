# Running services on a GPU

Some services can run on a GPU, and only where it is proven to give the same numbers as the CPU. The benchmark program itself stays on the CPU. What moves are the model and attack **services** (the Docker containers).

| Document | What it covers |
|---|---|
| [plan.md](plan.md) | The plan: what goes on the GPU, how each move is verified, the architecture, the phases. Its opening section records the decisions made so far. |
| [wave1-findings.md](wave1-findings.md) | Parity results per service, and what each failure turned out to be. |
| [e2e-wave1.md](e2e-wave1.md) | The first benchmark run on a GPU deployment, compared with a CPU deployment. |
| [spike-cuda-ipc.md](spike-cuda-ipc.md), [spike-cohosting.md](spike-cohosting.md) | Feasibility checks for later phases. |

"The GPU study" in code comments is the measurement report that preceded this work (2026-10-01). The numbers these documents and the code rely on are restated in them.

## Current status

On the GPU, each by a measured PASS:
- AudioSeal (embed and detect);
- DAC;
- BigVGAN, in float64;
- SE1 and SE2.

Everything else runs on the CPU:

| Services | Why on CPU |
|---|---|
| EnCodec, XCodec2 | Measured; they do not meet the tolerance as written |
| Diffusion, VAE | Not yet verified |
| Everything else | No GPU path |

CPU behaviour is unchanged. Each touched engine was checked bit for bit against the shipped CPU output.

These results were measured on an H200 NVL MIG slice (`2g.35gb`); `wave1-findings.md` has the numbers. Verdicts are local output, like `build/images.json`: written to `build/verdicts/` and not committed. Each one counts only for the image IDs and GPU model it was measured on, so every machine runs the gate once on its own GPU. Until it has, every service there stays on the CPU.

## Using it

```sh
# 1. Build the images, CPU and GPU variants. Records image IDs in build/images.json.
python scripts/build_images.py --variant cpu
python scripts/build_images.py --variant gpu

# 2. On a new GPU model, run the parity gate (see "The gate" below).

# 3. Start the services a config needs. --gpu takes an index for a whole GPU, or a GPU-/MIG- UUID.
deepmark-services plan --config my.json --gpu 0     # what would run where, and why
deepmark-services up   --config my.json --gpu 0     # start them; waits until all answer

# 4. Run the benchmark as usual.
deepmark-benchmark --config my.json
```

Without `--gpu`, everything starts on the CPU, exactly as plain `docker compose up` would.

`deepmark-services up` writes `build/compose.deploy.yml` and `build/deployment.json`. The benchmark checks that the running containers are those images before it measures anything, and records the placement in `run_metadata.json`. Containers started by hand still work; nothing is recorded or checked for them.

With Podman, set `CONTAINER_ENGINE=podman`. Docker is the default; its GPU syntax is generated, but has so far been exercised only by unit tests.

## The gate

A service goes on the GPU only with a PASS from `deepmark-parity`. The PASS is for that image, that GPU model, the engine the image ships, strict numerics, and its precision. The gate:
1. runs the engine inside its image, on the CPU and on the GPU, over a fixed corpus (31 cases: speech plus edge cases);
2. compares the outputs against tolerance tiers, defined in `src/deepmarkpy/parity/tiers.py` and set per service in `parity/gates.json`.

Example for one service:

```sh
deepmark-parity corpus --sources <dir of speech clips>              # once
deepmark-parity run audioseal embed --variant cpu --label cpu
deepmark-parity run audioseal embed --variant gpu --label gpu --gpu 0
deepmark-parity wm audioseal --label cpu                             # attacks' input set
deepmark-parity run descript_audio_codec apply --variant cpu --label cpu
deepmark-parity run descript_audio_codec apply --variant gpu --label gpu --gpu 0
deepmark-parity probe descript_audio_codec --cand gpu --gpu 0        # codecs: tie-break evidence
deepmark-parity judge descript_audio_codec apply --ref cpu --cand gpu  # writes build/verdicts/...
```

The tiers:

| Tier | Requirement |
|---|---|
| **EXACT** | No difference at all. |
| **NUMERIC** | max \|d\| ≤ 1e-4. Relative error ≤ 1e-5 median and 3e-5 worst, or an RMS error under 1e-6 (−120 dBFS). No decoded watermark bit may flip. PESQ/ViSQOL/STOI barely move. The GPU must repeat itself bit for bit. |
| **QUANTIZED** (codecs) | NUMERIC, except at most 1% of frames. Every such frame must be shown to be a codebook near-tie. |

Missing evidence fails; a check that did not run never counts as passed.
