# DeepMark Benchmark

DeepMark Benchmark is a modular and scalable Python platform for evaluating the robustness of audio watermarking systems. It enables testing against various attacks, including both simple signal manipulations and advanced AI-based disruptions, using a containerized architecture for consistency and ease of use.

## Two ways to use this repository

1. **Primary — run the benchmark.** Evaluate watermarking models against
   40+ attacks through the CLI (`deepmark-benchmark`) with the containerized
   model/attack services managed by docker-compose. This is the workflow the
   rest of this README describes.
2. **Additional — consume the plugin engines.** Install `deepmarkpy` as a
   library and import any plugin's inference engine — for example
   `from deepmarkpy.plugins.attacks.vae.inference import VAEEngine` — to
   embed the watermarking models and attacks in your own serving stack,
   without this repo's HTTP layer or orchestrator. Engines derive from
   `deepmarkpy.core.inference.BaseAttackEngine` / `BaseModelEngine`, so
   generic serving code can be typed once per family. See
   [docs/CONSUMING.md](docs/CONSUMING.md).

## Features

*   **Extensible Plugin System:** Easily add new watermarking models and attacks.
*   **Containerized Services:** Key models and attacks run as isolated Docker services for dependency management and reproducibility.
*   **Centralized Configuration:** Service network ports are managed via a single `.env` file.
*   **Client-Server Architecture:** The benchmark runner communicates with containerized plugins via HTTP.
*   **Standardized Execution:** Provides a CLI for running benchmarks and collecting results.

## Architecture Overview

This benchmark uses a client-server architecture. Core watermarking models and complex attacks (often AI-based) run as independent web services managed by Docker Compose. The benchmark runner (`deepmark-benchmark`, i.e. `src/deepmarkpy/run.py`) acts as a client, communicating with these services via HTTP requests to perform embedding, attacking, and detection. This isolates complex dependencies within containers. Each containerized plugin keeps all of its inference logic in one `inference.py` engine class behind a thin FastAPI adapter, which is also what makes those engines importable as a library.

## Prerequisites

*   Python 3.10+
*   Docker (Install Docker)
*   Docker Compose (Install Docker Compose)

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/deepmark/deepmarkpy-benchmark.git
cd deepmarkpy-benchmark
```

### 2. Create the Environment File (`.env`)

Service ports live in a `.env` file, which Docker Compose reads automatically
and which the CLI loads so host-side clients use the same ports. It is not
tracked, so create it from the template:

```bash
cp .env.example .env
```

*   **Action:** Review the ports. The defaults are fine unless one collides with something already running, in which case change it in `.env` before starting the services.
*   `HOST` is the address uvicorn binds *inside* each container and must stay `0.0.0.0`. The services are kept off the network by `docker-compose.yml`, which publishes every port on `127.0.0.1` only.

### 3. Install Core Dependencies (Optional - for development/direct script interaction)

> **Note:** the host environment no longer includes PyTorch. The `encodec`
> and `descript_audio_codec` attacks now run as Docker services (like the
> other ML attacks); `torch`, `torchaudio`, `encodec`, and
> `descript-audio-codec` were removed from `requirements.txt`. Existing
> environments keep working; fresh installs are considerably lighter.

Install the benchmark as a package (this provides the `deepmark-benchmark`
command; `python src/run.py` keeps working as a deprecated alias):

```bash
pip install -e .[all]
```

It's recommended to use a virtual environment for the benchmark runner itself:

Linux/macOS: 
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows
```bash
python -m venv venv
venv\Scripts\activate
```

Install core benchmark runner dependencies:

```bash
pip install -r requirements.txt
```

### 4. Install Docker (For AI-Based Attacks and Models)
If you plan to use AI-powered attacks or models, install [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/).

### 5. Install Rubberband (Windows Only)

If using time stretch and pitch shift attacks on Windows, you'll need Rubberband CLI:

1. Download Rubberband CLI:
   - Get Windows executable from [Rubber Band website](https://breakfastquay.com/rubberband/)

2. Extract Files:
   - Unzip to a directory (e.g. C:\Program Files\rubberband)

3. Add to PATH:
   - Open System Properties > Advanced > Environment Variables
   - Under System Variables, find "Path"
   - Click Edit > New
   - Add your rubberband directory path
   - Click OK to save

### 6. Download Additional Datasets (For Specific Attacks)

Some attacks require additional datasets to function:

| Attack | Dataset Required | Description |
|--------|------------------|-------------|
| `ReplayAttack` | AIR (Acoustic Impulse Response) files | Room impulse responses for simulating acoustic replay |
| `MixingAttack` | Music dataset | Background music for mixing with watermarked audio |

**Download the datasets:**
1. Download from [Google Drive](https://drive.google.com/drive/folders/17ZSP9gxumXs8V2K0ARBK5JQVtjxJbmyZ?usp=sharing)
2. Extract and place in the project root:
   ```
   deepmarkpy-benchmark/
   ├── AIR_wav_files/        # AIR files for ReplayAttack
   ├── music/                # Music files for MixingAttack
   └── ...
   ```

*Note: These attacks will fail if the required datasets are not present.*

## Running the Benchmark

### 1. Build and Start Services

These commands build the Docker images for all containerized models/attacks (defined in `docker-compose.yml`) using the configuration from `.env`, then start them in the background. This step is **required** if you intend to use plugins like `audioseal`, `vae`, `diffusion`, etc.
```bash
docker build -f Dockerfile.base -t ml-services-base:latest .
docker-compose -f docker-compose.yml build
docker-compose up -d
```
You can check the status of the services using `docker-compose ps`. The first build might take some time.

> **Tip:** You don't need to run all services at once. If you only need specific attacks or models, you can build and run them individually:
> ```bash
> docker-compose up -d audioseal diffusion  # Only start AudioSeal model and Diffusion attack
> ```

### 2. Run the CLI

Measurement settings live in a JSON config file, one per mode, and the other
flags are operational (the 2.x flags still run without `--config`; see
[Running with the 2.x flags](#running-with-the-2x-flags)). Start from a
commented template, choose its attacks, check it, then run it:

```bash
deepmark-benchmark --init benchmark > my_config.json
# Edit my_config.json to select attacks, e.g. "attacks": {"groups": ["audio_distortion"]}
deepmark-benchmark --config my_config.json --validate-only
deepmark-benchmark --config my_config.json --wav_files_dir /path/to/audio
```

The template leaves `attacks` empty, which runs every discovered attack: that
needs every service running (`docker-compose up -d`) and the AIR and music
datasets. The `audio_distortion` attacks run natively, so the run above needs
only the model's service (`docker-compose up -d audioseal` for the template's
`AudioSealModel`).

`--validate-only` checks the file against the discovered plugins and exits
without running anything. It needs neither the audio nor the Docker services:
an unreachable model service is reported but is not an error, and attack
services are not checked.

| Mode | What it measures |
|------|------------------|
| `benchmark` | Watermark survival under each attack, plus the audio-quality cost |
| `no_attacks` | Baseline: does the model read back its own watermark, and what does embedding alone cost? |
| `detection_reliability` | False positives on clean audio and false negatives on watermarked audio |

Each file declares its `"mode"` and accepts only that mode's keys: a
`no_attacks` file that mentions attacks is an error. Pass several files
(`--config my_config.json my_reliability.json`) to run several modes in one
invocation; each must declare a different mode, and they run in the order
given.

**Models and attacks.** In `benchmark` mode, two or more `models` add a
comparative report. `attacks.groups` and `attacks.list` combine; leaving both
empty runs every discovered attack in `benchmark` mode and only the no-attack
baseline in `detection_reliability` mode. A group expands to the attacks it
declares, so a plugin that failed to load stops the run rather than shrinking
it.

| Group | Attacks |
|-----|---------|
| `process_disruption` | `CrossModelAttack`, `CollusionAttack`, `ZeroBitCollusionAttack`, `Collusion2Attack`, `SameModelAttack` |
| `audio_editing` | `CutSamplesAttack`, `CropBeginningAttack`, `CropRandomAttack`, `WaveletAttack`, `LowpassFilterAttack`, `HighpassFilterAttack`, `BandstopFilterAttack`, `SmoothingAttack`, `ChorusAttack`, `FlangerAttack`, `EchoAttack`, `EqualizerAttack`, `QuantizationAttack`, `STFTQuantizationAttack`, `PCMQuantizationAttack`, `Mp3CompressionAttack`, `EncodecAttack`, `DescriptAudioCodecAttack`, `OpusCodecAttack`, `Codec2VocoderAttack`, `ResamplingPolyAttack`, `MixingAttack` |
| `audio_distortion` | `GaussianNoiseAttack`, `PinkNoiseAttack`, `SignInversionAttack`, `LPCAttack`, `AdditiveNoiseAttack` |
| `desynchronization` | `TimeStretchAttack`, `PitchShiftAttack`, `InvertedTimeStretchAttack`, `ZeroCrossInsertsAttack`, `FlipSamplesAttack` |
| `ai_attacks` | `SpeechEnhancement1Attack`, `SpeechEnhancement2Attack`, `SpeechTokenizationAttack`, `NeuralVocoderAttack`, `DiffusionAttack`, `VAEAttack` |
| `transmission` | `ReplayAttack`, `NetworkTransmissionAttack` |

`deepmark-benchmark --help` lists the flags: `--config`, `--init`,
`--validate-only`, `--wav_files_dir` (required unless the config sets it),
`--report_dir` (default `./report`; **its contents are deleted at the start
of every run**), `--seed` (off by default; it seeds host-side RNGs only, so
the diffusion, VAE, speech_enhancement_2 and network_transmission services
stay stochastic), `--save_audio`, `--verbose`, `--plugins_dir` (also
`DEEPMARK_PLUGINS_DIR`) and `--version`, plus the 2.x flags below. Of the
operational flags, all but `--config`, `--init`, `--validate-only` and
`--version` also exist under `general` in the config file, and the flag wins.

Exit codes: `0` the run finished. A model skipped after an infrastructure
failure, or a report that failed to render, is logged as an error but still
exits 0. `1` a mode produced no results (for example an audio directory that
does not exist or holds no audio files, or every model failed). `2` invalid
configuration or unreachable model service; nothing was run.

Every attack belongs to exactly one group; the canonical mapping lives in
`src/deepmarkpy/utils/attack_groups.py` — update it there when adding a new
attack so the reports pick the right metrics for it, and a test will fail if
one is left ungrouped.

Metrics that compare the two signals sample by sample (PSNR, SI-SDR, STOI,
MCD, NCM) report an attack's timing shift as if it were quality loss, and the
benchmark does not resynchronize — a desynchronization attack is meant to move
the time axis. By default PESQ, ViSQOL and STOI are measured for every attack,
and the `desynchronization` group adds MCD and the reference-free NISQA
dimensions; PSNR, SI-SDR, SII and NCM are off there. A sample-aligned value
under a desynchronization attack is still reported, but marked with a dagger
and a footnote saying why: read it as evidence the attack shifted the signal,
not as a quality score. `SignInversionAttack`'s SI-SDR is marked the same way,
because SI-SDR is scale-invariant and cannot see a polarity flip. Disable
`stoi` and `mcd` for `desynchronization`, or `si_sdr` for `audio_distortion`,
to drop the marked values instead (this needs
`calculate_quality_metrics: true`; when it is false, STOI stays on for every
group).

**Codec2 Vocoder Attack:**

`Codec2VocoderAttack` simulates transmission through a low-bitrate voice channel
(similar to MELP/MELPe military vocoders). It encodes audio at a given bitrate
using the Codec2 codec and decodes it back to PCM. The `bitrate_codec2` parameter
accepts either a single value or a list of bitrates, either in the plugin's
`config.json` or in your config file's `attack_parameters`:

```json
"attack_parameters": {
  "Codec2VocoderAttack": { "bitrate_codec2": [700, 1200, 2400] }
}
```

When a list is provided, the benchmark automatically expands it into separate
runs — one per bitrate — and reports results as `Codec2VocoderAttack_700`,
`Codec2VocoderAttack_1200`, etc. Once a Codec2 version is defined in
`attack_parameters` (`"Codec2VocoderAttack:hi": { "bitrate_codec2": [3200] }`),
each row also names its version: `Codec2VocoderAttack_700 (default)`, ...,
`Codec2VocoderAttack_3200 (hi)`. Supported bitrates: 700, 1200, 1300, 1400,
1600, 2400, 3200 bps; any other value in `attack_parameters` is a validation
error (`E045`).

> **Attack naming convention:** Attack class names must not contain underscores.
> The benchmark uses underscores as a separator between the base attack name and
> a parameter suffix (e.g. `Codec2VocoderAttack_700`). If an attack name contains
> an underscore, the group lookup will incorrectly treat the part after the last
> underscore as a suffix.

### Configuration Files

`deepmark-benchmark --init <mode> > my_config.json` writes a fully commented
template; the same files are in `src/deepmarkpy/config_templates/`. JSON has
no comments, so the templates document themselves with `_`-prefixed keys,
which the parser ignores. Besides the settings below, they cover
`crop_before_attack` (the percentage cropped from the start of the
watermarked audio before each attack), `efficiency` (embed, detect and attack
latency, and container memory; off by default), `duration_groups` (a report
part per audio-length bin) and `comparison` (the accuracy statistic the
comparative report ranks; it must be among accuracy's statistics and cannot
be `std`).

**`calculate_quality_metrics`** is the master switch for the signal metrics.
When `true`, the `metrics` block decides what runs. When `false` or absent,
the signal metrics' enable flags are ignored and PESQ, ViSQOL and STOI are
computed for every group; accuracy always is, `ber`/`emr` keep their flags,
and per-metric `statistics` still apply.

#### Per-group metrics and statistics

`metrics.defaults` applies to every attack group, and
`metrics.per_group.<group>` overrides it for one. Inheritance is per metric: a
group section listing only `pesq` inherits every other metric from `defaults`.
The group keys are the six attack families; the four report subsections of
`audio_editing` (`frequency_filtering`, `temporal_editing`, `audio_effects`,
`compression_quantization`), which inherit from it and are not selectable in
`attacks.groups`; and `other`, for attacks in no family, including third-party
plugins. A metric's statistics come from the first that exists of: its group
entry, the parent group's entry, `metrics.defaults`, the top-level
`statistics` list, then all eight (`mean`, `std`, `median`, `p5`, `p10`,
`p95`, `p99`, `worst_case`). Report columns follow that list's order.
Accuracy's list needs a statistic other than `std` (`E046`).

#### Attack parameters and versions

`attack_parameters` overrides plugin parameters, keyed by attack class name
for its default version, or `AttackName:version` for a named one. Naming a
version the plugin does not have *defines* it, provided you give **all** of
that attack's parameters; give only some and the entry is skipped with `W010`.
Attack and parameter names, and each value's type, are checked against the
discovered plugins before the run starts. An attack named without a version
runs every version it has, so

```json
"attack_parameters": {
  "GaussianNoiseAttack": { "snr_db_gaussian_noise": 25 },
  "GaussianNoiseAttack:aggressive": { "snr_db_gaussian_noise": 10 }
},
"attacks": { "list": ["GaussianNoiseAttack"] }
```

runs `GaussianNoise (default)` at 25 dB and `GaussianNoise (aggressive)` at
10 dB. `--validate-only` prints each version's effective values, and
`run_metadata.json` records them under `attack_parameters_resolved`. A plugin
can declare versions itself by nesting each parameter set under a name in its
`config.json`, one of them `default`.

#### Validation

Problems are reported together, before anything runs, each with a stable
code, its JSON path, the offending value and, when the value looks like a
typo, a suggestion:

```
[E027] my_config.json: metrics.per_group.desync
    unknown attack group. Configurable groups: process_disruption, ...
    got: "desync"
    did you mean 'desynchronization'?
```

Warnings (`W` codes) are printed and the run proceeds.

### Running with the 2.x flags

Without `--config` the 2.x flags still run: they are assembled into a config
and validated like a file. With `--config` the file decides and each ignored
flag is named in a warning; a `--<plugin parameter>` flag next to `--config`
is an error.

| Flag | Config key |
|---|---|
| `--wm_model`, `--wm_models` | `models` |
| `--attack_types` | `attacks.list` |
| `--attack_groups` | `attacks.groups` |
| `--no_attacks` | `"mode": "no_attacks"` |
| `--detection_reliability` | `"mode": "detection_reliability"` |
| `--calculate_quality_metrics` | `calculate_quality_metrics` |
| `--crop_before_attack` | `crop_before_attack` |
| `--<plugin parameter>` | `attack_parameters.<Attack>.<parameter>` |

`--no_attacks` with `--detection_reliability` runs both modes, which needs a
single model. Per-group metrics and statistics, `efficiency`,
`duration_groups`, `comparison` and per-version attack parameters have no
flag; a `--<plugin parameter>` reaches the default version only and must be
spelled in full.

### 3. Quality Metrics (Optional)

The config file sets which of these run for each attack group; see
[Per-group metrics and statistics](#per-group-metrics-and-statistics).

**Audio Quality Metrics:**

| Metric | Description | Range |
|--------|-------------|-------|
| PESQ | Perceptual Evaluation of Speech Quality | 1.0 - 4.5 |
| PSNR | Peak Signal-to-Noise Ratio | dB (higher = better) |
| SI-SDR | Scale-Invariant Signal-to-Distortion Ratio | dB (higher = better) |
| MCD | Mel Cepstral Distortion | dB (lower = better) |
| ViSQOL* | Virtual Speech Quality Objective Listener | 1.0 - 5.0 (MOS) |

*ViSQOL is **optional**. It comes from the prebuilt `visqol-python` wrapper, which `requirements.txt` pins and the `metrics` extra installs (`pip install 'deepmarkpy[metrics]'`, included in `.[all]`). Without it, `W007` warns before the run, the reports name ViSQOL in a footnote instead of a column, and all other metrics are still computed.

**Non-Intrusive Quality (NISQA):**

| Metric | Description | Range |
|--------|-------------|-------|
| NISQA MOS | Overall speech quality (Mean Opinion Score) | 1.0 - 5.0 |
| NISQA NOI | Noisiness | 1.0 - 5.0 |
| NISQA DIS | Discontinuity | 1.0 - 5.0 |
| NISQA COL | Coloration | 1.0 - 5.0 |
| NISQA LOUD | Loudness | 1.0 - 5.0 |

NISQA is a **non-intrusive** metric (it does not require a clean reference signal), which makes it particularly useful for desynchronization attacks where intrusive metrics like PESQ break down. It runs as the `nisqa` Docker service, whose image downloads the model weights when it is built:

```bash
docker-compose up -d nisqa
```

If the service is not reachable, `W006` warns before the run, the reports name the NISQA metrics in a footnote instead of tabling them, and all other metrics still run.

**Speech Intelligibility Measures:**

| Metric | Description | Range |
|--------|-------------|-------|
| STOI | Short-Time Objective Intelligibility | 0 - 1 (higher = better) |
| SII | Speech Intelligibility Index (ANSI S3.5-1997) | 0 - 1 (higher = better) |
| NCM | Normalized Covariance Metric | 0 - 1 (higher = better) |

> **What is measured:** each per-attack metric compares the original
> clean audio against the **watermarked-then-attacked** signal — i.e.
> the combined effect of embedding and the attack. The "No Attack
> (watermark only)" row in the detailed report isolates the embedding
> cost so you can tell the two contributions apart.

### 4. Detection Reliability (Optional)

A `detection_reliability` config measures false positive and false negative
rates. It takes **exactly one model**, which must implement
`is_watermarked()` (see [Adding a New Watermarking
Model](#adding-a-new-watermarking-model)).

```bash
deepmark-benchmark --init detection_reliability > my_reliability.json
deepmark-benchmark --config my_reliability.json --wav_files_dir /path/to/audio
```

Without attacks the mode measures:
- **False positive**: detection on clean (unwatermarked) audio reports a watermark present.
- **False negative**: detection on watermarked audio fails to find the watermark.

Add attacks in the file and FP/FN are additionally reported per attack —
the attack applied to clean audio for FP, to watermarked audio for FN:

```json
"attacks": { "list": ["GaussianNoiseAttack", "LowpassFilterAttack"] }
```

Note that every attack runs **twice per file** here, so a full attack set
costs about double what the same set costs in benchmark mode.

Results are saved to `report/detection_reliability.json` and a dedicated `detection_reliability_report.pdf` is generated.

> **Note:** Only models that implement `is_watermarked()` support this mode.
> Each model defines its own detection logic — zero-bit models check the
> binary output directly, while confidence-based models compare against a
> threshold. A model without it is refused by validation (`E044`) before
> anything runs.

### 5. Save Audio (Optional)

Use `--save_audio` (or `"save_audio": true` under `general`) to write
intermediate audio files to disk for manual inspection. Files are saved
to `<report_dir>/audio/`, or to `<report_dir>/audio/<mode>/` when one
invocation runs several modes; with several models, each model's audio
goes under `<report_dir>/<ModelName>/`.

```bash
deepmark-benchmark --config my_reliability.json --wav_files_dir /path/to/audio \
                  --save_audio
```

In detection reliability mode the following files are saved per input file:
- `{filename}_watermarked.wav` — watermarked audio (before any attack)
- `{filename}_{AttackName}_clean.wav` — attack applied to the clean (unwatermarked) audio
- `{filename}_{AttackName}.wav` — attack applied to the watermarked audio

In the standard benchmark mode, watermarked and attacked-watermarked files are saved.

### 6. View Results

The benchmark generates the following outputs in the `report/` directory.
Each mode writes distinct filenames, so running several in one
invocation does not make them overwrite each other.

**`benchmark` mode, one model:**
- `benchmark_results.json` – Detailed per-file, per-attack results
- `benchmark_stats.json` – Per-attack statistics, carrying exactly the metrics and statistics the config asked for; with `duration_groups` set, it is keyed by duration bin instead (`{"<bin>": {"stats": {...}, "n_files": n}}`), plus an all-files `Overall` entry when `include_overall` is true and the files fall in more than one bin
- `run_metadata.json` – Version, git revision, seed, plugin inventory, and the config file the run came from
- `benchmark_report.tex/.pdf` – Accuracy per attack family, the accuracy ranking chart, and attack-strength curves for any attack run at two or more versions
- `detailed_report.tex/.pdf` – Per-family metric breakdowns against a "no attack (watermark only)" baseline (whenever any signal metric is enabled)

**`benchmark` mode, several models:**
- `report/<ModelName>/` – Individual model reports (same as above)
- `report/comparison/` – Comparative report with:
  - Accuracy comparison table with rank-based coloring, showing `comparison.primary_statistic`
  - One further table per other statistic configured for accuracy
  - Radar chart comparing all models

**`no_attacks` mode:**
- `no_attacks_<ModelName>.json` – Per-file baseline results
- `no_attacks_report.tex/.pdf` – Detection fidelity and the audio-quality cost of embedding

**`detection_reliability` mode:**
- `detection_reliability.json` – FP/FN counts and per-attack statistics
- `detection_reliability_report.tex/.pdf` – Baseline and per-family reliability

Every table's columns come from the config file. If a metric is enabled
but produced no value anywhere — an unreachable service, a missing
optional package — it is named in a footnote under the section rather
than rendered as a table of `N/A`s or dropped without a word. With
`efficiency` enabled, every report but the comparative one adds
processing-time tables, and `container_footprint` adds a Container Memory
section.

## Adding a New Plugin

DeepMark Benchmark is designed to allow easy addition of new attacks and watermarking models.

### Adding a New Attack

1.	Create a New Attack Folder

Inside `src/deepmarkpy/plugins/attacks`, create a new folder with the attack name:
```Shell
mkdir src/deepmarkpy/plugins/attacks/new_attack
```
2.	Add attack.py
Create a file attack.py inside your folder:
```python 
import numpy as np
from deepmarkpy.core.base_attack import BaseAttack

class NewAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """Applies the attack and returns the modified audio."""
        # Example: Invert the audio signal
        return -audio
```
3.	Add config.json
```json
{
    "attack_parameter": 0.5
}
```

> **Important:** Use unique parameter names in your config to avoid conflicts with other attacks. A good practice is to suffix parameters with your attack name:
> ```json
> {
>     "snr_db_myattack": 20,
>     "threshold_myattack": 0.5
> }
> ```
> A 2.x `--<parameter>` flag reaches every attack that declares that name.

4.	Dockerizing (Optional)

    If your attack requires AI models, it runs as a container and
    `attack.py` becomes a thin HTTP client. The container side follows the
    standard layout:

  - Add `inference.py` holding all inference logic in one class deriving
    from `BaseAttackEngine`, named after the plugin, plus the stable alias:

    ```python
    import numpy as np
    from deepmarkpy.core.inference import BaseAttackEngine

    class NewAttackEngine(BaseAttackEngine):
        def __init__(self, config: dict, device: str | None = None):
            self.config = config          # load weights here

        def apply(self, audio, sampling_rate: int, **params) -> np.ndarray:
            """Return the attacked audio."""

    Engine = NewAttackEngine
    ```

  - Add `app.py`: a thin FastAPI adapter that parses the request, calls the
    engine, and serializes the result. Keep inference out of it.
  - Add port to the .env file.
  - Write a Dockerfile to containerize it.
  - Add it to docker-compose.yml.

5.	Run the Benchmark
```bash 
# Add it to your config file's attacks.list, then:
deepmark-benchmark --config my_config.json --wav_files_dir /path/to/audio
```

### Adding a New Watermarking Model

1.	Create a New Model Folder

Inside `src/deepmarkpy/plugins/models`, create a folder:
```Shell 
mkdir src/deepmarkpy/plugins/models/new_model
```

2.	Add model.py
```python
import numpy as np
from deepmarkpy.core.base_model import BaseModel

class NewModel(BaseModel):
    def embed(self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Embeds a watermark in the audio."""
        return audio + 0.01 * watermark_data

    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Detects watermark from the audio."""
        return np.random.randint(0, 2, size=16)

    def is_watermarked(self, detect_output) -> bool:
        """Decide whether a watermark is present based on detect() output.

        Required for detection_reliability mode. Each model defines
        its own logic here. Examples:
          - Zero-bit model: return bool(detect_output)
          - Confidence model: return confidence >= threshold
        """
        return bool(np.any(detect_output))
```

> **`is_watermarked()` is optional** — only needed if you want the model to
> support the `detection_reliability` mode. Models without it work normally in the
> standard benchmark mode.

3.	Add config.json
```json
{
    "sampling_rate": 16000,
    "watermark_size": 16,
    "returns_confidence": false,
    "is_zero_bit": false
}
```

4.	Dockerizing (Optional)

    Models that load ML weights run as containers, with `model.py` acting as
    a thin HTTP client. Add `inference.py` with an engine class deriving
    from `BaseModelEngine`, plus a thin `app.py`, a Dockerfile, a port in
    `.env`, and a `docker-compose.yml` service:

    ```python
    import numpy as np
    from deepmarkpy.core.inference import BaseModelEngine

    class NewModelEngine(BaseModelEngine):
        def __init__(self, config: dict, device: str | None = None):
            self.config = config          # load weights here

        def embed(self, audio, watermark_data, sampling_rate: int) -> np.ndarray:
            """Return the watermarked audio."""

        def detect(self, audio, sampling_rate: int):
            """Return the detected watermark."""

    Engine = NewModelEngine
    ```

5.	Run the Benchmark with the New Model
```Shell
# Put "NewModel" in your config file's models list, then:
deepmark-benchmark --config my_config.json --wav_files_dir /path/to/audio
```

### Docker Integration

To run AI-based plugins inside Docker:
```Shell
docker-compose up --build -d
```
To stop:
```shell
docker-compose down
```

## Contributing

We welcome contributions! Feel free to:
- Report issues
- Suggest new features
- Submit pull requests

Benchmark behavior is intentionally frozen in this release line: known
quirks are catalogued internally and scheduled for a dedicated fix
release. Avoid changing observable behavior in passing — the golden and
contract fixture suites under `tests/fixtures/` will fail if you do.

## Citation

If you use DeepMark Benchmark in your research, please cite our paper:

```
@ARTICLE{11488564,
  author={Kovačević, Slavko and Nešović, Elena and Pavlović, Kosta and Nedić, Petar and Djurović, Igor},
  journal={IEEE Access}, 
  title={DeepMark Benchmark: Redefining Audio Watermarking Robustness}, 
  year={2026},
  volume={14},
  number={},
  pages={62031-62044},
  keywords={Digital audio players;Digital audio broadcasting;Radio broadcasting;Frequency modulation;Filtering;Filters;Equalizers;Low-pass filters;Notch filters;Circuits and systems;Audio watermarking;benchmarking;deep learning;generative AI;robustness evaluation;watermark removal},
  doi={10.1109/ACCESS.2026.3685903}}
```

## License

This project is licensed under MIT License.
