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

This command builds the Docker images for all containerized models/attacks (defined in `docker-compose.yml`) using the configuration from `.env` and starts them in the background. This step is **required** if you intend to use plugins like `audioseal`, `vae`, `diffusion`, etc.
```bash
docker build -f Dockerfile.base -t ml-services-base:latest .
docker-compose -f docker-compose.yml build
```
You can check the status of the services using `docker-compose ps`. The first build might take some time.

> **Tip:** You don't need to run all services at once. If you only need specific attacks or models, you can build and run them individually:
> ```bash
> docker-compose up -d audioseal diffusion  # Only start AudioSeal model and Diffusion attack
> ```

### 2. Run the CLI

Everything the benchmark *measures* is described by a JSON config file.
The command line carries only operational settings: which config to run,
where the audio is, where the reports go, seeding, verbosity, audio
dumping, and the plugin directory. A run is therefore reproducible from
its config file plus an audio directory.

Start from a template — one per mode, each fully commented:

```bash
mkdir -p configs
deepmark-benchmark --init benchmark              > configs/benchmark.json
deepmark-benchmark --init no_attacks             > configs/no_attacks.json
deepmark-benchmark --init detection_reliability  > configs/detection_reliability.json
```

Edit the file, check it, then run it:

```bash
deepmark-benchmark --config configs/benchmark.json --validate-only
deepmark-benchmark --config configs/benchmark.json --wav_files_dir /path/to/audio
```

`--validate-only` checks the file against the discovered plugins and
exits without running anything, so a typo costs a second rather than an
hour. It does not require the Docker services to be up.

**Three modes, three files.** Each config declares its own `"mode"`, and
only accepts the keys that mode actually uses — a `no_attacks` file that
mentions attacks is an error, not a silently ignored section.

| Mode | What it measures | File |
|------|------------------|------|
| `benchmark` | Watermark survival under each attack, plus the audio-quality cost | `configs/benchmark.json` |
| `no_attacks` | Baseline: does the model read back its own watermark, and what does embedding alone cost? | `configs/no_attacks.json` |
| `detection_reliability` | False positives on clean audio and false negatives on watermarked audio | `configs/detection_reliability.json` |

Pass several files to run several modes in one invocation. Each must
declare a different mode; they run in the order given and write distinct
filenames into the shared report directory:

```bash
deepmark-benchmark --config configs/benchmark.json configs/detection_reliability.json \
                  --wav_files_dir /path/to/audio
```

**Comparing models.** List two or more models in `benchmark.json` and
each is run in turn, with a comparative report added:

```json
"models": ["AudioSealModel", "AwareModel", "PerthModel"]
```

**Selecting attacks.** In `benchmark.json` or
`detection_reliability.json`:

```json
"attacks": {
  "groups": ["audio_distortion", "desynchronization"],
  "list": ["LowpassFilterAttack", "GaussianNoiseAttack"]
}
```

Groups and list combine, duplicates removed. In `benchmark` mode leaving
both empty runs every discovered attack; in `detection_reliability` mode
it measures the no-attack baseline only, since attacks are *added* to a
baseline that is always measured.

A group expands to the attacks it *declares*, not to the ones that
happen to have imported. If a plugin failed to load, the run stops and
names it rather than quietly measuring a smaller set.

| Group | Attacks |
|-----|---------|
| `process_disruption` | `CrossModelAttack`, `CollusionAttack`, `ZeroBitCollusionAttack`, `Collusion2Attack`, `SameModelAttack` |
| `audio_editing` | `CutSamplesAttack`, `CropBeginningAttack`, `CropRandomAttack`, `WaveletAttack`, `LowpassFilterAttack`, `HighpassFilterAttack`, `BandstopFilterAttack`, `SmoothingAttack`, `ChorusAttack`, `FlangerAttack`, `EchoAttack`, `EqualizerAttack`, `QuantizationAttack`, `STFTQuantizationAttack`, `PCMQuantizationAttack`, `Mp3CompressionAttack`, `EncodecAttack`, `DescriptAudioCodecAttack`, `OpusCodecAttack`, `Codec2VocoderAttack`, `ResamplingPolyAttack`, `MixingAttack` |
| `audio_distortion` | `GaussianNoiseAttack`, `PinkNoiseAttack`, `SignInversionAttack`, `LPCAttack`, `AdditiveNoiseAttack` |
| `desynchronization` | `TimeStretchAttack`, `PitchShiftAttack`, `InvertedTimeStretchAttack`, `ZeroCrossInsertsAttack`, `FlipSamplesAttack` |
| `ai_attacks` | `SpeechEnhancement1Attack`, `SpeechEnhancement2Attack`, `SpeechTokenizationAttack`, `NeuralVocoderAttack`, `DiffusionAttack`, `VAEAttack` |
| `transmission` | `ReplayAttack`, `NetworkTransmissionAttack` |

**All command-line options**

| Flag | Purpose |
|------|---------|
| `--config PATH [PATH ...]` | Config file(s) to run, one per mode, in the order given. Required unless `--init` is used |
| `--init MODE` | Print a fresh, fully-commented config file for `MODE` to stdout and exit |
| `--validate-only` | Validate the config file(s) against the discovered plugins and exit without running anything |
| `--wav_files_dir DIR` | Directory of `.wav`/`.mp3` files. Overrides `general.wav_files_dir`; required if neither sets it |
| `--report_dir DIR` | Where reports and saved audio go (default `./report`). **Its contents are deleted at the start of every run**, so do not point it at a directory holding anything else |
| `--seed N` | Seed the host-side RNGs so a run can be repeated. Off by default, which keeps the watermark payload and attack noise fresh per run. Does not reach the diffusion, VAE, speech_enhancement_2 or network_transmission services, which stay stochastic server-side |
| `--save_audio` | Write watermarked and attacked audio to `<report_dir>/audio/` |
| `--verbose` | Per-file and per-attack progress logging |
| `--plugins_dir DIR` | Load third-party plugins from this directory (also `DEEPMARK_PLUGINS_DIR`) |
| `--version` | Print the installed version and exit |

That is the whole list — it does not grow with the plugin set. Models,
attacks, attack parameters, metrics, statistics, duration groups and
cropping are config-file settings, so there is one place to look for
each of them and no flag whose name has to be globally unique.

**Precedence.** The six operational keys above also exist under
`general` in every config file. The command line wins, the config file
is next, and the built-in default applies when neither says anything
(`report_dir` = `report`, no seed, no verbosity, no audio dumping).
`wav_files_dir` has no built-in default: set it in one place or the
other, or the run stops before it starts. Nothing else appears in both
places, so there is no other conflict to resolve.

**Exit codes.** `0` success, `1` a runtime failure (for example a
missing audio directory), `2` an invalid configuration or an unreachable
model service — nothing was run.

Every attack belongs to exactly one group; the canonical mapping lives in
`src/deepmarkpy/utils/attack_groups.py` — update it there when adding a new
attack so the reports pick the right metrics for it, and a test will fail if
one is left ungrouped.

Metrics that compare the two signals sample by sample (PSNR, SI-SDR, STOI,
MCD, NCM) report an attack's timing shift as if it were quality loss, and the
benchmark does not resynchronize — a desynchronization attack is meant to move
the time axis. Which is why the shipped default disables them for the
`desynchronization` group, leaving MCD, ViSQOL and the reference-free NISQA
dimensions. Enable them there if you want them, in which case they are printed
like any other metric: a metric appears in a report because the config asked
for it, and the report adds no commentary of its own about whether the value
is worth reading.

The same choice is available per group for every metric. `SignInversionAttack`
is the other classic case: SI-SDR is scale-invariant, so a polarity flip leaves
its score exactly at the no-attack value even though detection collapses.
Disable `si_sdr` for `audio_distortion` if that reading is unhelpful.

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
`Codec2VocoderAttack_1200`, etc. Supported bitrates: 700, 1200, 1300, 1400,
1600, 2400, 3200 bps. Unsupported values are skipped with a warning.

> **Attack naming convention:** Attack class names must not contain underscores.
> The benchmark uses underscores as a separator between the base attack name and
> a parameter suffix (e.g. `Codec2VocoderAttack_700`). If an attack name contains
> an underscore, the group lookup will incorrectly treat the part after the last
> underscore as a suffix.

### Configuration Files

JSON has no comment syntax, so the shipped templates document themselves
with keys that start with an underscore. The parser ignores every
`_`-prefixed key — keep them, delete them, or add your own notes the
same way.

An abridged `configs/benchmark.json`:

```json
{
  "mode": "benchmark",
  "general": {
    "wav_files_dir": null,
    "report_dir": "report",
    "seed": null,
    "verbose": false,
    "save_audio": false,
    "plugins_dir": null
  },
  "models": ["AudioSealModel"],
  "attacks": { "groups": [], "list": [] },
  "attack_parameters": {
    "GaussianNoiseAttack": { "snr_db_gaussian_noise": 25 }
  },
  "calculate_quality_metrics": true,
  "statistics": ["mean", "std", "median", "p5", "p95", "worst_case"],
  "metrics": {
    "defaults": {
      "accuracy": { "enabled": true, "statistics": ["mean", "worst_case"] },
      "pesq": { "enabled": true },
      "mcd":  { "enabled": true }
    },
    "per_group": {
      "desynchronization": { "pesq": { "enabled": false } }
    }
  },
  "comparison": { "primary_statistic": "mean" },
  "crop_before_attack": null,
  "duration_groups": { "boundaries": [], "include_overall": false }
}
```

**Per-attack parameters** are keyed by attack class name for its default
version, or `AttackName:version` for one named version. A bare name
touches only the default, never the named presets. Naming a version the
plugin does not have *defines* it, provided you give **all** of that
attack's parameters; give only some and the entry is skipped with a
`W010` warning, since the rest would silently come from the default
preset. Both the attack name and each parameter are checked against the
discovered plugins, with the type checked against the plugin's own
default, before the run starts.

To see which version will run with which values, before running
anything:

```bash
deepmark-benchmark --config configs/benchmark.json --validate-only
```

```
configs/benchmark.json: attack parameters in effect --
    GaussianNoiseAttack (mild): snr_db_gaussian_noise=45
    GaussianNoiseAttack (default): snr_db_gaussian_noise=35
    GaussianNoiseAttack (aggressive): snr_db_gaussian_noise=20
    GaussianNoiseAttack (brutal): snr_db_gaussian_noise=10
    GaussianNoiseAttack (extreme): snr_db_gaussian_noise=3
```

Those five versions come from the config file: no shipped attack
declares presets of its own, so `mild`, `aggressive`, `brutal` and
`extreme` are ones that file defines. The values shown are the
*effective* ones — the plugin's default with the override for that
version applied on top, which is what `apply()` receives. The
same mapping is written to `run_metadata.json` under
`attack_parameters_resolved`, so a finished run can still be traced back
to the values that produced it.

**`calculate_quality_metrics`** is the master switch for the signal
metrics. When `true`, the `metrics` block decides what runs. When `false`
or absent, only accuracy and the always-on trio — PESQ, ViSQOL, STOI —
are computed, uniformly across every group, and the enable flags are
ignored; per-metric `statistics` still apply, and `ber`/`emr` keep
honouring their flags because both are derived from accuracy for free.
Set it `false` for a fast robustness-only pass.

#### Per-group metrics and statistics

Which metrics make sense depends on the attack family. A time-stretch
moves the time axis, so PESQ and STOI report the shift rather than a
quality change; a length-changing crop cannot be compared to the
original at all. Those exclusions used to be hardcoded in the report
generators, where no config could reach them. They are now yours to set.

`metrics.defaults` applies to every attack group. `metrics.per_group`
overrides it for one group. **Inheritance is per metric, not per block**:
a group section listing only `pesq` inherits every other metric from
`defaults` untouched, so each section states only what it changes.

Statistics resolve in this order — first list that exists wins, and the
report columns appear in *that list's* order:

1. `metrics.per_group.<group>.<metric>.statistics`
2. `metrics.per_group.<parent group>.<metric>.statistics` (subsections only)
3. `metrics.defaults.<metric>.statistics`
4. the top-level `statistics` list
5. all eight statistics

Delete the top-level `statistics` key entirely and all eight are
computed: `mean`, `std`, `median`, `p5`, `p10`, `p95`, `p99`,
`worst_case`.

Configurable group keys:

| Key | Covers |
|-----|--------|
| `process_disruption`, `audio_editing`, `audio_distortion`, `desynchronization`, `ai_attacks`, `transmission` | The six attack families |
| `frequency_filtering`, `temporal_editing`, `audio_effects`, `compression_quantization` | Report subsections of `audio_editing`. They inherit from `audio_editing`, which inherits from `defaults`. Not selectable in `attacks.groups` |
| `other` | Attacks belonging to no declared family, including third-party plugins |

A section for a group whose attacks this run does not select is fine —
it is reported as unused and ignored. A **misspelled** group name is an
error, with the closest valid name suggested.

The shipped templates reproduce the metric matrix the benchmark has
always applied, so an unedited `--init` file changes nothing about what
a run reports. A test enforces that.

> **NISQA costs the same whichever dimensions you enable.** All five
> come back from a single request, so listing fewer narrows the tables
> without making the run faster. The validator says so when it sees a
> partial selection.

#### Validation

Every problem in every file is reported together, before anything runs,
each with a stable code, the exact JSON path, the offending value, and a
suggestion when the value looks like a typo:

```
Found 3 configuration problems:

[E005] configs/benchmark.json: mode
    unknown mode. Valid modes: benchmark, no_attacks, detection_reliability.
    got: "benchmrak"
    did you mean 'benchmark'?

[E027] configs/benchmark.json: metrics.per_group.desync
    unknown attack group. Configurable groups: process_disruption, ...
    got: "desync"
    did you mean 'desynchronization'?

[E031] configs/benchmark.json: crop_before_attack
    must be greater than 0 and less than 100: it is the percentage cropped
    from the start of the watermarked audio. Use null to disable cropping.
    got: 150
```

Warnings are printed and the run proceeds: a `per_group` section for a
group this run does not reach, a `metrics` block that
`calculate_quality_metrics` is ignoring, or a metric whose backing
service is unreachable.

> **Upgrading from the single `benchmark_config.json`?** Its `modes`
> block and `"mean:T std:F ..."` statistic strings are gone. Passing an
> old file produces an `[E038]` error naming what changed; run
> `deepmark-benchmark --init benchmark` for a fresh one.

### Running with the pre-2.0 flags

Scripts written against the previous release keep working. When
`--config` is absent the old flags are accepted and assembled into a
configuration internally, so they run through the same validator and the
same reports:

```bash
deepmark-benchmark \
  --wm_models AudioSealModel PerthModel \
  --wav_files_dir ./audio \
  --attack_groups audio_distortion desynchronization \
  --calculate_quality_metrics \
  --crop_before_attack 10 \
  --snr_db_gaussian_noise 25
```

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

Passing `--no_attacks` and `--detection_reliability` together runs both,
as it always did — one config per mode.

Give `--config` and the file decides; any flag from the table is then
ignored and named in a warning. The flags cover what they always covered,
which is less than a config file can say: per-group metrics and
statistics, the `efficiency` section, `duration_groups`, `comparison` and
per-version attack parameters have no flag, and a bare parameter flag
reaches an attack's default preset only.

### Attack Versions

An attack can carry several parameter presets (versions). No shipped
attack does — every plugin's `config.json` holds one flat parameter set,
which is its `default` version — so in practice you define the ones you
want in your config file (see below), and a plugin may also declare them
itself by putting each preset under a name:

```json
{
  "default": {"snr_db_gaussian_noise": 35},
  "aggressive": {"snr_db_gaussian_noise": 10},
  "mild": {"snr_db_gaussian_noise": 50}
}
```

Define them in `attack_parameters` — giving every parameter the attack
has, so the version is fully specified — and select them with the
`AttackName:version` syntax:

```json
"attack_parameters": {
  "GaussianNoiseAttack:aggressive": {"snr_db_gaussian_noise": 10},
  "GaussianNoiseAttack:mild": {"snr_db_gaussian_noise": 50}
},
"attacks": {
  "list": ["GaussianNoiseAttack:aggressive", "GaussianNoiseAttack:mild"]
}
```

Naming a multi-version attack **without** a version expands it into one
run per version. The report labels each as `GaussianNoise (aggressive)`.
A bare name and `:default` are the same target. Any other version name
must be defined in `attack_parameters`; naming one that is not is a
validation error listing the versions that exist.

### 3. Quality Metrics (Optional)

Set `"calculate_quality_metrics": true` in the config file to compute
audio quality metrics and generate a detailed report, and choose which
ones under `metrics` (see [Per-group metrics and
statistics](#per-group-metrics-and-statistics)):

```json
"calculate_quality_metrics": true,
"metrics": {
  "defaults": {
    "pesq": { "enabled": true },
    "stoi": { "enabled": true },
    "mcd":  { "enabled": false }
  }
}
```

**Audio Quality Metrics:**

| Metric | Description | Range |
|--------|-------------|-------|
| PESQ | Perceptual Evaluation of Speech Quality | 1.0 - 4.5 |
| PSNR | Peak Signal-to-Noise Ratio | dB (higher = better) |
| SI-SDR | Scale-Invariant Signal-to-Distortion Ratio | dB (higher = better) |
| MCD | Mel Cepstral Distortion | dB (lower = better) |
| ViSQOL* | Virtual Speech Quality Objective Listener | 1.0 - 5.0 (MOS) |

*ViSQOL is **optional**. The [`visqol`](https://github.com/google/visqol) package is not in `requirements.txt` because its installation requires Bazel and platform-specific build steps. Install it separately if you want ViSQOL scores in your reports; otherwise this metric is silently skipped and all other metrics are still computed.

**Non-Intrusive Quality (NISQA):**

| Metric | Description | Range |
|--------|-------------|-------|
| NISQA MOS | Overall speech quality (Mean Opinion Score) | 1.0 - 5.0 |
| NISQA NOI | Noisiness | 1.0 - 5.0 |
| NISQA DIS | Discontinuity | 1.0 - 5.0 |
| NISQA COL | Coloration | 1.0 - 5.0 |
| NISQA LOUD | Loudness | 1.0 - 5.0 |

NISQA is a **non-intrusive** metric (it does not require a clean reference signal), which makes it particularly useful for desynchronization attacks where intrusive metrics like PESQ break down. To enable NISQA:

1. Install the package:
   ```bash
   pip install nisqa
   ```

2. Download the model weights (`nisqa.tar`, ~1.1 MB) from the [NISQA GitHub repository](https://github.com/gabrielmittag/NISQA/tree/master/weights):
   ```bash
   mkdir -p weights
   wget -O weights/nisqa.tar https://github.com/gabrielmittag/NISQA/raw/master/weights/nisqa.tar
   ```

3. The benchmark automatically looks for `weights/nisqa.tar` in the project root. To use a custom path, set the environment variable:
   ```bash
   export NISQA_WEIGHTS_PATH=/path/to/nisqa.tar
   ```

If NISQA is not installed or the weights file is missing, the metric is silently skipped and all other metrics still run.

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

> **ViSQOL is optional.** The `visqol` package is not installed by the
> default `requirements.txt` because it requires Bazel and
> platform-specific build steps. If it is not installed the benchmark
> logs an informational message once and skips the ViSQOL column while
> all other metrics still run. To enable it, follow the build
> instructions at https://github.com/google/visqol and install the
> resulting Python package into the same environment.

### 4. Detection Reliability (Optional)

`configs/detection_reliability.json` measures false positive and false
negative rates. It takes **exactly one model**, which must implement
`is_watermarked()` (see [Adding a New Watermarking
Model](#adding-a-new-watermarking-model)).

```bash
deepmark-benchmark --config configs/detection_reliability.json \
                  --wav_files_dir /path/to/audio
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

There is no `ber` metric in this mode: each file is scored as a binary
detected/not-detected outcome, so there are no payload bits to disagree
and a bit error rate would be meaningless. Naming it is a validation
error rather than a silently ignored key.

Results are saved to `report/detection_reliability.json` and a dedicated `detection_reliability_report.pdf` is generated.

> **Note:** Only models that implement `is_watermarked()` support this mode.
> Each model defines its own detection logic — zero-bit models check the
> binary output directly, while confidence-based models compare against a
> threshold. If a model does not implement this method, a clear error is
> raised at runtime.

### 5. Save Audio (Optional)

Use `--save_audio` (or `"save_audio": true` under `general`) to write
intermediate audio files to disk for manual inspection. Files are saved
to `<report_dir>/audio/`.

```bash
deepmark-benchmark --config configs/detection_reliability.json \
                  --wav_files_dir /path/to/audio \
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
- `benchmark_stats.json` – Per-attack statistics, carrying exactly the metrics and statistics the config asked for
- `run_metadata.json` – Version, git revision, seed, plugin inventory, and the config file the run came from
- `benchmark_report.tex/.pdf` – Accuracy per attack family, bar chart, and performance analysis
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
than rendered as a table of `N/A`s or dropped without a word.

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
> This prevents parameter overwrites when multiple attacks are used together.

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
deepmark-benchmark --config configs/benchmark.json --wav_files_dir /path/to/audio
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
    "watermark_size": 16
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
deepmark-benchmark --config configs/benchmark.json --wav_files_dir /path/to/audio
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
