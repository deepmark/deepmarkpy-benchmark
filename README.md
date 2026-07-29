# DeepMark Benchmark

DeepMark Benchmark is a modular and scalable Python platform for evaluating the robustness of audio watermarking systems. It enables testing against various attacks, including both simple signal manipulations and advanced AI-based disruptions, using a containerized architecture for consistency and ease of use.

## Features

*   **Extensible Plugin System:** Easily add new watermarking models and attacks.
*   **Containerized Services:** Key models and attacks run as isolated Docker services for dependency management and reproducibility.
*   **Centralized Configuration:** Service network ports are managed via a single `.env` file.
*   **Client-Server Architecture:** The benchmark runner communicates with containerized plugins via HTTP.
*   **Standardized Execution:** Provides a CLI for running benchmarks and collecting results.

## Architecture Overview

This benchmark uses a client-server architecture. Watermarking models and
dependency-heavy attacks run as independent services managed by Docker Compose.
The installed CLI (`deepmark-benchmark`) stays slim and calls those services
over HTTP. Signal-processing attacks continue to run directly in the CLI
process.

Every AI attack image installs the same local `deepmarkpy` package and imports
its packaged implementation. Containers contain no copied attack logic; they
only provide an isolated dependency environment and the shared HTTP adapter.
The isolated attacks are VAE, diffusion, Encodec, Descript Audio Codec,
speech tokenization, neural vocoder, and both speech-enhancement attacks.

XCodec2 and ClearVoice intentionally use separate Python environments from the
modern Torch stack. Do not install several AI extras into the CLI environment;
build and run the corresponding Compose services instead.

The CLI connects to each attack on `localhost` and the port declared in `.env`.
For services on another host, set `DEEPMARK_ATTACK_HOST` globally or set an
attack-specific host variable such as `VAE_PORT_HOST`. The existing port
variables can also be overridden in the CLI environment.

## Prerequisites

*   Python 3.11+
*   Docker (Install Docker)
*   Docker Compose (Install Docker Compose)

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/deepmark/deepmarkpy-benchmark.git
cd deepmarkpy-benchmark
```

### 2. Review Environment File (`.env`)

This repository includes a `.env` file which defines the default network ports used by the various Docker services (models and attacks). Docker Compose automatically reads this file when starting the services.

*   **Action:** Review the ports defined in the `.env` file. You generally don't need to change the defaults unless they conflict with other services already running on your machine. If a conflict exists, modify the corresponding port number in the `.env` file before proceeding.

### 3. Install Core Dependencies (Optional - for development/direct script interaction)

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

Install the package in editable mode for local development:

```bash
pip install -e ".[dev]"
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

These commands build the shared model base, build all service images using the
configuration from `.env`, and start them. This is required for plugins such as
`audioseal`, `vae`, `diffusion`, and `encodec`.
```bash
docker build -f Dockerfile.base -t ml-services-base:latest .
docker compose build
docker compose up -d
```
You can check the status of the services using `docker compose ps`. The first build might take some time.

> **Tip:** You don't need to run all services at once. If you only need specific attacks or models, you can build and run them individually:
> ```bash
> docker compose up --build -d audioseal diffusion
> ```

### 2. Run the CLI
Ensure the Docker services are running (`docker compose up -d`) if you are using containerized plugins. Then, execute the main benchmark script from your activated virtual environment (if used) or directly:

**Single model:**
```bash
deepmark-benchmark --wav_files_dir /path/to/audio \
                  --wm_model AudioSealModel \
                  --attack_types GaussianNoiseAttack LowpassFilterAttack
```

**Multiple models (comparative report):**
```bash
deepmark-benchmark --wav_files_dir /path/to/audio \
                  --wm_models AudioSealModel AwareModel PerthModel \
                  --attack_types GaussianNoiseAttack LowpassFilterAttack
```

When using `--wm_models` with two or more models, the benchmark runs each model individually and generates a comparative report with accuracy tables, radar chart, and per-metric comparisons. If only one model is provided via `--wm_models`, it behaves the same as `--wm_model`.

**Using attack groups:**
```bash
deepmark-benchmark --wav_files_dir /path/to/audio \
                  --wm_model AudioSealModel \
                  --attack_groups audio_distortion desynchronization
```

`--attack_groups` accepts one or more group tags. The tag is what you pass on the
command line; the table below lists every group together with the attacks it
covers. Groups can be combined with `--attack_types` to add individual attacks.

| Tag | Attacks |
|-----|---------|
| `process_disruption` | `CrossModelAttack`, `CollusionAttack`, `ZeroBitCollusionAttack`, `Collusion2Attack`, `SameModelAttack` |
| `audio_editing` | `CutSamplesAttack`, `CropBeginningAttack`, `CropRandomAttack`, `WaveletAttack`, `LowpassFilterAttack`, `HighpassFilterAttack`, `BandstopFilterAttack`, `SmoothingAttack`, `ChorusAttack`, `FlangerAttack`, `EchoAttack`, `EqualizerAttack`, `QuantizationAttack`, `STFTQuantizationAttack`, `PCMQuantizationAttack`, `Mp3CompressionAttack`, `EncodecAttack`, `DescriptAudioCodecAttack`, `OpusCodecAttack`, `Codec2VocoderAttack`, `ResamplingPolyAttack`, `MixingAttack` |
| `audio_distortion` | `GaussianNoiseAttack`, `PinkNoiseAttack`, `SignInversionAttack`, `LPCAttack` |
| `desynchronization` | `TimeStretchAttack`, `PitchShiftAttack`, `InvertedTimeStretchAttack`, `ZeroCrossInsertsAttack`, `FlipSamplesAttack`, `ReplacementAttack`, `Replacement2Attack` |
| `ai_attacks` | `SpeechEnhancement1Attack`, `SpeechEnhancement2Attack`, `SpeechTokenizationAttack`, `NeuralVocoderAttack`, `DiffusionAttack` |
| `transmission` | `ReplayAttack`, `NetworkTransmissionAttack` |

Attacks not listed in any group fall under "Other Attacks" in the detailed
report. The canonical mapping lives in `src/deepmarkpy/utils/attack_groups.py` — update it
there when adding a new attack so the reports pick the right metrics for it.

**Codec2 Vocoder Attack:**

`Codec2VocoderAttack` simulates transmission through a low-bitrate voice channel
(similar to MELP/MELPe military vocoders). It encodes audio at a given bitrate
using the Codec2 codec and decodes it back to PCM. The `bitrate` parameter in
`config.json` accepts either a single value or a list of bitrates:

```json
{"bitrate": [700, 1200, 2400]}
```

From the CLI, override it with the namespaced attack flag:

```bash
deepmark-benchmark --wav_files_dir /path/to/wavs --wm_model AudioSealModel \
                  --attack_types Codec2VocoderAttack \
                  --codec2_vocoder.bitrate 1200
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

### 3. Quality Metrics (Optional)

Use `--calculate_quality_metrics` to compute audio quality metrics and generate a detailed report:

```bash
deepmark-benchmark --wav_files_dir /path/to/audio --wm_model AudioSealModel \
                  --attack_types GaussianNoiseAttack LowpassFilterAttack \
                  --calculate_quality_metrics
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

Use `--detection_reliability` to measure false positive and false negative rates. This mode supports a **single model** per run and requires the model to implement `is_watermarked()` (see [Adding a New Watermarking Model](#adding-a-new-watermarking-model)).

```bash
deepmark-benchmark --wav_files_dir /path/to/audio \
                  --wm_model PerthModel \
                  --detection_reliability
```

Without attacks the mode measures:
- **False positive**: detection on clean (unwatermarked) audio reports a watermark present.
- **False negative**: detection on watermarked audio fails to find the watermark.

When combined with `--attack_types` or `--attack_groups`, FP/FN are additionally reported per attack (attack applied to clean audio for FP, attack applied to watermarked audio for FN).

```bash
deepmark-benchmark --wav_files_dir /path/to/audio \
                  --wm_model AudioSealModel \
                  --detection_reliability \
                  --attack_types GaussianNoiseAttack LowpassFilterAttack
```

Results are saved to `report/detection_reliability.json` and a dedicated `detection_reliability_report.pdf` is generated.

> **Note:** Only models that implement `is_watermarked()` support this mode.
> Each model defines its own detection logic — zero-bit models check the
> binary output directly, while confidence-based models compare against a
> threshold. If a model does not implement this method, a clear error is
> raised at runtime.

### 5. Save Audio (Optional)

Use `--save_audio` to write intermediate audio files to disk for manual inspection. Files are saved to `<report_dir>/audio/`.

```bash
deepmark-benchmark --wav_files_dir /path/to/audio \
                  --wm_model AudioSealModel \
                  --detection_reliability \
                  --attack_types GaussianNoiseAttack \
                  --save_audio
```

In detection reliability mode the following files are saved per input file:
- `{filename}_watermarked.wav` — watermarked audio (before any attack)
- `{filename}_{AttackName}_clean.wav` — attack applied to the clean (unwatermarked) audio
- `{filename}_{AttackName}.wav` — attack applied to the watermarked audio

In the standard benchmark mode, watermarked and attacked-watermarked files are saved.

### 6. View Results

The benchmark generates the following outputs in the `report/` directory:

**Single model (`--wm_model`):**
- `benchmark_results.json` – Detailed per-file, per-attack results
- `benchmark_stats.json` – Mean accuracy per attack
- `benchmark_report.tex/.pdf` – Accuracy table, bar chart, and performance analysis
- `detailed_report.tex/.pdf` – Full report with attacks grouped by category and per-group quality/intelligibility metrics (only with `--calculate_quality_metrics`)

**Multiple models (`--wm_models`):**
- `report/<ModelName>/` – Individual model reports (same as single model)
- `report/comparison/` – Comparative report with:
  - Accuracy comparison table with rank-based coloring
  - Radar chart comparing all models
  - Per-metric comparison tables filtered to relevant attacks (only with `--calculate_quality_metrics`)

## Adding a New Plugin

DeepMark Benchmark is designed to allow easy addition of new attacks and watermarking models.

### Adding a New Attack

1.	Create a New Attack Folder

Inside src/deepmarkpy/plugins/attacks, create a new folder with the attack name:
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

> **Important:** Use clean, attack-local parameter names in `config.json`.
> The CLI namespaces them by attack key, so parameters no longer need attack-name
> suffixes:
> ```json
> {
>     "snr_db": 20,
>     "threshold": 0.5
> }
> ```
> Users override these as `--new_attack.snr_db 20` and
> `--new_attack.threshold 0.5`. Existing suffixed flags are kept as deprecated
> aliases where mappings exist.

4.	Isolating an AI Attack

    If an attack requires an AI runtime:

  - Put the numerical/model class in `implementation.py`.
  - Make the discoverable class in `attack.py` inherit
    `deepmarkpy.core.remote_attack.RemoteAttack`.
  - Create `app.py` with
    `deepmarkpy.server.attack_service.create_attack_app`.
  - Add a complete optional dependency extra in `pyproject.toml`.
  - Add an isolated target to `Dockerfile.ai-attacks`.
  - Add its port to `.env` and its service to `docker-compose.yml`.

  The container must install `deepmarkpy` with its attack extra. Do not copy
  the implementation into a container-specific module.

5.	Run the Benchmark
```bash 
deepmark-benchmark --wav_files_dir /path/to/audio --wm_model AudioSealModel --attack_types NewAttack
```

### Adding a New Watermarking Model

1.	Create a New Model Folder

Inside src/deepmarkpy/plugins/models, create a folder:
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

        Required for --detection_reliability support. Each model defines
        its own logic here. Examples:
          - Zero-bit model: return bool(detect_output)
          - Confidence model: return confidence >= threshold
        """
        return bool(np.any(detect_output))
```

> **`is_watermarked()` is optional** — only needed if you want the model to
> support `--detection_reliability`. Models without it work normally in the
> standard benchmark mode.

3.	Add config.json
```json
{
    "watermark_size": 16
}
```

4.	Run the Benchmark with the New Model
```Shell
deepmark-benchmark --wav_files_dir /path/to/audio --wm_model NewModel --attack_types CutSamplesAttack
```

### Docker Integration

To run AI-based plugins inside Docker:
```Shell
docker compose up --build -d
```
To stop:
```shell
docker compose down
```

## Contributing

We welcome contributions! Feel free to:
- Report issues
- Suggest new features
- Submit pull requests

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
