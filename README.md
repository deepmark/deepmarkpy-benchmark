# DeepMarkPy Benchmark

DeepMark Benchmark is a modular and scalable Python platform for evaluating the robustness of audio watermarking systems. It enables testing against various attacks, including both simple signal manipulations and advanced AI-based disruptions, using a containerized architecture for consistency and ease of use.

## Features

*   **Extensible Plugin System:** Easily add new watermarking models and attacks.
*   **Containerized Services:** Key models and attacks run as isolated Docker services for dependency management and reproducibility.
*   **Centralized Configuration:** Service network ports are managed via a single `.env` file.
*   **Client-Server Architecture:** The benchmark runner communicates with containerized plugins via HTTP.
*   **Standardized Execution:** Provides a CLI for running benchmarks and collecting results.

## Architecture Overview

This benchmark uses a client-server architecture. Core watermarking models and complex attacks (often AI-based) run as independent web services managed by Docker Compose. The main benchmark script (`src/run.py`) acts as a client, communicating with these services via HTTP requests over a Docker network to perform embedding, attacking, and detection. This isolates complex dependencies within containers.

## Prerequisites

*   Python 3.9+
*   Docker (Install Docker)
*   Docker Compose (Install Docker Compose)

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/deepmarkpy/deepmarkpy-benchmark.git
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
Ensure the Docker services are running (`docker-compose up -d`) if you are using containerized plugins. Then, execute the main benchmark script from your activated virtual environment (if used) or directly:

**Single model:**
```bash
python src/run.py --wav_files_dir /path/to/audio \
                  --wm_model AudioSealModel \
                  --attack_types GaussianNoiseAttack LowpassFilterAttack
```

**Multiple models (comparative report):**
```bash
python src/run.py --wav_files_dir /path/to/audio \
                  --wm_models AudioSealModel AwareModel PerthModel \
                  --attack_types GaussianNoiseAttack LowpassFilterAttack
```

When using `--wm_models` with two or more models, the benchmark runs each model individually and generates a comparative report with accuracy tables, radar chart, and per-metric comparisons. If only one model is provided via `--wm_models`, it behaves the same as `--wm_model`.

**Using attack groups:**
```bash
python src/run.py --wav_files_dir /path/to/audio \
                  --wm_model AudioSealModel \
                  --attack_group audio_distortion desynchronization
```

Available attack groups: `process_disruption`, `audio_editing`, `audio_distortion`, `desynchronization`, `ai_attacks`, `transmission`. Groups can be combined with `--attack_types` to add individual attacks.

### 3. Quality Metrics (Optional)

Use `--calculate_quality_metrics` to compute audio quality metrics and generate a detailed report:

```bash
python src/run.py --wav_files_dir /path/to/audio --wm_model AudioSealModel \
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

### 4. View Results

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

Inside src/plugins/attacks, create a new folder with the attack name:
```Shell
mkdir src/plugins/attacks/new_attack
```
2.	Add attack.py
Create a file attack.py inside your folder:
```python 
import numpy as np
from core.base_attack import BaseAttack

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

    If your attack requires AI models:
  - Add app.py for FastAPI service.
  - Add port to the .env file.
  - Write a Dockerfile to containerize it.
  - Add it to docker-compose.yml.

5.	Run the Benchmark
```bash 
python src/run.py --wav_files_dir /path/to/audio --wm_model AudioSealModel --attack_types NewAttack
```

### Adding a New Watermarking Model

1.	Create a New Model Folder

Inside src/plugins/models, create a folder:
```Shell 
mkdir src/plugins/models/new_model
```

2.	Add model.py
```python
import numpy as np
from core.base_model import BaseModel

class NewModel(BaseModel):
    def embed(self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Embeds a watermark in the audio."""
        return audio + 0.01 * watermark_data

    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Detects watermark from the audio."""
        return np.random.randint(0, 2, size=16)
```

3.	Add config.json
```json
{
    "watermark_size": 16
}
```

4.	Run the Benchmark with the New Model
```Shell
python src/run.py --wav_files_dir /path/to/audio --wm_model NewModel --attack_types CutSamplesAttack
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

## Citation

If you use DeepMark Benchmark in your research, please cite our paper:

```
@ARTICLE{11488564,
  author={Kovačević, Slavko and Nešović, Elena and Pavlović, Kosta and Nedić, Petar and Djurović, Igor},
  journal={IEEE Access}, 
  title={DeepMark Benchmark: Redefining Audio Watermarking Robustness}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Digital audio players;Digital audio broadcasting;Radio broadcasting;Frequency modulation;Filtering;Filters;Equalizers;Low-pass filters;Notch filters;Circuits and systems;Audio watermarking;benchmarking;deep learning;generative AI;robustness evaluation;watermark removal},
  doi={10.1109/ACCESS.2026.3685903}}
```

## License

This project is licensed under MIT License.
