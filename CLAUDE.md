# DeepMarkPy Benchmark — Development Guide

## What This Project Is

Open-source benchmarking framework for evaluating audio watermarking robustness. Evaluates watermarking models against 40+ attacks (signal processing, AI-based, transmission). Published at the GenAI Watermarking Workshop 2025.

## Architecture

- **Package-based**: import root is `deepmarkpy`; models and attacks are auto-discovered from `src/deepmarkpy/plugins/models/` and `src/deepmarkpy/plugins/attacks/` via `PluginManager`
- **Client-server**: Complex ML models/attacks run in Docker containers, accessed via HTTP (FastAPI). Simple attacks run natively
- **Base classes**: `BaseModel` (embed/detect) in `src/deepmarkpy/core/base_model.py`, `BaseAttack` (apply) in `src/deepmarkpy/core/base_attack.py`
- **Config-driven**: Each plugin has a `config.json` with defaults. Model configs include `returns_confidence` and `is_zero_bit` flags. Detection reliability uses `is_watermarked()` on the model class

## Key Files

- `src/deepmarkpy/cli.py` — CLI entrypoint, exposed as `deepmark-benchmark`
- `src/deepmarkpy/benchmark.py` — Core benchmark orchestration (run loop, accuracy computation)
- `src/deepmarkpy/plugin_manager.py` — Auto-discovers plugins from package resources
- `src/deepmarkpy/utils/metrics.py` — PESQ, STOI, PSNR, SI-SDR
- `src/deepmarkpy/utils/report_generator.py` — LaTeX + chart generation
- `docker-compose.yml` — All containerized services
- `.env.example` — Port configuration template

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests are in `tests/` and use `conftest.py` for shared fixtures (sample audio, watermarks, result dicts). Install the package in editable mode when running tests locally.

Current: 74 tests, ~2s runtime. No Docker required for tests.

## Running the Benchmark

```bash
# Start Docker services (if using containerized models/attacks)
docker-compose up -d audioseal
# Run benchmark
deepmark-benchmark --wav_files_dir /path/to/wavs --wm_model AudioSealModel --attack_types GaussianNoiseAttack
```

## Development Conventions

- **Attack parameter names are attack-local** — keep `config.json` keys clean (for example, `snr_db`, `order`) and expose CLI overrides through namespaced flags such as `--replay.snr_db` or `--bandstop_filter.order`. Legacy suffixed flags are compatibility aliases only
- **Model capabilities declared in config.json** — use `returns_confidence: true/false` and `is_zero_bit: true/false` for general dispatch. For detection reliability, models must implement `is_watermarked(detect_output) -> bool` in `model.py`
- **Native attacks** need only `attack.py` + `config.json` in their directory
- **Dockerized attacks/models** additionally need `app.py`, `Dockerfile`, `requirements.txt`
- **Use `logger` not `print()`** for all output. Use `logging.getLogger(__name__)` (never overwrite the `logging` module)
- **uvicorn startup**: In `app.py` files, use `uvicorn.run(app, host=host, port=app_port)` — never `{host}` (creates a set)

## Common Gotchas

- Plugin loading imports ALL plugins at startup. If a dependency is missing (e.g., `pywt`, `audiocomplib`), that plugin silently fails to load
- The `.env` file is gitignored; copy `.env.example` to `.env` for local development
- Accuracy values are percentages (0-100), NOT decimals (0-1). All thresholds and comparisons must use percentage scale
- `CrossModelAttack.apply()` returns a tuple `(audio, watermark)`, not just audio — handled specially in benchmark.py
- Perth is a zero-bit model (detect returns a scalar, not a bit array)
- AudioSeal and AWARE return `(watermark, confidence)` from detect; others return just the watermark
