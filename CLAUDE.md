# DeepMarkPy Benchmark — Development Guide

## What This Project Is

Open-source benchmarking framework for evaluating audio watermarking robustness. Evaluates watermarking models against 40+ attacks (signal processing, AI-based, transmission). Published in IEEE Access, vol. 14, 2026, pp. 62031-62044 (DOI 10.1109/ACCESS.2026.3685903).

**Behavior freeze:** the v1.0.0 release line preserves benchmark behavior bit-for-bit. Known behavior quirks are catalogued internally and frozen until a dedicated deferred-fix release — do not fix surprising-but-working behavior in passing. Discovery sets are locked by `tests/test_discovery_lock.py`; native-attack goldens and HTTP-contract fixtures (`tests/fixtures/`) gate any change to plugin behavior.

## Architecture

- **Plugin-based**: Models and attacks auto-discovered from `src/deepmarkpy/plugins/models/` and `src/deepmarkpy/plugins/attacks/` via `PluginManager`
- **Client-server**: Complex ML models/attacks run in Docker containers, accessed via HTTP (FastAPI). Simple attacks run natively
- **Base classes**: `BaseModel` (embed/detect) in `src/deepmarkpy/core/base_model.py`, `BaseAttack` (apply) in `src/deepmarkpy/core/base_attack.py`
- **Config-driven**: Each plugin has a `config.json` with defaults. Model configs include `returns_confidence` and `is_zero_bit` flags. Detection reliability uses `is_watermarked()` on the model class

## Key Files

- `src/deepmarkpy/run.py` — CLI entrypoint (`deepmark-benchmark` console script; `src/run.py` is a deprecation shim)
- `src/deepmarkpy/benchmark.py` — Core benchmark orchestration (run loop, accuracy computation)
- `src/deepmarkpy/plugin_manager.py` — Auto-discovers plugins by walking directories
- `src/deepmarkpy/utils/metrics.py` — PESQ, STOI, PSNR, SI-SDR
- `src/deepmarkpy/utils/report_generator.py` — LaTeX + chart generation
- `docker-compose.yml` — All containerized services
- `.env.example` — Port configuration template

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests are in `tests/` and use `conftest.py` for shared fixtures (sample audio, watermarks, result dicts). Tests import the installed `deepmarkpy` package (`pip install -e .`).

Current: 438 tests as of `bb8b6c7`, ~5s runtime (the count moves whenever tests are added — `pytest tests/ --collect-only -q | tail -1` is authoritative). No Docker required for tests. `pywt` and `pyrubberband` are declared dependencies, so the full attack set loads and `test_attack_groups.py` is expected to pass. Golden replay tests (`test_native_goldens.py`) enforce only where the numeric environment matches their manifest and skip elsewhere, so around 29 of them skip outside the recording environment (numpy 2.2.6 / scipy 1.16.0 / librosa 0.11.0).

## Running the Benchmark

```bash
# Start Docker services (if using containerized models/attacks)
docker-compose up -d audioseal
# Run benchmark
deepmark-benchmark --wav_files_dir /path/to/wavs --wm_model AudioSealModel --attack_types GaussianNoiseAttack
```

## Development Conventions

- **CPU only** — no service requests a GPU and none is expected to. Install
  `+cpu` torch wheels; never pin `nvidia-*`, `triton`, or a `+cuXXX` build.
  The default wheel now bundles CUDA on arm64 too, so an unconstrained
  `pip install torch` silently adds gigabytes that cannot execute.
  `tests/test_cpu_only_builds.py` enforces this across every pin file

- **Attack parameter names must be unique across all attacks** — they share a flat CLI namespace. Suffix with attack name (e.g., `snr_db_replay`, `order_bandstop`). The system warns on collisions but doesn't prevent them
- **Model capabilities declared in config.json** — use `returns_confidence: true/false` and `is_zero_bit: true/false` for general dispatch. For detection reliability, models must implement `is_watermarked(detect_output) -> bool` in `model.py`
- **Native attacks** need only `attack.py` + `config.json` in their directory
- **Dockerized attacks/models** additionally need `app.py`, `Dockerfile`, `requirements.txt`
- **Use `logger` not `print()`** for all output. Use `logging.getLogger(__name__)` (never overwrite the `logging` module)
- **uvicorn startup**: In `app.py` files, use `uvicorn.run(app, host=host, port=app_port)` — never `{host}` (creates a set)

## Common Gotchas

- Plugin loading imports ALL plugins at startup. If a dependency is missing (e.g., `pycodec2`, `audiocomplib`), that plugin does not load; the failure is recorded in `PluginManager.failed`, and asking for that attack by name or via `--attack_groups` raises rather than measuring a smaller set. `run_metadata.json` carries the same list, but only `run_single_model` writes it — `--no_attacks` and `--detection_reliability` produce no metadata file
- `.env` is tracked in git even though `.gitignore` lists it (the entry never untracked it) — local port edits show up as modifications; `.env.example` mirrors it. `run.py` loads it into the environment before plugins are constructed, so a port set there reaches the host clients as well as Compose; real environment variables still win
- Accuracy values are percentages (0-100), NOT decimals (0-1). All thresholds and comparisons must use percentage scale
- `CrossModelAttack.apply()` returns a tuple `(audio, watermark)`, not just audio — handled specially in benchmark.py
- Perth is a zero-bit model (detect returns a scalar, not a bit array)
- AudioSeal and AWARE return `(watermark, confidence)` from detect; others return just the watermark
