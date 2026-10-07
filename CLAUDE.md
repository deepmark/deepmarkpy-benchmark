# DeepMarkPy Benchmark — Development Guide

## What This Project Is

Open-source benchmarking framework for evaluating audio watermarking robustness. Evaluates watermarking models against 40+ attacks (signal processing, AI-based, transmission). Published in IEEE Access, vol. 14, 2026, pp. 62031-62044 (DOI 10.1109/ACCESS.2026.3685903).

**Behavior:** v1.x preserved pre-package behavior bit-for-bit. **v2.0.0 is the deferred-fix release** and deliberately changes attack outputs — four attacks were corrected (equalizer, bandstop, PCM/MP3 rounding) and four more became available once `pywt`/`pyrubberband` were declared. Results are not comparable across the v1/v2 boundary.

Since v2.0.0 the same discipline applies: discovery sets are locked by `tests/test_discovery_lock.py`, and native-attack goldens plus HTTP-contract fixtures (`tests/fixtures/`) gate any change to plugin behavior. Do not change a golden without deciding to. A handful of quirks remain deliberately frozen — SilentCipher's sampling-rate and eval-mode issues both move published numbers and are catalogued internally for a later release.

## Architecture

- **Plugin-based**: Models and attacks auto-discovered from `src/deepmarkpy/plugins/models/` and `src/deepmarkpy/plugins/attacks/` via `PluginManager`
- **Client-server**: Complex ML models/attacks run in Docker containers, accessed via HTTP (FastAPI). Simple attacks run natively
- **Base classes**: `BaseModel` (embed/detect) in `src/deepmarkpy/core/base_model.py`, `BaseAttack` (apply) in `src/deepmarkpy/core/base_attack.py`
- **Config-driven**: Each plugin has a `config.json` with defaults. Model configs include `returns_confidence` and `is_zero_bit` flags. Detection reliability uses `is_watermarked()` on the model class

## Key Files

- `src/deepmarkpy/run.py` — CLI entrypoint (`deepmark-benchmark` console script; `src/run.py` is a deprecation shim). Operational flags, plus the 2.x measurement flags routed through `_configs_from_flags`; mode dispatch lives in `_MODE_RUNNERS`
- `src/deepmarkpy/config.py` — Config schema, validator and `ModeConfig`. Errors E001–E019, E021–E037 and E043–E046; warnings W001–W005, W010–W013 and W015–W016 (`run.py` adds W006 and W007). `load_configs(paths)` reads files; `load_config_data(mapping, source=...)` runs the same checks over a mapping an embedding application built itself. Both go through `_validate_all`
- `src/deepmarkpy/config_templates/*.json` — The commented templates `--init` prints; a test asserts they reproduce the `ATTACK_GROUPS` metric matrix
- `src/deepmarkpy/utils/metric_resolver.py` — `MetricResolver`: which metrics and statistics each attack group gets. Every report generator asks it instead of carrying constants
- `src/deepmarkpy/benchmark.py` — Core benchmark orchestration (run loop, accuracy computation)
- `src/deepmarkpy/plugin_manager.py` — Auto-discovers plugins by walking directories
- `src/deepmarkpy/utils/metrics.py` — PESQ, STOI, PSNR, SI-SDR
- `src/deepmarkpy/utils/efficiency.py` — latency timing and container memory; every line it logs carries `[efficiency]`
- `src/deepmarkpy/utils/report_generator.py` — LaTeX + chart generation
- `src/deepmarkpy/utils/report_charts.py` — the basic report's accuracy ranking and attack-strength curves. The comparative report draws its radar itself; the detailed, no_attacks and detection_reliability reports are tables only
- `docker-compose.yml` — All containerized services
- `.env.example` — Port configuration template

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests are in `tests/` and use `conftest.py` for shared fixtures (sample audio, watermarks, result dicts). Tests import the installed `deepmarkpy` package (`pip install -e .`).

The test count moves whenever tests are added; `pytest tests/ --collect-only -q | tail -1` is authoritative. No Docker required for tests: `test_cli_end_to_end.py` drives `main()` against a throwaway in-process model plugin, so the CLI wiring is covered without any container. Report tests stub out `compile_latex`; they assert on the `.tex`, not on pdflatex. Chart drawing is *not* stubbed — `test_report_figures.py` writes real PNGs and checks that every `\includegraphics` in a report points at a file that exists. `pywt` and `pyrubberband` are declared dependencies, so the full attack set loads and `test_attack_groups.py` is expected to pass. Golden replay tests (`test_native_goldens.py`) enforce only where the numeric environment matches their manifest and skip elsewhere, so around 29 of them skip outside the recording environment (numpy 2.2.6 / scipy 1.16.0 / librosa 0.11.0).

## Running the Benchmark

```bash
# Start Docker services (if using containerized models/attacks)
docker-compose up -d audioseal
# Create a config (one per mode), check it, then run it
deepmark-benchmark --init benchmark > my_config.json
deepmark-benchmark --config my_config.json --validate-only
deepmark-benchmark --config my_config.json --wav_files_dir /path/to/wavs
```

One config file per mode (`benchmark`, `no_attacks`, `detection_reliability`) holds every measurement setting; CLI flags are operational. Without `--config`, `_configs_from_flags` assembles the 2.x flags into a mapping validated by `load_config_data`; with it, the `_LEGACY_FLAGS` are ignored with a warning and a per-parameter flag is an error.

## Development Conventions

- **CPU only** — no service requests a GPU and none is expected to. Install
  `+cpu` torch wheels; never pin `nvidia-*`, `triton`, or a `+cuXXX` build.
  The default wheel now bundles CUDA on arm64 too, so an unconstrained
  `pip install torch` silently adds gigabytes that cannot execute.
  `tests/test_cpu_only_builds.py` enforces this across every pin file

- **Attack parameters live in the config file** under `attack_parameters.<AttackName>[:<version>].<param>`, routed per expanded attack entry (`expand_attacks(parameters=...)`), so a bare key targets only the default version. A key naming a version the plugin lacks *defines* that version when every parameter is given and is skipped with `W010` when only some are. Suffix parameter names with the attack (`snr_db_replay`, `order_bandstop`): a 2.x `--<param>` flag reaches every attack that declares the name
- **Never hardcode a metric or statistic list in a report generator, and never infer one by sampling the data.** Ask the `MetricResolver` the run was given, and resolve a section's metrics with `signal_metrics_for_group(group_key)`, not `all_signal_metrics()`, which is the union across every group. `tests/test_report_config_fidelity.py` asserts a generated `.tex`'s columns are exactly what its config asked for
- **Figures never break a report.** There are three: the basic report's accuracy ranking and attack-strength curves (`utils/report_charts.py`) and the comparative report's radar. They plot accuracy at the statistic the config resolves, as the tables do. Each chart function in `report_charts` returns `True`/`False` instead of raising, and a caller omits the `\includegraphics` when a chart returns `False`. Chart text goes through `report_charts.plain()`, because the labels are the LaTeX ones the tables use
- **Per-group metric relevance is declared in `attack_groups.py`** (`ATTACK_GROUPS` for the six selectable families, `ATTACK_SUBGROUPS` for the four `audio_editing` report subsections). `MetricResolver.from_attack_groups()` turns that into the built-in default matrix, and the shipped templates reproduce it — a test fails if they drift
- **Efficiency metrics are configured in their own `efficiency` section, not in `metrics`**, in all three modes and off by default. They describe the machine, not the watermarking method, and do not reproduce: they get their own tables, and nothing in `tests/fixtures/` or the goldens may assert on them. `container_footprint` adds a **Container Memory** section to every report but the comparative one, read from the `docker` CLI: whole running containers, each model's report listing only its own model. `detection_reliability` times only the attack and detect calls on watermarked audio, the ones the other modes time
- **Model capabilities declared in config.json** — use `returns_confidence: true/false` and `is_zero_bit: true/false` for general dispatch. Whether a watermark was *found* is answered only by `is_watermarked(detect_output) -> bool` in `model.py`: `detection_reliability` refuses a model without it (`E044`), and the `no_attacks` report fills its "Detected" column only for models that implement it. `core/base_model.implements_is_watermarked()` is the one predicate both use. AudioSeal, AWARE and Perth implement it; SilentCipher, TimbreWM and WavMark do not
- **Native attacks** need only `attack.py` + `config.json` in their directory
- **Dockerized attacks/models** additionally need `app.py`, `Dockerfile`, `requirements.txt`
- **Use `logger` not `print()`** for all output. Use `logging.getLogger(__name__)` (never overwrite the `logging` module)
- **uvicorn startup**: In `app.py` files, use `uvicorn.run(app, host=host, port=app_port)` — never `{host}` (creates a set)

## Common Gotchas

- Plugin loading imports ALL plugins at startup. If a dependency is missing (e.g., `pycodec2`, `audiocomplib`), that plugin does not load; the failure is recorded in `PluginManager.failed`, and naming that attack, or a group that declares it, fails validation (`E014`) rather than measuring a smaller set. `run_metadata.json` carries the same list, but only `run_single_model` writes it — the `no_attacks` and `detection_reliability` modes produce no metadata file
- With `calculate_quality_metrics` false or absent, only accuracy and PESQ/ViSQOL/STOI are computed and the `metrics` enable flags are ignored; `ber`/`emr` keep their flags and per-metric `statistics` still apply. `MetricResolver` implements this, so no generator has to
- An attack renders under its group in the basic/detection-reliability reports and under its `audio_editing` subgroup in the detailed report. `MetricResolver.metrics_for_attack` computes the **union** of the two, so neither section ends up with a hole — meaning a subgroup exclusion narrows the tables but does not always save compute
- `.env` is untracked; `.env.example` is the template and a fresh clone needs `cp .env.example .env`. `run.py` loads it into the environment before plugins are constructed, so a port set there reaches the host clients as well as Compose; real environment variables still win. `HOST` is uvicorn's bind address *inside* each container and must stay `0.0.0.0` — the loopback restriction is the publish side in `docker-compose.yml`
- Accuracy values are percentages (0-100), NOT decimals (0-1). All thresholds and comparisons must use percentage scale
- `CrossModelAttack.apply()` returns a tuple `(audio, watermark)`, not just audio — handled specially in benchmark.py
- Perth is a zero-bit model (detect returns a scalar, not a bit array)
- AudioSeal and AWARE return `(watermark, confidence)` from detect; others return just the watermark
