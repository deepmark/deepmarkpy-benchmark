# DeepMarkPy Benchmark — Development Guide

## What This Project Is

Open-source benchmarking framework for evaluating audio watermarking robustness. Evaluates watermarking models against 40+ attacks (signal processing, AI-based, transmission). Published in IEEE Access, vol. 14, 2026, pp. 62031-62044 (DOI 10.1109/ACCESS.2026.3685903).

**Behavior:** v1.x preserved pre-package behavior bit-for-bit. **v2.0.0 is the deferred-fix release** and deliberately changes attack outputs — four attacks were corrected (equalizer, bandstop, PCM/MP3 rounding) and four more became available once `pywt`/`pyrubberband` were declared. Results are not comparable across the v1/v2 boundary.

Within v2.x the same discipline applies: discovery sets are locked by `tests/test_discovery_lock.py`, and native-attack goldens plus HTTP-contract fixtures (`tests/fixtures/`) gate any change to plugin behavior. Do not change a golden without deciding to. A handful of quirks remain deliberately frozen — SilentCipher's sampling-rate and eval-mode issues both move published numbers and are catalogued internally for a later release.

## Architecture

- **Plugin-based**: Models and attacks auto-discovered from `src/deepmarkpy/plugins/models/` and `src/deepmarkpy/plugins/attacks/` via `PluginManager`
- **Client-server**: Complex ML models/attacks run in Docker containers, accessed via HTTP (FastAPI). Simple attacks run natively
- **Base classes**: `BaseModel` (embed/detect) in `src/deepmarkpy/core/base_model.py`, `BaseAttack` (apply) in `src/deepmarkpy/core/base_attack.py`
- **Config-driven**: Each plugin has a `config.json` with defaults. Model configs include `returns_confidence` and `is_zero_bit` flags. Detection reliability uses `is_watermarked()` on the model class

## Key Files

- `src/deepmarkpy/run.py` — CLI entrypoint (`deepmark-benchmark` console script; `src/run.py` is a deprecation shim). Operational flags, plus the pre-2.0 measurement flags routed through `_configs_from_flags`; mode dispatch lives in `_MODE_RUNNERS`
- `src/deepmarkpy/config.py` — Config schema, validator (error catalog E001–E043, warnings W001–W005, W010–W013), and `ModeConfig`. `load_configs(paths)` reads files; `load_config_data(mapping, source=...)` runs the identical checks over a mapping an embedding application built itself, so it can show the same codes before writing anything. Both go through `_validate_all`, so the two paths cannot drift
- `src/deepmarkpy/config_templates/*.json` — The three commented templates `--init` prints, and the canonical source: a test asserts they reproduce the `ATTACK_GROUPS` metric matrix. `configs/` at the repo root holds *working* files started from them and is expected to diverge; a test only checks those still validate
- `src/deepmarkpy/utils/metric_resolver.py` — `MetricResolver`: the single answer to "which metrics and which statistics for this attack group". Every report generator asks it instead of carrying constants
- `src/deepmarkpy/benchmark.py` — Core benchmark orchestration (run loop, accuracy computation)
- `src/deepmarkpy/plugin_manager.py` — Auto-discovers plugins by walking directories
- `src/deepmarkpy/utils/metrics.py` — PESQ, STOI, PSNR, SI-SDR
- `src/deepmarkpy/utils/efficiency.py` — the efficiency family: what a run cost in time. Measured in the run loop, never by `compute_metrics`, and every line it logs carries `[efficiency]` so timing output can be read or filtered on its own
- `src/deepmarkpy/utils/report_generator.py` — LaTeX + chart generation
- `src/deepmarkpy/utils/report_charts.py` — every figure in every report: shared palette, the four robustness tiers, and one function per chart. Each report gets different figures, because none of them holds the same data
- `docker-compose.yml` — All containerized services
- `.env.example` — Port configuration template

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests are in `tests/` and use `conftest.py` for shared fixtures (sample audio, watermarks, result dicts). Tests import the installed `deepmarkpy` package (`pip install -e .`).

Current: 900 tests as of the flag-compatibility work, ~80s runtime (the count moves whenever tests are added — `pytest tests/ --collect-only -q | tail -1` is authoritative). No Docker required for tests: `test_cli_end_to_end.py` drives `main()` against a throwaway in-process model plugin, so the CLI wiring is covered without any container. Report tests stub out `compile_latex`; they assert on the `.tex`, not on pdflatex. Chart drawing is *not* stubbed — `test_report_figures.py` writes real PNGs and checks that every `\includegraphics` in a report points at a file that exists. `pywt` and `pyrubberband` are declared dependencies, so the full attack set loads and `test_attack_groups.py` is expected to pass. Golden replay tests (`test_native_goldens.py`) enforce only where the numeric environment matches their manifest and skip elsewhere, so around 29 of them skip outside the recording environment (numpy 2.2.6 / scipy 1.16.0 / librosa 0.11.0).

## Running the Benchmark

```bash
# Start Docker services (if using containerized models/attacks)
docker-compose up -d audioseal
# Create a config (one per mode), then run it
deepmark-benchmark --init benchmark > configs/benchmark.json
deepmark-benchmark --config configs/benchmark.json --validate-only
deepmark-benchmark --config configs/benchmark.json --wav_files_dir /path/to/wavs
```

**Configuration is the interface.** One file per mode (`benchmark`,
`no_attacks`, `detection_reliability`), each declaring its own `"mode"` and
accepting only that mode's keys. The CLI's own settings are operational
(`--config`, `--init`, `--validate-only`, `--wav_files_dir`, `--report_dir`,
`--seed`, `--verbose`, `--save_audio`, `--plugins_dir`); everything
measurement-related lives in the file. Pass several `--config` files to run
several modes in one invocation — each must declare a different mode.

**The pre-2.0 flags still run, for scripts written against the previous
release.** `--wm_model(s)`, `--attack_types`, `--attack_groups`,
`--no_attacks`, `--detection_reliability`, `--calculate_quality_metrics`,
`--crop_before_attack` and the per-parameter flags the plugins declare
(`--snr_db_gaussian_noise`, `--no-bandpass_replay`) are accepted when
`--config` is absent. `_configs_from_flags` assembles them into a config
mapping and hands it to `load_config_data`, so they go through the same
validator, the same `MetricResolver` and the same run loop — there is no
second code path and no second set of defaults. `--config` wins when both
are given, and every ignored flag is named in a warning. What the flags
cannot express is what they never could: per-group metrics and statistics,
`efficiency`, `duration_groups`, `comparison`, and per-version attack
parameters (a bare parameter flag reaches the default preset only).

## Development Conventions

- **CPU only** — no service requests a GPU and none is expected to. Install
  `+cpu` torch wheels; never pin `nvidia-*`, `triton`, or a `+cuXXX` build.
  The default wheel now bundles CUDA on arm64 too, so an unconstrained
  `pip install torch` silently adds gigabytes that cannot execute.
  `tests/test_cpu_only_builds.py` enforces this across every pin file

- **Attack parameters live in the config file** under `attack_parameters.<AttackName>[:<version>].<param>`. They are routed per expanded attack entry (via `expand_attacks(parameters=...)`), not flattened into one kwargs mapping, so a bare key targets only the default version and two attacks may share a parameter name. A key naming a version the plugin lacks *defines* that version when every parameter is given, and is skipped with `W010` when only some are — a partly-specified version would inherit the rest from the default preset and be a mislabelled default. Suffixing parameter names with the attack (`snr_db_replay`, `order_bandstop`) remains the convention but is no longer load-bearing
- **Never hardcode a metric or statistic list in a report generator, and never infer one by sampling the data.** Ask the `MetricResolver` the run was given; `tests/test_report_config_fidelity.py` asserts a generated `.tex`'s columns are exactly what its config asked for
- **Figures obey the same rule as tables, and never break a report.** All plotting lives in `utils/report_charts.py`; a generator passes it values it already resolved through the `MetricResolver`, so a chart cannot plot a metric the tables are forbidden to show — `tests/test_figure_metric_fidelity.py` asserts that across every report, including under `calculate_quality_metrics: false`, where only the always-on trio is computable whatever the enable flags say. Resolve a section's metrics with `signal_metrics_for_group(group_key)`, never `all_signal_metrics()`: the latter is the union across every group and will name a metric the section's own tables leave out. Every chart function returns `True`/`False` instead of raising, and the caller omits the `\includegraphics` when it returns `False` — a figure is never referenced unless it was written. Chart text goes through `report_charts.plain()`, because the labels are the LaTeX ones the tables use
- **Per-group metric relevance is declared in `attack_groups.py`** (`ATTACK_GROUPS` for the six selectable families, `ATTACK_SUBGROUPS` for the four `audio_editing` report subsections). `MetricResolver.from_attack_groups()` turns that into the built-in default matrix, and the shipped templates reproduce it — a test fails if they drift
- **Efficiency metrics are configured in their own `efficiency` section, not in `metrics`.** `{"enabled": bool, "metrics": {...}}`, present in all three modes and off by default. `container_footprint` is the odd one: it is not a metric on any table but a gate. Set it true and every report gains a **Container Memory** section listing the models, dockerized attacks and metric services the run used, read once from the `docker` CLI. It reports the whole container rather than the model, and only running containers appear -- native plugins and stopped services are absent rather than zero. Off by default, because reading it shells out to docker; `enabled: false` means no timing is taken at all. They are the one family that does **not** reproduce — they describe the machine, not the watermarking method — so they get their own table below the quality ones, never a column beside them, and nothing in `tests/fixtures/` or the goldens may assert on them. All three modes measure. `detection_reliability` has its own run loop and calls detect twice per file and twice per attack (clean for the false-positive rate, watermarked for the false negative); only the calls matching what the other modes time are recorded, so a metric means the same thing everywhere
- **Model capabilities declared in config.json** — use `returns_confidence: true/false` and `is_zero_bit: true/false` for general dispatch. Whether a watermark was *found* is a separate question, answered only by `is_watermarked(detect_output) -> bool` in `model.py`: `detection_reliability` refuses to run without it, and the `no_attacks` report drops its "Detected" column rather than thresholding `detect()` output itself. `core/base_model.implements_is_watermarked()` is the one predicate both use. AudioSeal, AWARE and Perth implement it; SilentCipher, TimbreWM and WavMark do not
- **Native attacks** need only `attack.py` + `config.json` in their directory
- **Dockerized attacks/models** additionally need `app.py`, `Dockerfile`, `requirements.txt`
- **Use `logger` not `print()`** for all output. Use `logging.getLogger(__name__)` (never overwrite the `logging` module)
- **uvicorn startup**: In `app.py` files, use `uvicorn.run(app, host=host, port=app_port)` — never `{host}` (creates a set)

## Common Gotchas

- Plugin loading imports ALL plugins at startup. If a dependency is missing (e.g., `pycodec2`, `audiocomplib`), that plugin does not load; the failure is recorded in `PluginManager.failed`, and asking for that attack by name or via `attacks.groups` raises rather than measuring a smaller set. `run_metadata.json` carries the same list, but only `run_single_model` writes it — the `no_attacks` and `detection_reliability` modes produce no metadata file
- `calculate_quality_metrics` in the config file is a real override, and the only one left: when it is `false` or absent, only accuracy and the always-on trio (PESQ/ViSQOL/STOI) are computed and the `metrics` enable flags are ignored. `ber`/`emr` still honour their flags (both derive from accuracy for free), and per-metric `statistics` still apply. `MetricResolver` implements this, so no generator has to
- An attack renders under its group in the basic/detection-reliability reports and under its `audio_editing` subgroup in the detailed report. `MetricResolver.metrics_for_attack` computes the **union** of the two, so neither section ends up with a hole — meaning a subgroup exclusion narrows the tables but does not always save compute
- `.env` is untracked; `.env.example` is the template and a fresh clone needs `cp .env.example .env`. `run.py` loads it into the environment before plugins are constructed, so a port set there reaches the host clients as well as Compose; real environment variables still win. `HOST` is uvicorn's bind address *inside* each container and must stay `0.0.0.0` — the loopback restriction is the publish side in `docker-compose.yml`
- Accuracy values are percentages (0-100), NOT decimals (0-1). All thresholds and comparisons must use percentage scale
- `CrossModelAttack.apply()` returns a tuple `(audio, watermark)`, not just audio — handled specially in benchmark.py
- Perth is a zero-bit model (detect returns a scalar, not a bit array)
- AudioSeal and AWARE return `(watermark, confidence)` from detect; others return just the watermark
