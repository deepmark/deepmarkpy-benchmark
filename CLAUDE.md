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
- `src/deepmarkpy/config.py` — Config schema, validator and `ModeConfig`; its codes are listed under [Validation codes](#validation-codes). `load_configs(paths)` reads files; `load_config_data(mapping, source=...)` runs the same checks over a mapping an embedding application built itself. Both go through `_validate_all`
- `src/deepmarkpy/config_templates/*.json` — The commented templates `--init` prints; a test asserts the benchmark and detection_reliability ones reproduce the `ATTACK_GROUPS` metric matrix
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

The test count moves whenever tests are added; `pytest tests/ --collect-only -q | tail -1` is authoritative. No Docker required for tests: `test_cli_end_to_end.py` drives `main()` against a throwaway in-process model plugin, so the CLI wiring is covered without any container. Most report tests assert on the `.tex` and stub out `compile_latex` with the `no_pdflatex` fixture; `test_report_generator.py`'s full-report tests run pdflatex where it is installed. Chart drawing is *not* stubbed — `test_report_figures.py` writes real PNGs and checks that every `\includegraphics` in a report points at a file that exists. `pywt` and `pyrubberband` are declared dependencies, so the full attack set loads and `test_attack_groups.py` is expected to pass. Golden replay tests (`test_native_goldens.py`) require byte-identical output where the machine and package versions match `tests/fixtures/goldens/manifest.json` (Darwin arm64; numpy 2.2.6, scipy 1.16.0, librosa 0.11.0, soundfile 0.13.1, audiocomplib 0.2.0), and agreement within `CROSS_ENV_RTOL` (1e-6 relative) anywhere else; they skip only when ffmpeg is missing (`Mp3CompressionAttack`) or an attack is not discovered.

## Running the Benchmark

```bash
# Start the model's service (the template's model is AudioSealModel)
docker-compose up -d audioseal
# Create a config (one per mode)
deepmark-benchmark --init benchmark > my_config.json
# Edit it to select attacks that run natively: "attacks": {"groups": ["audio_distortion"]}
# Check it, then run it
deepmark-benchmark --config my_config.json --validate-only
deepmark-benchmark --config my_config.json --wav_files_dir /path/to/wavs
```

The template's empty attack selection runs every discovered attack, which needs every service (`docker-compose up -d`) and the AIR and music datasets. Only the selected models' services are probed before a run; a stopped attack service fails the run when that attack is reached.

One config file per mode (`benchmark`, `no_attacks`, `detection_reliability`) holds every measurement setting; apart from the 2.x flags, CLI flags are operational. Without `--config`, `_configs_from_flags` assembles the 2.x flags into a mapping validated by `load_config_data`; with it, the `_LEGACY_FLAGS` are ignored with a warning and a per-parameter flag is an error.

## Validation codes

`config.py` emits every code below except W006 and W007, which `run.py` logs before a run. An error stops the run before anything starts; a warning does not, and W001, W003, W004, W011 and W013 are info.

- **Files:** E001 missing or unreadable, E002 not JSON or not UTF-8, E003 top level not an object, E004 no `mode`, E005 unknown mode, E006 two files in one invocation declare the same mode
- **Keys and types:** E007 unknown key (at the top level, in `general`, `attacks`, `metrics`, `efficiency`, `duration_groups` or `comparison`, or in a metric entry), E008 a top-level key another mode uses, E009 wrong type
- **Models:** E010 none listed, E011 unknown model (also as `different_model_name_cross_model`), E012 several in `detection_reliability`, E044 a `detection_reliability` model without `is_watermarked()`
- **Attacks:** E013 unknown group, E014 unknown attack or a group with an undiscovered member, E015 unknown version, E016 listed twice (or a bare name and `:default` both in `attack_parameters`), E017 unknown attack in `attack_parameters`, E018 unknown parameter, E019 parameter type unlike the plugin default, E045 empty or unsupported `bitrate_codec2`
- **Metrics and statistics:** E021 unknown statistic, E022 empty statistics list, E023 statistic listed twice, E024 unknown metric (also under `efficiency.metrics`), E025 `ber` in `detection_reliability`, E026 `accuracy` disabled, E027 unknown `per_group` key, E028 `per_group` in `no_attacks`, E029 statistics on `emr` or `container_footprint`, E043 an efficiency metric under `metrics`, E046 an accuracy statistics list holding only `std`
- **Crop, duration groups, comparison, seed:** E030 crop not a number, E031 crop outside (0, 100), E032 boundary not a positive number, E033 boundaries not ascending, E034 duplicate boundary, E035 primary statistic `std` or not a statistic, E036 primary statistic never computed for accuracy, E037 seed not an integer from 0 to 4294967295
- **Warnings:** W001 a `per_group` section no selected attack reaches, W002 signal-metric enable flags ignored because `calculate_quality_metrics` is off, W003 only some NISQA dimensions on, W004 `calculate_quality_metrics` on with no `metrics` block, W005 `include_overall` without boundaries, W006 NISQA metrics on but the service unavailable, W007 ViSQOL on but `visqol` not installed, W010 a new version given only some parameters (skipped), W011 parameters for an attack or version the run does not select, W012 `metrics` without `defaults`, W013 `efficiency` off with metrics listed, W015 a ranked group's accuracy statistics leave out the primary, W016 accuracy, BER or EMR under an `audio_editing` subsection

## Development Conventions

- **CPU only** — no service requests a GPU and none is expected to. Install
  `+cpu` torch wheels; never pin `nvidia-*`, `triton`, or a `+cuXXX` build.
  The default wheel now bundles CUDA on arm64 too, so an unconstrained
  `pip install torch` silently adds gigabytes that cannot execute.
  `tests/test_cpu_only_builds.py` enforces this across every pin file

- **Attack parameters live in the config file** under `attack_parameters.<AttackName>[:<version>].<param>`, routed per expanded attack entry (`expand_attacks(parameters=...)`), so a bare key targets only the default version. A key naming a version the plugin lacks *defines* that version when every parameter is given and is skipped with `W010` when only some are. Suffix parameter names with the attack (`snr_db_replay`, `order_bandstop`): a 2.x `--<param>` flag reaches every attack that declares the name
- **Never hardcode a metric or statistic list in a report generator, and never infer one by sampling the data.** Ask the `MetricResolver` the run was given, and resolve a section's metrics with `signal_metrics_for_group(group_key)`, not `all_signal_metrics()`, which is the union across every group. `tests/test_report_config_fidelity.py` asserts a generated `.tex`'s columns are exactly what its config asked for
- **Figures never break a report.** There are three: the basic report's accuracy ranking and attack-strength curves (`utils/report_charts.py`) and the comparative report's radar. Each is left out of its report when it cannot be drawn: the `report_charts` functions return `True`/`False` instead of raising, the comparative `generate_full_report` catches a radar failure, and a caller writes the `\includegraphics` only for a chart that was drawn. The basic report's figures read each attack's group's first accuracy statistic other than `std`; the radar reads the statistic the main comparison table ranks, and that group statistic where an attack's group does not compute it. Text a chart takes from the tables' LaTeX labels goes through `report_charts.plain()`
- **Per-group metric relevance is declared in `attack_groups.py`** (`ATTACK_GROUPS` for the six selectable families, `ATTACK_SUBGROUPS` for the four `audio_editing` report subsections). `MetricResolver.from_attack_groups()` turns that into the built-in default matrix, and the shipped benchmark and detection_reliability templates reproduce it — a test fails if they drift
- **Efficiency metrics are configured in their own `efficiency` section, not in `metrics`**, in all three modes and off by default. They describe the machine, not the watermarking method, and do not reproduce: they get their own tables, and nothing in `tests/fixtures/` or the goldens may assert on them. `container_footprint` adds a **Container Memory** section to every report but the comparative one, read from the `docker` CLI: whole running containers, each model's report listing only its own model. `detection_reliability` does not time its calls on clean audio, so each latency times the calls the other modes time
- **Model capabilities declared in config.json** — use `returns_confidence: true/false` and `is_zero_bit: true/false` for general dispatch. Whether a watermark was *found* is answered only by `is_watermarked(detect_output) -> bool` in `model.py`: `detection_reliability` refuses a model without it (`E044`), and the `no_attacks` report fills its "Detected" column only for models that implement it. `core/base_model.implements_is_watermarked()` is the one predicate both use. AudioSeal, AWARE and Perth implement it; SilentCipher, TimbreWM and WavMark do not
- **Native attacks** need only `attack.py` + `config.json` in their directory
- **Dockerized attacks/models** additionally need `app.py`, `Dockerfile`, `requirements.txt`
- **Use `logger` not `print()`** for all output, with one exception where a stream is the interface: `--init` writes its template to stdout, and `_report_config_error` writes config errors to stderr. Use `logging.getLogger(__name__)` (never overwrite the `logging` module)
- **uvicorn startup**: In `app.py` files, use `uvicorn.run(app, host=host, port=app_port)` — never `{host}` (creates a set)

## Common Gotchas

- Plugin loading imports ALL plugins at startup. If a dependency is missing (e.g., `pycodec2`, `audiocomplib`), that plugin does not load; the failure is recorded in `PluginManager.failed`, and naming that attack, or a group that declares it, fails validation (`E014`) rather than measuring a smaller set. `run_metadata.json` carries the same list, but only `run_single_model` writes it — the `no_attacks` and `detection_reliability` modes produce no metadata file
- With `calculate_quality_metrics` false or absent, the signal metrics' enable flags are ignored and PESQ/ViSQOL/STOI are the only signal metrics computed, for every group; `ber`/`emr` keep their flags and per-metric `statistics` still apply. `MetricResolver` implements this, so no generator has to
- An attack renders under its group in the basic/detection-reliability reports and under its `audio_editing` subgroup in the detailed report. `MetricResolver.metrics_for_attack` computes the **union** of the two, so neither section ends up with a hole — meaning a subgroup exclusion narrows the tables but does not always save compute
- `.env` is untracked; `.env.example` is the template and a fresh clone needs `cp .env.example .env`. `run.py` loads it into the environment before plugins are constructed, so a port set there reaches the host clients as well as Compose; real environment variables still win. `HOST` is uvicorn's bind address *inside* each container and must stay `0.0.0.0` — the loopback restriction is the publish side in `docker-compose.yml`
- Accuracy values are percentages (0-100), NOT decimals (0-1). All thresholds and comparisons must use percentage scale
- `CrossModelAttack.apply()` returns a tuple `(audio, watermark)`, not just audio — handled specially in benchmark.py
- Perth is a zero-bit model (detect returns a scalar, not a bit array)
- AudioSeal and AWARE return `(watermark, confidence)` from detect; others return just the watermark
