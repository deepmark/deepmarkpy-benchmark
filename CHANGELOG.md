# Changelog

## Unreleased

**A model that decides whether a watermark is present is scored on that
decision in benchmark mode, now the only mode, and eight attack defaults
change.** Results for the changed attacks, and for AudioSeal, AWARE and Perth
on every attack, are not comparable with earlier runs.

### Added

- `returns_is_watermarked` in a model's `config.json`. A declaring model is
  scored on its own `is_watermarked()`: benchmark mode also applies every
  attack to the clean file and reports true and false positives per attack
  (`tp_count`/`tp_n`, `fp_count`/`fp_n`), with accuracy and BER over the
  detected files only and EMR as the files detected with every bit right,
  over all files. Both executors run it, in the same order. AudioSeal,
  AWARE and Perth declare it; SilentCipher, TimbreWM and WavMark are measured
  as before. The basic and detailed reports give TP and FP in a table of
  their own after each accuracy table; the comparative report gives TP and
  FP rate tables.
- The `NoAttack` attack: the audio unchanged, the no-attack baseline as a row
  of benchmark mode. It belongs to no group; reports label it `No attack`.
- `E047`: a model declaring `returns_is_watermarked` without implementing
  `is_watermarked()`, or a zero-bit model not declaring it.
- `--init default` and `--init default_sweep`: ready-made benchmark runs, the
  default attack set at plugin defaults, and the same set with strength sweeps
  of `BandstopFilterAttack`, `GaussianNoiseAttack`, `HighpassFilterAttack`,
  `InvertedTimeStretchAttack`, `Mp3CompressionAttack`, `ResamplingPolyAttack`
  and `TimeStretchAttack`. `--init` now takes a preset, and
  `src/deepmarkpy/config_templates/` is renamed `src/deepmarkpy/presets/`.

### Changed

- `ResamplingPolyAttack` resamples to `target_sr_resampling_poly` (default
  8000 Hz) and back to the audio's own rate, in place of decimating by the
  integer `down_factor_resampling_poly`, so any target rate can be swept. Its
  output now has the input's length. For a 16 kHz model the default is the
  same 8 kHz round trip as before, through a slightly different anti-alias
  filter; for a 44.1 kHz model it was a 22.05 kHz round trip.

- Default attack parameters: `GaussianNoiseAttack` 25 dB (was 35),
  `PinkNoiseAttack` 35 dB (was 25), `BandstopFilterAttack` 1000-1200 Hz (was
  350-500), `Mp3CompressionAttack` quality 9 (was 3), `SpeechEnhancement1Attack`
  noise 0 (was 0.01), `TimeStretchAttack` rate 1.25 (was 1.4),
  `InvertedTimeStretchAttack` rate 1.4 (was 0.8), and `LowpassFilterAttack`
  3400 Hz at every sampling rate (was 3500 at 16 kHz up to 6000 at 44.1 and
  48 kHz), now set by a single `cutoff_freq_lowpass` in place of the
  per-sampling-rate `cutoff_freq_per_sr_lowpass`, so `attack_parameters`
  versions can sweep it like any other attack. The native goldens were
  re-recorded, on Linux x86_64 (rubberband 4.0.0, ffmpeg 7.0.2): rubberband's
  output differs between Linux and macOS arm64 by more than the
  cross-environment tolerance, so the time-stretch, inverted-time-stretch and
  pitch-shift goldens now hold on Linux and not on macOS, where until now it
  was the other way round.
- A zero-bit model has no accuracy, BER or EMR: TP and FP are its whole
  measurement.

### Fixed

- Plugin discovery registered every attack class a plugin's module held,
  including ones it imports from another plugin. Depending on directory order,
  `TimeStretchAttack` was registered with `InvertedTimeStretchAttack`'s
  `config.json`, and `EqualizerAttack` and `HighpassFilterAttack` with
  `MixingAttack`'s, so their parameters could not be set or swept in
  `attack_parameters` (`E018`) and `run_metadata.json` recorded the wrong
  ones. The attacks themselves ran at their own defaults. Discovery now
  registers only the classes a module defines.

### Removed

- The `no_attacks` mode, with `--no_attacks`, `--init no_attacks`, its report
  and `E028`. The `NoAttack` attack replaces it.
- The `detection_reliability` mode, with `--detection_reliability`, `--init
  detection_reliability`, its report and `E012`, `E025` and `E044`. Benchmark
  mode reports the same true and false positives for a declaring model (a
  false negative is a missing true positive), and `NoAttack` gives them
  without an attack. A config file still naming either removed mode is told
  what replaced it (`E005`).

## v3.0.0 — 2026-10-07

**Measurement settings move from flags into one JSON config file per mode.**
`--config` runs one or more of them, `--init <mode>` prints a commented
template, and `--validate-only` checks a file against the discovered plugins
without running anything. The validator reports every problem together, each
with a stable code and its JSON path. Without `--config` the 2.x flags still
run: they are assembled into a config and validated the same way.

### Added

- Per-group metrics and statistics: each attack group enables its own metrics
  and the statistics they are reported at.
- Per-version attack parameters (`AttackName:version`), including versions a
  config defines itself.
- Duration groups: a report part per audio-length bin.
- An `efficiency` section, off by default: embed, detect and attack latency,
  and the memory of the containers a run used.
- Attack-strength curves in the benchmark report: accuracy across an attack's
  versions.

### Changed

- `--save_audio` with several modes in one invocation, including
  `--no_attacks --detection_reliability`, writes each mode to `audio/<mode>/`.
- The detailed report is written on every run and holds tables only.
- `no_attacks` and the reliability baseline measure PESQ, ViSQOL and STOI
  without `--calculate_quality_metrics`, as benchmark mode always has.
- An unknown CrossModel second model (`E011`) and a `detection_reliability`
  model without `is_watermarked()` (`E044`) fail validation before anything
  runs.

### Breaking

- Removed `ReplacementAttack` and `Replacement2Attack`. No other attack's
  output changes.
- Each `metrics.<metric>` in `detection_reliability.json` is now an object of
  statistics rather than the mean.
- Attack parameter flags (`--snr_db_gaussian_noise`) must be spelled in full;
  2.x accepted an unambiguous prefix.
- Exit codes: `1` when a mode produces no results, `2` for an invalid
  configuration or an unreachable model service (nothing runs). A skipped model
  or a report that fails to render is logged and still exits `0`.
- Internal Python APIs of `deepmarkpy.run`, `deepmarkpy.benchmark`,
  `attack_groups` and the report generators changed. The engine API in
  `docs/CONSUMING.md` did not.
- `tqdm` is no longer a package dependency.

## v2.0.0 — 2026-08-01

**Attack outputs change. Results are not comparable with v1.x.** v1.x preserved
the pre-package behavior bit-for-bit; this release is the deferred-fix release
and corrects it deliberately.

### Behavior changes (why this is a major version)

Four attacks now produce different audio:

- **Equalizer** — filters were applied in parallel rather than cascaded, the
  band width formula inverted Q so every band was twice as wide, output was
  peak-normalized unconditionally, and two of the ten configured bands were
  never built.
- **Bandstop** — designed in transfer-function form, which lost enough
  precision at its 350–500 Hz band to reach ~1e147 output at 48 kHz. Now
  second-order sections.
- **PCM quantization and MP3** — nine float-to-int casts truncated toward zero
  instead of rounding, costing 6 dB of quantization noise and doubling the
  deadzone around silence.
- **Mixing** — passed `gains=` where the equalizer reads `gains_equalizer`, so
  its configured gains were silently ignored.

Four attacks are newly available: `WaveletAttack`, `PitchShiftAttack`,
`TimeStretchAttack` and `InvertedTimeStretchAttack`. `pywt` and `pyrubberband`
were undeclared, so they silently failed to load on a clean install — the
benchmark reported 43 attacks where the paper promised more.

Two more corrections that change results only when they fire: a failing
`InvertedTimeStretch` used to return its input, which scored as the detector's
unattacked ceiling; and `ZeroCrossInserts` dropped the head of every file
longer than one second.

### Transport

Audio crosses the HTTP hop as base64 of the raw float64 buffer rather than a
JSON array of decimals: 20.9 → 10.7 bytes per sample in both directions, with
no decimal formatting or parsing on either end. float64 is deliberate — float32
would halve it again but is lossy, and would move reported numbers. Verified
bit-identical against the v1 fixtures. Services still accept the JSON list form.

### Security

- All 16 services publish on `127.0.0.1` only. They were bound to `0.0.0.0` and
  reachable from the LAN and any VPN interface, unauthenticated, exposing
  `/embed` and `/detect` as a watermark oracle.
- `speech_enhancement_2` validated `model_name` against ClearVoice's own model
  list. It was interpolated into a config path with no check, which made the
  endpoint a filesystem-existence oracle that returned YAML key names in its
  error body.
- Request audio is length-capped; every service had accepted an unbounded body.
- `timbrewm` loads its checkpoint with `weights_only`.

### Reliability

- Five service clients had no timeout and no status check; all ten now match.
- A null `audio` in a 200 response is an error naming the service, instead of
  becoming a 0-d object array that failed later somewhere unrelated.
- A requested attack that is not loadable raises instead of being skipped, and
  `--attack_groups` no longer silently drops attacks whose plugin failed to
  import — nor expands an all-failed group to the entire registry.
- Results are written after every file rather than only at the end.
- `.env` is loaded by the CLI, so a port set there reaches the host clients and
  not just Compose.
- `network_transmission` reports `netem_active`, so a run where the kernel
  impairment never engaged is no longer indistinguishable from one where it did.

### Reporting

- Metrics that cannot express what an attack did are marked and explained
  per-case rather than with one hard-coded sentence, in all three generators.
- `attack_snr_db` is recorded per file, making each attack's real strength
  visible — `additive_noise` sets an absolute amplitude, so its effective SNR
  moves with input level while its SNR-parameterized siblings hold constant.
- `AdditiveNoise`, `Replacement2` and `VAE` were in no attack group and fell
  through group resolution and report sectioning.
- NISQA's availability is recorded in `run_metadata.json`; its default weights
  path was broken by the src-layout move, so the documented setup produced
  nothing.

### Images

18.6 GB → 14.1 GB. `speech_enhancement_2` carried a full CUDA 13.0 stack on a
CPU-only benchmark; `aware` duplicated a 411 MB torch tree for one patch
version; `timbrewm` and `silent_cipher` shipped a torchaudio that could not be
imported. The benchmark is CPU-only by decision, enforced by a test.

### Setup

`.env` is no longer tracked — it duplicated `.env.example` variable for
variable, and being tracked while gitignored made every local port edit show
up as a repo modification. **A fresh clone now needs `cp .env.example .env`
before `docker-compose up`.** `HOST` stays `0.0.0.0`: it is uvicorn's bind
address inside each container, and what keeps the services off the network is
docker-compose publishing every port on `127.0.0.1`.

### Removed

- `docs/MIGRATION.md` (the v1.0.0 packaging migration).
- CI workflow, to be reintroduced in a later release.
- `.env` from version control; `.env.example` is the template.

## v1.1.0 — 2026-07-30

- Engine base classes: `deepmarkpy.core.inference.BaseAttackEngine` and
  `BaseModelEngine` (import-light, stdlib + numpy) type the per-plugin
  inference contract; every engine derives from the matching base.
- Engines now carry their own class names (`VAEEngine`, `AudioSealEngine`,
  ..., mirroring the client classes).
- No behavior, HTTP contract, or CLI change (all 16 service contract
  fixtures re-verified).

## v1.0.0 — 2026-07-30

First release of the `deepmarkpy` package. Behavior is bit-for-bit identical
to the pre-package benchmark (verified by discovery lock tests, native-attack
golden fixtures, and HTTP-contract fixtures for all 16 services).

### Packaging

- The flat `src/` layout is now the `deepmarkpy` package (`pip install .`),
  with a `deepmark-benchmark` console script. `python src/run.py` keeps
  working through a deprecation shim.
- Python imports change accordingly: `from core.base_attack import ...`
  becomes `from deepmarkpy.core.base_attack import ...` (see
  `docs/MIGRATION.md`).
- Plugin discovery uses one module identity (`deepmarkpy.plugins.*`); the
  `sys.path` mutation is gone. Third-party plugin directories load via the
  new `--plugins_dir` flag or `DEEPMARK_PLUGINS_DIR` environment variable.
  Import failures are recorded in `PluginManager.failed`.
- New additive CLI surface: `--plugins_dir`, `--report_dir` (default
  `./report`, unchanged behavior). All existing flags and values unchanged.
- Container images install the package; behavior and the docker-compose
  workflow are unchanged.

### Plugin internals

- Every containerized plugin's inference logic lives in one `inference.py`
  module exposing a uniform `Engine` class behind a thin FastAPI `app.py`
  (see `docs/CONSUMING.md`). HTTP contracts are byte-identical.
- The `encodec` and `descript_audio_codec` attacks run as Docker services
  (ports `ENCODEC_PORT`/`DESCRIPT_AUDIO_CODEC_PORT`); PyTorch is no longer a
  host dependency.

### Known issues

- Known behavior quirks are intentionally preserved in this release and
  deferred to a later fix release.
