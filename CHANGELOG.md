# Changelog

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

### Removed

- `docs/MIGRATION.md` (the v1.0.0 packaging migration).
- CI workflow, to be reintroduced in a later release.

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
