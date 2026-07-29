# DeepMarkPy Benchmark — Packaging & Plugin Reorganization Plan

**Status:** Ready for owner review (v2 — revised after adversarial verification pass)
**Date:** 2026-07-29
**Owner:** Slavko Kovačević (DeepMark)
**Applies to:** `deepmark/deepmarkpy-benchmark` (public), consumed by a private DeepMark repository

---

## 0. Executive summary

The public benchmark repository becomes an installable Python package (`deepmarkpy`) whose AI-plugin internals are standardized so that a private DeepMark repository can `pip install` it at a pinned version and build **its own** container images around each plugin's inference module. Two initiatives, strictly sequenced:

1. **Plugin-internal standardization** — all inference logic for each AI plugin consolidated into one conventionally named module (`inference.py`) exposing a uniform `Engine` class; `app.py` reduced to a thin FastAPI transport adapter. Container-internal change only; nothing user-visible moves.
2. **Packaging** — the flat `src/` namespace (`core`, `utils`, `plugins`, …) becomes a proper `deepmarkpy` package with `pyproject.toml`, a console entry point, tiered optional dependencies, and plugin assets shipped as package data. One coordinated breaking release (v2.0.0).

**Hard rules for every PR in this effort:**

- **Bit-for-bit behavior preservation.** No benchmark number, no HTTP response byte, no audio sample, no user-visible text (log lines, warnings, exception messages) may change. Known defects (Section 4) are documented and **explicitly not fixed** — including transport-level ones (missing timeouts, error-in-200 responses). All fixes are deferred to a separate, later release.
- **Published-paper red lines** (Section 3) are inviolable.
- Every phase and every PR leaves the repository fully working: tests green, `docker-compose` workflow intact, CLI unchanged (except the additive P2 surface specified in §6, whose defaults preserve current behavior exactly).

**Out of scope** (decided 2026-07-29): SageMaker specifics of any kind, image building/publishing (the private repo containerizes for itself), parallel/concurrent dispatch in the benchmark loop, wire-format changes (binary audio), and all defect fixes. The consumable plugin set is **ML plugins only**: 6 models + 9 ML attacks (Section 5.4). `network_transmission` remains compose-only (requires `NET_ADMIN`; not part of the consumable set).

---

## 1. Scope and non-goals

### 1.1 Goals

| # | Goal | Deliverable |
|---|------|-------------|
| G1 | Standardized AI-plugin internals | Per-plugin `inference.py` with uniform `Engine` class; thin `app.py`; consistent per-plugin file inventory |
| G2 | Installable package | `deepmarkpy` distribution: `pyproject.toml`, src-layout, console script, tiered extras, plugin dirs (incl. `config.json`, per-plugin `requirements.txt`, `Dockerfile`) as package data |
| G3 | Consumer contract | Documented, semver-stable import paths for the 15 consumable plugins' `Engine` classes + per-plugin build-requirements documentation for the private repo's own images |
| G4 | Safety net | Golden/behavior-freeze tests (per the determinism policy, §4.3) that prove G1/G2 changed nothing observable |

### 1.2 Non-goals (do not do these, even if they look adjacent)

- **No SageMaker anything** — no `/ping`, `/invocations`, port-8080 conventions, `model_fn` handlers, S3 envelopes. The private repo owns its serving layer.
- **No image publishing/CI-to-registry** — the private repo builds its own images.
- **No defect fixes** — see Section 4. This includes "obviously safe" fixes like adding request timeouts or `raise_for_status()` to existing attack clients: deferred.
- **No logging normalization and no dead-code removal** — even where code violates repo conventions (`print()`, module-level `logging.*` calls) or is provably inert, it moves **verbatim**. Both are severable cosmetic passes that belong to the deferred-fix release; changing them here would blur the bit-for-bit guarantee (container stdout/stderr is observable).
- **No wire-format changes** — JSON float-list payloads stay exactly as they are.
- **No concurrency work** in `benchmark.py` — the loop stays sequential.
- **No renames of plugin classes** — class names are registry keys and CLI values (`--wm_model AudioSealModel`) and are published in the paper's Algorithm 1 semantics.
- **No changes to `config.json` schemas or values.**
- **No adding seeds/generators to production code** — stochastic plugins stay stochastic (see §4.3).
- **No new benchmark features.**

### 1.3 Decisions already made (do not relitigate)

| Decision | Answer | Date |
|----------|--------|------|
| Who containerizes for the cloud? | The private repo, for its own use. This repo only makes plugins consumable. | 2026-07-29 |
| Consumable plugin set | ML plugins only (~15): 6 models + 9 ML attacks; native DSP attacks run in the orchestrator process; model-callback attacks stay orchestrator-side | 2026-07-29 |
| Defect policy | Preserve bit-for-bit; all fixes in a separate later release | 2026-07-29 |
| Re-containerize `encodec` / `descript_audio_codec` | Yes — standing decision predating this plan ("all ML plugins go back in containers") | pre-existing |
| Standard logic-module name | `inference.py` (echoes ecosystem convention; avoids `processor.py`/`engine.py` ambiguity) | this plan, §5.1 |

---

## 2. Ground truth (verified current state)

Everything below was verified against the working tree on 2026-07-29, then independently re-verified by a second review pass. Line numbers may drift; re-verify before relying on them in a PR.

### 2.1 Core mechanics

- `PluginManager` walks `src/plugins/{attacks,models}` (46 attack dirs + 6 model dirs), imports **only files literally named `attack.py` or `model.py`** (`src/plugin_manager.py:78`), registers every `BaseAttack`/`BaseModel` subclass by **class name**, and attaches the directory's `config.json`. It **mutates `sys.path`** (inserts `src/`, `plugin_manager.py:30-32`) and hardcodes `package_prefix="plugins"` (`plugin_manager.py:47,60`). Import failures are logged and swallowed — a plugin with a missing dependency silently vanishes from the registry.
- Requesting a missing/unloaded **model** raises (`src/benchmark.py:115-117, 269-271`). Requesting a missing/unloaded **attack** only logs `logger.warning(... "not found. Skipping.")` and the run completes with exit code 0, silently omitting it (`src/benchmark.py:363`). Both behaviors are preserved by this plan.
- `BaseModel` loads `config.json` via `inspect.getfile(self.__class__)` (`src/core/base_model.py:28-37`) and provides `_make_request` (timeout 300 s, `raise_for_status`) (`base_model.py:41-81`). `BaseAttack` has **no** HTTP helper. Dockerized attack clients split two ways: `vae`, `diffusion`, `neural_vocoder`, `speech_enhancement_1`, `speech_enhancement_2` hand-roll `requests.post` with **no timeout and no status check**; `opus_codec` (timeout 120 s), `network_transmission` (180 s), and `speech_tokenization` (600 s) already use timeout + `raise_for_status`.
- Import namespace is flat top-level: `core`, `utils`, `plugins`, `benchmark`, `plugin_manager`, `run` — ~90 flat-import lines in host-side `src/` code (~108 including tests; ~123 including the container-side `utils.utils` imports that P2 also rewrites); three `sys.path` insert sites (`plugin_manager.py:30-32`, `tests/conftest.py:10`, `tests/test_run.py:9`). No `pyproject.toml`/`setup.py` exists anywhere.
- Tests: **111 tests** collected (CLAUDE.md's "74" is stale), ~2-3 s, no Docker, no CI, no Makefile — the repo has no `.github/` directory. All tests import top-level names via conftest's path insert. `test_plugin_manager.py` constructs a real `PluginManager()`, importing all ~52 plugin modules.

### 2.2 Where inference logic actually lives

**Models (all 6 dockerized).** `model.py` is a pure HTTP client in every case; **all server-side inference lives in `app.py`**. There is no server-side logic file to rename — this is an *extraction*, not a consolidation:

| Model | app.py inference content | Extraction complexity |
|-------|--------------------------|----------------------|
| `audio_seal` | ~40 lines: tensor prep, `get_watermark`, additive embed, 1000-sample min-length guard, RuntimeError fallback | Medium |
| `aware` | Thin — inference is inside the pip-installed `aware` package; app.py does resample + NaN sanitize + error-in-200 wrapping | Low |
| `perth` | Model call + dead bit-packing (bits packed then `watermark=None` passed); 0-d/NaN handling. No `__main__` block (runs only via Dockerfile CMD) | Low-medium |
| `silent_cipher` | Bit↔byte message packing/unpacking + `encode_wav`/`decode_wav` + bare `except:` → `null` | Medium |
| `timbrewm` | Heaviest: 43-line model factory dispatching over six module variants + 3 YAML loads + bipolar mapping + sign-threshold decoding; `timeout_keep_alive=120` | High |
| `wavmark` | Thinnest: resample + 2 library calls | Low |

**Dockerized ML attacks.** Logic placement is inconsistent; naming has no convention:

| Attack | Logic file today | Real logic in app.py |
|--------|------------------|----------------------|
| `vae` | `vae.py` (RAVE, HF download) | ~15 lines: block-size-2048 truncation + 48 kHz resample round-trip |
| `diffusion` | `ddpm.py` (closest to target shape) | ~2 lines: computes `1000 - diffusion_steps` then dispatches — that inversion must land inside `Engine.apply`, not be dropped or duplicated |
| `neural_vocoder` | `big_vgan.py` | ~1 line; container imports utils under alias `app_utils` |
| `speech_enhancement_1` | `speech_brain.py` | 0 lines; app.py has no `__main__` block |
| `speech_enhancement_2` | **none** | ~40 lines incl. ClearVoice instantiated per request; no `__main__` block |
| `speech_tokenization` | `xcodec.py` | ~15 lines resampling (redundantly duplicated inside `xcodec.py`) |
| `opus_codec` | **none** | ~90 lines (`opusenc`/`opusdec` subprocess round-trip); plus real client-side DSP in `attack.py:103-113` (resample-back + length lock) |
| `network_transmission` | **none** | 557-line app.py (RTP, `tc netem`, jitter buffer, Opus FEC/PLC, APM, AGC). Compose-only; `NET_ADMIN` |
| `encodec`, `descript_audio_codec` | in-process `attack.py` (native, torch on host) | n/a — to be containerized per standing decision |

### 2.3 The "standalone file" blockers (why renaming alone fails)

- Container-side code of every `ml-services-base` plugin imports `utils.utils` (`load_config`, `resample_audio`, `renormalize_audio` — the only three functions used). This resolves **only** because `Dockerfile.base:27` bakes `src/utils` into `/app/utils` (with `PYTHONPATH=/app`); `neural_vocoder` grafts it a second time under the alias `app_utils`.
- Every compose service builds with `context: .` (whole repo, **no `.dockerignore`**) — any repo edit invalidates every image's COPY layers.
- Compose does not build `Dockerfile.base`; README requires a manual `docker build -f Dockerfile.base -t ml-services-base:latest .` first. 12/14 services are `FROM ml-services-base:latest`; `opus_codec` and `network_transmission` are standalone `python:3.10-slim`.
- `Dockerfile.base` pins only `torch==2.7.1`; torchaudio, numpy, scipy, fastapi, pydantic, librosa, transformers etc. are **unpinned**, as are most per-plugin requirements — image rebuilds weeks apart can resolve different environments (addressed in P0.2).
- Four Dockerfiles `git clone` upstream repos at build time; **three are unpinned** (`aware` — which also `pip install -e .`s its own unpinned dependency tree, `timbrewm`'s TimbreWatermarking + 5 `sed` patches, `neural_vocoder`'s NVIDIA/BigVGAN; `network_transmission` pins one of its two clones).
- Weights: `speech_tokenization` bakes weights at image build; `timbrewm`'s checkpoints arrive at build inside the cloned upstream repo; `audio_seal`, `silent_cipher`, `perth`, `wavmark` download/load at container start (module import); ClearVoice (`speech_enhancement_2`) instantiates per request; `vae` downloads at module import with a local cache. No compose volumes, so runtime-downloaded weights re-download on every container recreate.
- All six model Dockerfiles COPY the host-side `model.py` (which imports `core.base_model`, absent in-image) into the container — dead weight that also couples images to the host namespace.

### 2.4 Packaging landmine (highest-severity risk in this plan)

`package_prefix="plugins"` + the `sys.path` insert mean a naive packaging can import plugins under **two module identities** (`deepmarkpy.plugins.x.attack` vs top-level `plugins.x.attack`). Then `issubclass(obj, BaseAttack)` at `plugin_manager.py:93` compares classes from two different `BaseAttack` objects and is silently `False`: **discovery returns an empty registry with no error.** Mitigation: discovery lock tests land in P0, before any namespace work; PluginManager's import strategy is redesigned in P2 (single module identity, prefix derived from `__package__`, `sys.path` insert deleted).

### 2.5 Layout↔runtime couplings that packaging must respect

- `report/` output dir is CWD-relative and hardcoded (`src/run.py:292` and cleanup calls); LaTeX brand assets are user-provisioned inside it; `pdflatex` optional.
- `.env` is read **only** by docker-compose, never by Python; host clients use `os.getenv("X_PORT", default)` with hardcoded `localhost`. Port defaults are triplicated (.env.example / Dockerfile ENV / client fallback).
- `ReplayAttack`/`MixingAttack` need `AIR_wav_files/` and `music/` at CWD-relative paths; NISQA weights at `weights/nisqa.tar` (env-overridable). NISQA itself is not in `requirements.txt`; `metrics.py` silently skips it when absent (README documents manual install).
- README documents `python src/run.py ...` (10+ occurrences) and the extension recipe `from core.base_attack import BaseAttack` (README:339, 388) — both break at P2 and must be updated in the same PR.
- Release tags `v1.0.0` and `v1.1.0` exist. **P0 must confirm which tag reproduces the IEEE Access paper's evaluated state and record it.**

---

## 3. Invariants (published-paper red lines + compatibility matrix)

The paper (IEEE Access, vol. 14, 2026, pp. 62031-62044) publishes the following. None of it may change observably:

1. Class names **`BaseModel`** / **`BaseAttack`**; methods **`embed`**, **`detect`**, **`generate_watermark`**, **`apply`**.
2. Per-plugin-directory **`config.json`**; config/implementation separation ("users adjust attack parameters without modifying source code").
3. **Directory-walk auto-discovery** (Algorithm 1): drop a directory with `attack.py`/`model.py` + `config.json` and it is found without core changes. This recipe must keep working for third parties **after** packaging (see P2: external plugin-dir support).
4. **Dual execution modes** — native and containerized — with containerized components invoked "through standardized HTTP interfaces". docker-compose remains a first-class, fully working workflow.
5. Six attack categories; 40 attacks; 6 models; published abbreviations (AS, WM, SC, TWM, AW, PTh); percent-scale accuracy; 80% robustness threshold; PESQ/STOI metrics; confidence-based FPR/FNR path for AudioSeal/AWARE.
6. Results shape: file → attack → `{accuracy}`; per-model `sampling_rate` config key.
7. Public GitHub URL `github.com/deepmark/deepmarkpy-benchmark` stays the canonical home.

**Compatibility matrix** (what external users can rely on across this effort):

| Surface | Through P1 | After P2 (v2.0.0) |
|---------|-----------|-------------------|
| Existing CLI flags & values (incl. class-name args) | unchanged | unchanged; additive-only new surface (`deepmark-benchmark` script, `--plugins_dir`, `--report_dir`) with defaults that preserve current behavior exactly |
| `python src/run.py ...` | unchanged | works via deprecation shim (default; see §8/D2) |
| `config.json` schemas & values | unchanged | unchanged |
| docker-compose workflow (base image build → up → run) | unchanged | unchanged commands; internals repointed |
| HTTP endpoints & payload shapes | byte-identical | byte-identical |
| Python imports (`from core.base_attack import ...`) | unchanged | **breaking** → `from deepmarkpy.core.base_attack import ...`; README/migration guide in same PR |
| Third-party plugin drop-in recipe | unchanged | preserved via source checkout AND new `--plugins_dir`/env external-plugins support |

---

## 4. Behavior-preservation policy, defect register, determinism policy

### 4.1 Policy

**Definition of "preserved":** for identical inputs (audio, watermark, kwargs, config), every attack/model produces identical outputs — audio arrays, HTTP request/response bodies, accuracy numbers, error behavior (including the *wrong* error behaviors listed below), and user-visible text (warnings, exception messages). For inherently stochastic plugins, "identical" is defined by the determinism policy (§4.3). For plugins whose execution environment changes (encodec/DAC containerization, P1.6), the standard is `np.allclose` within a documented tolerance **plus** the fixed-watermark accuracy protocol of P1.6, and the PR must report both.

**Allowed changes** (cannot alter benchmark-observable behavior — audio arrays, HTTP bytes, accuracy numbers, error responses, user-visible text):
- Moving code between files inside a container with an identical call sequence (including moving `print()`/module-level logging calls **verbatim** — normalizing them is deferred).
- Comments, type hints, docstrings.
- Build-reproducibility measures: git-clone commit pins; capturing pip-freeze constraint files that pin each image to its **currently-resolved** dependency set (a no-op at capture time); a `.dockerignore` restricted to paths no Dockerfile COPY can match (`.git`, `report/`, `weights/`, `*.pdf`, `tests/`, `.env`), verified by re-running contract fixtures against the post-`.dockerignore` rebuild.
- Deleting the empty `.gitmodules`.
- README/doc corrections that describe reality (e.g. README's false claim that `.env` ships; CLAUDE.md's stale test count and publication venue).
- New **programmatic-only** APIs that no existing code path calls (e.g. the P2 `PluginManager.failed` registry), provided no user-visible text changes.

**Forbidden until the deferred-fix release** (the defect register below, plus anything with the same character):
- Adding timeouts, `raise_for_status`, or a `BaseAttack._make_request` helper to **existing** clients.
- Changing any error response shape (error-in-200 stays error-in-200) or any warning/exception message text.
- Fixing kwarg mismatches, sample-rate mismatches, dead parameters, redundant resamples, dead code (however provably inert — including `big_vgan.py:7`'s WORKDIR-dependent no-op `sys.path.append`).
- Logging normalization (`print()` → `logger`, module-level `logging.*` → `logger`).
- Hoisting per-request model instantiation to load-once (`speech_enhancement_2`).
- Seeding or otherwise de-randomizing stochastic inference paths.
- Any dependency version change beyond the freeze-in-place constraint capture above.
- Raising on requested-but-missing **attacks** (today: warn-and-skip, exit 0 — preserved).

### 4.2 Known-defect register (documented, frozen, deferred)

Ship this as `docs/KNOWN_DEFECTS.md` in P0 so no agent "helpfully" fixes one mid-refactor. Each entry: symptom, location, why frozen, target release.

| # | Defect | Location |
|---|--------|----------|
| D1 | `mixing` passes `gains=` where `EqualizerAttack` reads `gains_equalizer` → configured gains silently ignored | `src/plugins/attacks/mixing/attack.py:242` |
| D2 | `inverted_time_stretch` passes `stretch_rate=` where `TimeStretchAttack` reads `stretch_rate_time_stretch` → configured value ignored | `src/plugins/attacks/inverted_time_stretch/attack.py:43,48` |
| D3 | `silent_cipher` config declares 16 kHz while app.py loads the 44.1k checkpoint | `src/plugins/models/silent_cipher/app.py:22` |
| D4 | `perth` packs watermark bits then passes `watermark=None` (payload discarded). Never deletable as "dead code" | `src/plugins/models/perth/app.py:48-50` |
| D5 | `speech_enhancement_2` returns HTTP 200 with `{"error":…, "audio": null}`; client guard passes → 0-d object array downstream | `…/speech_enhancement_2/app.py:72`, `attack.py:41` |
| D6 | `aware` returns errors as HTTP-200 bodies | `src/plugins/models/aware/app.py:77,112` |
| D7 | Five dockerized attack clients (`vae`, `diffusion`, `neural_vocoder`, `speech_enhancement_1`, `speech_enhancement_2`) use bare `requests.post` with no timeout and no `raise_for_status`; `BaseAttack` lacks a `_make_request` helper. (`opus_codec` 120 s, `network_transmission` 180 s, and `speech_tokenization` 600 s already have both — two co-existing client patterns) | respective `attack.py` files |
| D8 | `audio_seal` `detect` annotated `-> np.ndarray` but returns a tuple | `src/plugins/models/audio_seal/model.py:50,62` |
| D9 | `speech_tokenization` resamples redundantly in both app.py and xcodec.py | `…/app.py:45-58`, `xcodec.py:15,24` |
| D10 | `silent_cipher` bare `except:` swallows decode failures → `null` | `…/silent_cipher/app.py:73` |
| D11 | `resample_audio` fed raw list vs ndarray inconsistently, even within one file | `audio_seal/app.py:47,74`; `wavmark/app.py:46 vs 61` |
| D12 | Port defaults triplicated (.env.example / Dockerfile ENV / client fallback); `.env` never loaded by Python | repo-wide |
| D13 | `timbrewm` requirements downgrade base torch 2.7.1→2.0.0; `speech_tokenization` →2.4.1 | per-plugin `requirements.txt` |
| D14 | `pywt`, `pyrubberband` missing from `requirements.txt` → `wavelet`, `pitch_shift`, `time_stretch` silently absent on fresh installs (paper promises 40 attacks) | `requirements.txt` |
| D15 | Requested-but-missing attacks are warn-and-skipped (exit 0, results silently incomplete); only missing models raise | `src/benchmark.py:363` vs `:115-117` |

**D14 stays frozen in this effort** (per the all-fixes-deferred decision): P2's `[native-attacks]` extra ships **without** `pywt`/`pyrubberband`; the gap is documented in `KNOWN_DEFECTS.md` and `CONSUMING.md`, and the additions land in the deferred-fix release. The P0.1 lock tests and all discovery assertions are therefore recorded against the canonical environment (§4.3), in which these three attacks are absent — the asserted set does not change anywhere in this effort. (Owner may override via §8/D5; if so, DoD item 7 must be amended in the same commit.)

### 4.3 Determinism policy (governs all golden/fixture testing)

**Canonical environment:** `pip install -r requirements.txt` into a clean venv, exactly — no additional packages. The discovery lock tests assert the exact class-name sets registered in this environment (documented alongside the test). Byte-identity claims are valid only **same-machine, same-image-lineage**: fixtures are recorded and re-recorded on one designated machine (the owner's; document its architecture in the fixture README — torch CPU numerics differ across x86/arm64, so cross-machine byte-comparison is out of scope).

**Canonical input:** unless a plugin needs otherwise, golden inputs are `np.random.default_rng(42).standard_normal(sr)` (≈1 s) scaled to ±0.5, at the plugin's configured `sampling_rate`, generated by a committed helper — the exact expression lives in the fixture-generation script, not in prose. Fixtures live under `tests/fixtures/goldens/` (native attacks) and `tests/fixtures/contracts/` (HTTP services); audio payloads stored as compressed `.npz` (repo `.gitignore` blocks `*.wav`), schemas as JSON; keep inputs short so fixtures stay small.

**Service classification (P0.4 verifies each by calling it twice and diffing):**

- **Expected deterministic** — byte-identical re-record diffs required: `audio_seal`, `aware`, `perth`, `silent_cipher`, `timbrewm`, `wavmark`, `neural_vocoder`, `speech_tokenization`, `opus_codec`, and `vae` (eval-mode TorchScript forward — but some RAVE exports sample latents internally, so vae's classification **must** come from the double-call check, not assumption).
- **Stochastic** — schema assertions + tolerance/statistical checks (energy, length, SNR-band) instead of byte-identity: `diffusion` (`ddpm.py:27-28` draws a fresh OS-entropy seed per request; not even `allclose` across calls), `network_transmission` (kernel `tc netem` loss/jitter + random SSRC), `speech_enhancement_2` (unseeded server-side noise, `noise_strength_se2=0.01` from config, not overridable per request).
- **Conditionally deterministic:** `speech_enhancement_1` — noise is request-controlled (`noise_strength` field); contract fixtures are recorded with `noise_strength=0.0` **in the request** (a legal kwargs-path value, not a config change) and are then byte-comparable.
- **Native attacks (P0.3):** the golden harness calls `np.random.seed(<fixed constant>)` immediately before each attack invocation (pinning legacy global-RNG draws in `gaussian_noise`, `pink_noise`, `additive_noise`, `echo`, `cut_samples`, `flip_samples`, `zero_cross_inserts`, etc.). Model-callback attacks (`same_model`, `cross_model`, `collusion`, `collusion_2`, `zero_bit_collusion`) and corpus-dependent attacks (`replay`, `mixing`) are **excluded from P0.3 goldens** with that rationale recorded — they need live models or on-disk corpora, they are not touched by G1, and P2's protection for them is the discovery lock tests plus the existing unit suite. (`collusion_2` would additionally be unseedable even if included: it draws from a fresh `np.random.default_rng()` at `attack.py:129`, beyond `np.random.seed`'s reach.) `replacement`/`replacement_2` are goldenable (deterministic DSP) and included.
- **Adding seeds to production code is forbidden** (§4.1). Test-only seams require owner sign-off.

---

## 5. Target architecture

### 5.1 Per-plugin standard layout (AI/containerized plugins)

```
src/…/plugins/attacks/<name>/          # same for models/
├── attack.py        # host-side HTTP client extending BaseAttack (models: model.py / BaseModel) — UNCHANGED name (discovery contract)
├── inference.py     # ← ALL inference logic. The consumable unit's entry point.
├── app.py           # thin FastAPI adapter: parse request → Engine call → serialize response. Target ≲40 lines. No DSP/ML.
├── config.json      # unchanged
├── requirements.txt # container/runtime deps for this plugin
└── Dockerfile       # unchanged behavior; stops COPYing host-side client files
```

`inference.py` exposes a single class named **`Engine`** (uniform across all plugins; the module path disambiguates):

```python
class Engine:
    """All inference for <plugin>. No FastAPI/HTTP imports. No benchmark-host imports
    beyond deepmarkpy.utils (numpy/scipy/librosa-level deps only)."""
    def __init__(self, config: dict, device: str | None = None): ...   # weight load, device placement
    # attacks:
    def apply(self, audio: np.ndarray, sampling_rate: int, **params) -> np.ndarray: ...
    # models:
    def embed(self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int) -> np.ndarray: ...
    def detect(self, audio: np.ndarray, sampling_rate: int) -> "np.ndarray | tuple | float": ...
```

Rules:
- `Engine` owns **everything between "request parsed" and "response built"** — including the resampling round-trips, length guards, truncations, message bit-packing, and sanitization that currently live in `app.py`. `app.py` keeps only: Pydantic parsing, the `Engine` call, `.tolist()` serialization, and the exact current error semantics (even where those are wrong — D5/D6).
- `Engine.__init__` loads weights where the current code loads them at startup. **Exception:** where current code instantiates per request (`speech_enhancement_2`'s ClearVoice), `Engine.apply` keeps per-request instantiation — hoisting is a deferred fix.
- Optional lightweight ABCs (`deepmarkpy.core.inference.AttackEngine` / `ModelEngine`) may be introduced in P2 to type the contract; they must be import-light (numpy only) and are internal API, not a published concept.
- Where the current split differs (e.g. `opus_codec`'s client-side resample-back in `attack.py:103-113`), **do not move logic across the HTTP boundary** — that would change payloads. Standardize within each side only.
- Native (pure-DSP) attacks are untouched by G1: `attack.py` + `config.json` remains their complete, correct layout.

### 5.2 Package layout (after P2)

```
pyproject.toml
src/deepmarkpy/
├── __init__.py            # __version__, lazy top-level exports
├── run.py                 # CLI (console script: deepmark-benchmark)
├── benchmark.py
├── plugin_manager.py      # single-module-identity import strategy; external plugin-dir support; failed-plugin registry (programmatic only)
├── core/                  # base_model.py, base_attack.py (+ optional inference.py ABCs)
├── utils/                 # utils.py, metrics.py, report generators, …
└── plugins/
    ├── attacks/<name>/…   # per §5.1, incl. config.json/requirements.txt/Dockerfile as package data
    └── models/<name>/…
src/run.py                 # deprecation shim (default; open decision D2 in §8)
tests/
```

- **Distribution name:** `deepmarkpy` (open decision D1 in §8 if the owner prefers `deepmark-benchmark` on the index; import name stays `deepmarkpy` either way).
- **Dependency tiers:** core = `numpy, scipy, librosa, soundfile, requests` (+ small); extras: `[metrics]` (pesq, pystoi, visqol; nisqa — note: installing extras changes *metric availability* exactly as README's manual-install instructions do today, an availability change, not a numeric one — flag in the PR), `[native-attacks]` (pycodec2, audiocomplib — **without** pywt/pyrubberband per D14 freeze), `[reports]` (matplotlib, LaTeX helpers), `[all]`. Torch leaves the host requirements entirely once encodec/DAC are containerized (P1.6). `requirements.txt` remains as the pinned dev/repro lockfile.
- **PluginManager redesign (P2):** derive the package prefix from `__package__`; import plugins as `deepmarkpy.plugins.…` only; delete the `sys.path` insert; keep registering by class name with the directory's `config.json`; add `--plugins_dir` / `DEEPMARK_PLUGINS_DIR` so third parties can drop plugin directories outside site-packages (preserving red line 3 post-install); record import failures in a **programmatic-only** `failed` registry (no log/message changes). Missing-model errors keep their exact current message; missing-attack warn-and-skip is preserved verbatim (D15). Raise-on-failed-attack and richer messages belong to the deferred-fix release.
- **Containers install the package:** `Dockerfile.base` stops grafting `src/utils`; images instead `pip install` the local package (build-stage wheel or `pip install .` from the build context) and `inference.py`/`app.py` import `deepmarkpy.utils.utils`. `neural_vocoder`'s `app_utils` alias dies here. Compose commands and behavior stay identical.

### 5.3 Consumer contract (the private repo's interface)

- **What the consumer does:** `pip install deepmarkpy==X.Y.Z` (from PyPI, a private index, or `git+https://…@vX.Y.Z` — open decision D3), then `from deepmarkpy.plugins.attacks.vae.inference import Engine` and wraps `Engine` in its own serving layer and images. It never imports this repo's `app.py` and never depends on the HTTP layer.
- **Stability promise:** the module paths `deepmarkpy.plugins.{attacks,models}.<name>.inference` and the `Engine` signatures in §5.1 are semver-stable public API from v2.0.0. Anything else in plugin dirs is internal.
- **Per-plugin build documentation** (`docs/CONSUMING.md` + a "Build requirements" section per consumable plugin): Python/torch version constraints, per-plugin `requirements.txt` + the frozen constraint files from P0.2, apt/system packages (e.g. `opus-tools`), upstream clones with **pinned commits** and required patches (timbrewm's 5 `sed` edits become a documented patch file), weights acquisition (build-baked vs runtime download; exact artifact names), env vars, licensing notes. For timbrewm-class plugins this document *is* the consumable interface.

### 5.4 The consumable set (15)

Models (6): `audio_seal`, `aware`, `perth`, `silent_cipher`, `timbrewm`, `wavmark`.
ML attacks (9): `vae`, `diffusion`, `neural_vocoder`, `speech_enhancement_1`, `speech_enhancement_2`, `speech_tokenization`, `opus_codec`, `encodec`, `descript_audio_codec` (last two containerized in P1.6).

Explicitly **not** consumable (documented as such): `network_transmission` (needs `NET_ADMIN`; compose-only); the model-callback attacks `same_model`, `cross_model`, `collusion`, `collusion_2`, `zero_bit_collusion` (need live model instances / the registry / paired originals — orchestrator-side by design); `mixing` (composes sibling plugins + on-disk music corpus); `replacement_2` (imports from sibling `replacement`); all remaining native DSP attacks (consumed via the package's orchestrator, not as standalone engines).

---

## 6. Phased execution plan

Phases are strictly ordered. **P0's internal order matters**: pins and constraint capture (P0.2) come *before* fixture recording (P0.3/P0.4), so fixtures are recorded against the frozen, reproducible image lineage. Within P1, plugin PRs may proceed in parallel after P1.1 lands.

### P0 — Safety net and hygiene (no observable behavior change)

| PR | Content | Acceptance criteria |
|----|---------|---------------------|
| P0.1 | **Discovery lock tests**: assert the exact registered model/attack class-name sets in the canonical environment (§4.3); assert `config.json` attachment for a sample of plugins | Test fails if any plugin silently vanishes; canonical env documented in the test |
| P0.2 | **Reproducibility freeze & hygiene**: pin `aware`, TimbreWatermarking, and BigVGAN clones to commits (choose the commit resolvable from currently-built local images where determinable, else current upstream HEAD); capture per-image pip-freeze constraint files and wire them into the Dockerfiles (freeze-in-place — no version changes); add the restricted `.dockerignore` (§4.1); delete empty `.gitmodules`; fix README's false `.env` claim (+ add the copy-`.env.example` step); confirm and document the paper-reproducing tag (expected `v1.0.0` — verify) | All images rebuild; §P0.4 fixtures (once recorded) pass against a second rebuild — this, not layer-digest identity, is the reproducibility criterion |
| P0.3 | **Native-attack golden tests** per §4.3: seeded canonical input + `np.random.seed` protocol → committed `.npz` digests; `collusion_2` statistical-only; model-callback and corpus attacks excluded with rationale | Re-run twice → identical (statistical checks for the documented exceptions); fixtures small; exclusion list in the test file |
| P0.4 | **HTTP-contract freeze fixtures** per §4.3: double-call determinism classification for all 14 services (recorded); byte-identical goldens for deterministic services, schema+tolerance fixtures for stochastic ones, `noise_strength=0.0` request for SE1; recorded on the designated machine against post-P0.2 images; a documented `scripts/contract_check.py` re-records against live services (there is no Makefile/CI in this repo — do not reference either) | Classification table committed; fixtures for all 14 services; re-record procedure reproducible same-machine |
| P0.5 | **`docs/KNOWN_DEFECTS.md`** (§4.2 verbatim + why-frozen rationale) and **commit this plan** as `docs/REORG_PLAN.md` (per-session agents need it in-tree) | Register + plan merged; linked from README/CONTRIBUTING |
| P0.6 | Docs truth-up: CLAUDE.md (stale test count "74"→111, publication venue → IEEE Access, this plan's conventions). Optional per §8/D9: bootstrap minimal GitHub Actions (pytest only, no Docker) | — |

### P1 — Plugin-internal standardization (extraction; container-internal only)

| PR | Content |
|----|---------|
| P1.1 | **Template & conventions PR**: `inference.py`/`Engine` conventions doc (§5.1); convert the first plugin — **`vae` if P0.4 classified it deterministic, else `wavmark`** (a byte-comparable template is mandatory; `diffusion` is stochastic and cannot set the pattern). Moves app.py's truncation+resample (vae) into `Engine.apply` |
| P1.2 | Attacks with existing logic files: `diffusion` (`ddpm.py` → `inference.py`; the `1000 - diffusion_steps` inversion lands inside `Engine.apply`; verified under stochastic criteria per §4.3), `neural_vocoder` (`big_vgan.py` → `inference.py`; keep the `app_utils` import mechanics and the inert `sys.path.append` line intact until P2), `speech_enhancement_1` (`speech_brain.py` → `inference.py`), `speech_tokenization` (`xcodec.py` → `inference.py`; keep the redundant double-resample — D9) |
| P1.3 | Attacks with logic in app.py: `speech_enhancement_2` (extract ~40 lines; keep per-request ClearVoice instantiation — §5.1; stochastic criteria), `opus_codec` (extract subprocess round-trip server-side; leave client-side resample in `attack.py` untouched) |
| P1.4 | Models, easy: `wavmark` (if not already the template), `aware`, `perth` (preserve D4 dead bit-packing exactly), `audio_seal` (min-length guard + fallback into `Engine`) |
| P1.5 | Models, hard: `silent_cipher` (bit↔byte packing into `Engine`; preserve D3/D10), `timbrewm` (model factory + YAML loading into `Engine`; preserve exact checkpoint/patch behavior) |
| P1.6 | **Containerize `encodec` + `descript_audio_codec`** in the standard layout: new `inference.py`, thin `app.py`, `Dockerfile` (FROM `ml-services-base:latest`), `requirements.txt` (torch pinned to the version currently used in-process), compose services + `.env.example` ports following existing conventions (service names `encodec`, `descript_audio_codec`; vars `ENCODEC_PORT`, `DESCRIPT_AUDIO_CODEC_PORT`); `attack.py` becomes a thin HTTP client (pattern per §8/D8). **Comparison protocol:** record in-process goldens (fixed watermark via the existing `watermark_data` path, fixed input files, fixed `np.random.seed`) *before* removing torch from host requirements; then compare in-process vs container: `np.allclose` audio within an owner-approved tolerance **and** identical accuracy under the fixed-watermark protocol. Torch/torchaudio/encodec/descript-audio-codec removed from host `requirements.txt` **(flag: environment change for existing users; README + CHANGELOG)** |
| P1.7 | `network_transmission` (optional, last, lowest priority): extract the 557-line app.py into `inference.py` where separable (RTP/netem orchestration may remain app-level given its process/`tc` coupling); may be deferred past P2 without blocking anything (not consumable) |

Every P1 PR: (a) fixtures from P0.3/P0.4 re-verified against the rebuilt container per the §4.3 criteria for that service's class (byte-identical for deterministic, tolerance/statistical for stochastic); (b) image builds from a clean context; (c) compose smoke test; (d) no host-side files changed except the plugin dir (P1.6 excepted); (e) Dockerfiles stop COPYing host-side `model.py`/`attack.py` into images while touched; (f) code moved **verbatim** — tidying limited to §4.1 allowed changes.

### P2 — Packaging (one coordinated breaking release: v2.0.0)

Content (what): items 1-8 below. Execution (how): the commit sequence in §6.1 — the src-layout move is not decomposable into independently-green PRs without it.

1. `pyproject.toml` (PEP 621): name (§8/D1), version 2.0.0, Python floor verified against deps (expect `>=3.10`), core deps + tiered extras (§5.2), console script `deepmark-benchmark = deepmarkpy.run:main`, package data (`config.json`, per-plugin `requirements.txt` + constraint files, `Dockerfile`).
2. Move to src-layout `src/deepmarkpy/…` (use `git mv` — history must survive); rewrite all flat-namespace import sites (~90 host-side + ~18 test-side + container-side `utils.utils` per item 4); delete all three `sys.path` inserts; keep `src/run.py` as deprecation shim (§8/D2).
3. PluginManager redesign per §5.2 — single module identity; `__package__`-derived prefix; external plugin dir; programmatic-only `failed` registry. **No user-visible text changes**; missing-attack warn-and-skip preserved verbatim (D15).
4. Container repoint: `Dockerfile.base` installs the package instead of grafting `src/utils`; all Dockerfiles' COPY paths updated; `app.py`/`inference.py` imports → `deepmarkpy.utils.utils`; kill the `app_utils` alias; compose behavior identical (contract fixtures re-verified per §4.3).
5. Tests migrated to package imports in the same commit as the move; discovery lock tests keep the **same asserted sets** (D14 stays frozen, so no set changes; "never weakened" = no removals, and no additions are expected in this effort).
6. Docs in the same release: README (install, `pip install -e .[all]` dev flow, extension recipe with new imports **and** the `--plugins_dir` drop-in path, compose workflow), `CHANGELOG.md` (initialized retroactively from v1.x tags), migration guide v1→v2, `docs/CONSUMING.md` skeleton.
7. Report-path portability: add `--report_dir` (default `report`, CWD-relative — unchanged default behavior) so the console script is usable outside a checkout. Same pattern for `AIR_wav_files`/`music/` paths **only if** achievable via additive defaults-preserved flags; otherwise document CWD requirements.
8. Release: tag `v2.0.0`; verify `pip install` from a clean venv → `deepmark-benchmark --help`, discovery sets, native-attack goldens, compose stack up + contract fixtures re-verified.

**Gate:** P2 merges only when P0 fixtures pass per §4.3 and a clean-machine walkthrough of the README succeeds end-to-end.

#### 6.1 P2 execution appendix (commit sequence, rollback)

Prescribed commit sequence — each numbered commit must leave the suite green:

1. Add `pyproject.toml` + empty `src/deepmarkpy/` skeleton coexisting with the old layout (installs nothing yet; suite untouched).
2. **Atomic move**: `git mv` of `core/ utils/ plugins/ benchmark.py plugin_manager.py run.py` into `src/deepmarkpy/`, the full import rewrite, conftest/sys.path deletions, and test migration — one commit; the suite cannot be green mid-way, so this commit is indivisible.
3. PluginManager redesign + discovery lock tests passing against the installed package.
4. Container repoint + compose rebuild + contract-fixture verification.
5. Shim, docs, CHANGELOG, migration guide.

Rollback: cut a `v1.x` maintenance branch from the pre-P2 tip **before** merging; if v2.0.0 proves broken post-tag, patch-release (`v2.0.1`) — never retract a published tag; abort criteria for the P2 attempt: discovery sets diverge from P0.1, any deterministic contract fixture fails, or the clean-machine README walkthrough fails twice.

### P3 — Consumer enablement and closure

| PR | Content |
|----|---------|
| P3.1 | `docs/CONSUMING.md` completed per §5.3: the 15 consumable plugins, import contract, semver policy, per-plugin build requirements (pins, patches-as-files for timbrewm, weights, system deps, licenses) |
| P3.2 | Private-repo validation: the consumer team (or an agent with access) builds one model image (`audio_seal`) and one attack image (`vae`) from the wheel alone, following only `CONSUMING.md`. Friction found → doc/package-data fixes here |
| P3.3 | Close-out: `docs/KNOWN_DEFECTS.md` re-verified against final line numbers; deferred-fix release scoped as a follow-up plan (out of scope here) |

---

## 7. Risk register

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Silent empty plugin registry after namespace change (dual module identity, §2.4) | Critical | P0.1 lock tests land first; P2 deletes the `sys.path` insert and derives the prefix; §6.1 commit 3 verifies against the installed package |
| Fixture false-failures from nondeterminism misclassification | High | §4.3 double-call classification is recorded evidence, not assumption; stochastic services never gate on byte-identity |
| Extraction subtly changes numerics (resample order, dtype, squeeze semantics) | High | Verbatim-move rule; §4.3 fixtures diffed per P1 PR |
| encodec/DAC containerization shifts numerics across torch builds | High | Torch version pinned to current in-process version; P1.6 fixed-watermark protocol; owner sign-off on tolerance |
| Image environments drift mid-effort (unpinned pip deps) | High | P0.2 freeze-in-place constraint files before any fixture is recorded |
| All-at-once test breakage in P2 removes the safety net when most needed | High | §6.1 atomic-move commit; golden fixtures are import-independent artifacts (JSON/`.npz`) that survive the rename |
| Docker rebuild churn during P1 (whole-repo contexts invalidate all images) | Medium | Restricted `.dockerignore` lands in P0.2, before P1 starts |
| README/paper/code disagreement at intermediate states | Medium | Breaking changes batched into the single v2.0.0 release; docs updated in the same PRs that break them |
| Unpinned upstream clones change under us mid-effort | Medium | P0.2 pins before fixtures are recorded or P1 starts |
| Agents "fix" defects mid-refactor | Medium | `docs/KNOWN_DEFECTS.md` + §4.1 forbidden list + prompt rules; reviewer checklist |
| `timbrewm` extraction destabilizes the sed-patched upstream integration | Medium | P1.5 isolated PR; contract fixtures; patches converted to committed patch files only as P3.1 documentation |
| Consumer finds the wheel insufficient for image builds (missing data/pins) | Medium | P3.2 validation exercise before declaring done |

---

## 8. Open decisions (defaults apply unless the owner overrides)

| # | Decision | Default |
|---|----------|---------|
| D1 | Distribution name on the index | `deepmarkpy` |
| D2 | Keep `python src/run.py` working post-v2.0.0 | Yes — deprecation shim for ≥1 minor cycle (README switches to `deepmark-benchmark`) |
| D3 | Package distribution channel for the consumer | Pinned git tag (`pip install git+https://…@v2.0.0`); PyPI optional later |
| D4 | Ship per-plugin `Dockerfile`s + constraint files in the wheel as package data | Yes (reference value for the consumer; small) |
| D5 | D14 (missing `pywt`/`pyrubberband`) | **Defer** to the fix release (register stays intact; DoD item 7 holds). Owner may override with explicit sign-off, amending DoD item 7 in the same commit |
| D6 | Optional `AttackEngine`/`ModelEngine` ABCs in `deepmarkpy.core.inference` | Yes, import-light, internal |
| D7 | `network_transmission` extraction (P1.7) | Attempt after P2; skip if risk-heavy |
| D8 | HTTP-client pattern for the **new** encodec/DAC `attack.py` clients (new code, so no defect-preservation question) | Follow the `opus_codec`/`speech_tokenization` pattern (explicit timeout — suggest 600 s for heavy codecs — + `raise_for_status`), the repo's most recent standing style |
| D9 | Bootstrap minimal CI (GitHub Actions: pytest, no Docker) in P0.6 | Yes — additive infrastructure; without it "tests green throughout" has no enforcement point |

---

## 9. Definition of done

1. All 111+ tests (plus the new lock/golden suites) green throughout; discovery lock-test assertions never weakened (no removals; no changes expected at all given §8/D5).
2. Every P0.4 contract fixture passes from v1.x images to v2.0.0 images under its §4.3 criterion — byte-identical for deterministic services (same machine, frozen constraints), tolerance/statistical for stochastic ones, P1.6 protocol for encodec/DAC.
3. `pip install` from clean venv: `deepmark-benchmark --help` works; discovery matches the canonical-environment sets.
4. docker-compose workflow works exactly as README describes, from a clean clone, on a clean machine.
5. The private repo (or a stand-in exercise, P3.2) builds two images from the wheel + `CONSUMING.md` alone.
6. `CHANGELOG.md`, migration guide, `KNOWN_DEFECTS.md`, `CONSUMING.md`, `docs/REORG_PLAN.md` merged; paper-reproducing tag documented.
7. Zero entries of the defect register fixed; zero paper red lines crossed.
