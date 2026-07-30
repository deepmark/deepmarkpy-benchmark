# Changelog

## v1.0.0 — 2026-07-30

First release of the `deepmarkpy` package. Behavior is bit-for-bit identical
to the pre-package benchmark (verified by discovery lock tests, native-attack
golden fixtures, and HTTP-contract fixtures for all 16 services; see
`docs/REORG_PLAN.md` for the verification policy).

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
  (see `docs/ENGINE_CONVENTIONS.md`). HTTP contracts are byte-identical.
- The `encodec` and `descript_audio_codec` attacks run as Docker services
  (ports `ENCODEC_PORT`/`DESCRIPT_AUDIO_CODEC_PORT`); PyTorch is no longer a
  host dependency.

### Known issues

- The defects catalogued in `docs/KNOWN_DEFECTS.md` are intentionally
  preserved in this release and deferred to a later fix release.
