# `inference.py` / `Engine` conventions (P1, REORG_PLAN.md §5.1)

Every AI/containerized plugin consolidates its inference logic into one
module, `inference.py`, exposing a single class named **`Engine`** (uniform
across plugins; the module path disambiguates). The reference implementation
is `src/plugins/models/wavmark/` (P1.1 template).

## The contract

```python
class Engine:
    """All inference for <plugin>. No FastAPI/HTTP imports. No benchmark-host
    imports beyond utils (numpy/scipy/librosa-level deps only)."""
    def __init__(self, config: dict, device: str | None = None): ...
    # attacks:
    def apply(self, audio, sampling_rate, **params) -> np.ndarray: ...
    # models:
    def embed(self, audio, watermark_data, sampling_rate) -> np.ndarray: ...
    def detect(self, audio, sampling_rate) -> "np.ndarray | tuple | float": ...
```

- **`Engine` owns everything between "request parsed" and "response built"**:
  resampling round-trips, truncations, length guards, message bit-packing,
  sanitization. `app.py` keeps only Pydantic parsing, the `Engine` call,
  `.tolist()` serialization, and the exact current error semantics — even
  where those are wrong (error-in-200 shapes D5/D6 stay).
- **`__init__` loads weights where the current code loads them.**
  Startup-loaded stays startup-loaded; `speech_enhancement_2`'s per-request
  ClearVoice instantiation stays per-request inside `Engine.apply` (frozen
  defect scope). `device=None` means "decide as the current code does"
  (typically cuda-if-available).
- **`app.py` is a thin adapter, target ≲40 lines.** Config loading and its
  current failure behavior (e.g. `sys.exit(1)` with the exact log message)
  stay in `app.py`; the parsed dict is passed to `Engine`.
- **No logic moves across the HTTP boundary** (client↔server) — that changes
  payloads. Standardize within each side only. Host-side `model.py`/
  `attack.py` clients are untouched (their filenames are the discovery
  contract).

## Verbatim-move rules (REORG_PLAN.md §4.1/§4.2 — read before extracting)

- Moved logic keeps an **identical call sequence with identical argument
  values**. Only the mechanical `request.<field>` → parameter renames that
  any extraction forces are acceptable; nothing else changes. Known defects
  ride along **verbatim** — e.g. wavmark feeds `resample_audio` the raw
  request list in embed but the ndarray in detect (D11): the Engine preserves
  both, including which value each call receives.
- `print()` and module-level logging calls move verbatim; no normalization.
  Dead code moves too, however provably inert (D4's bit-packing).
- Startup code may reorder only within the startup phase and only when
  nothing user-visible changes (no stdout/stderr difference, no HTTP
  difference); disclose any such reorder in the PR description.
- Type hints, docstrings, and comments may be added (docstrings state
  contracts, not narration) — in the move commit for the new `Engine`
  surface, in a separate tidy commit for anything else.

## Dockerfile rule (while touched)

Stop COPYing the host-side client (`model.py`/`attack.py`) into the image —
replace whole-directory COPYs with explicit COPYs of what the container
runs (`app.py`, `inference.py`, `config.json`; requirements/constraints are
already copied earlier). Change nothing else about the image's runtime
behavior.

## Per-plugin verification checklist (every P1 PR)

1. P0.3/P0.4 fixtures exist for the plugin, with its determinism class
   recorded (`tests/fixtures/contracts/README.md`).
2. Move commit, then optional tidy commit — never combined.
3. Rebuild the image from a clean context (`--no-cache` for the plugin's
   layers); `docker-compose up` the service.
4. Re-verify the plugin's contract fixture under its §4.3 criterion
   (`python scripts/contract_check.py verify <service>`): byte-identical for
   deterministic services, schema+tolerance for stochastic ones.
5. Full test suite + discovery lock tests green; no host-side files changed
   except the plugin directory.
6. PR description: what moved, verification output pasted, determinism
   class, allowed-change items applied, explicit confirmation that no
   `KNOWN_DEFECTS.md` entry was altered and no user-visible text changed.
