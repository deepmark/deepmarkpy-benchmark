# Native-attack golden fixtures

Recorded by `scripts/generate_native_goldens.py`;
replayed by `tests/test_native_goldens.py`. `manifest.json` carries the
authoritative metadata: recording date, environment lineage, numeric-path
package versions, input expression, seed protocol, per-attack output
dtype/shape, and the exclusion lists with rationale.

## Designated machine

Byte-identity claims are same-machine, same-environment only. These fixtures
were recorded on the owner's designated machine:

- **Architecture:** macOS arm64 (Apple Silicon; Darwin 25.3.0)
- **Environment:** the canonical environment — clean venv, CPython 3.11.5,
  `pip install -r requirements.txt` exactly (see `manifest.json`
  `numeric_env` for the resolved numeric-path package versions)
- **External tools:** `Mp3CompressionAttack` shells out to the machine's
  ffmpeg (exact build recorded in `manifest.json`)

torch CPU numerics differ across x86/arm64, so cross-machine byte comparison
is out of scope. The replay test enforces byte-identity only when the
running environment's numeric-path package versions match the manifest, and
skips with an explanatory reason otherwise.

## Re-recording

Only as a deliberate, reviewed decision (the fixtures define "unchanged
behavior" for the benchmark):

```bash
<canonical-venv>/bin/python scripts/generate_native_goldens.py
```

The script re-verifies double-run byte-identity for every golden before
writing it and refuses to golden anything nondeterministic.
