# Consuming deepmarkpy plugin engines (skeleton — completed in P3)

The private consumer installs the package at a pinned tag and wraps each
plugin's `Engine` in its own serving layer and images:

```bash
pip install "deepmarkpy @ git+https://github.com/deepmark/deepmarkpy-benchmark@v1.0.0"
```

```python
from deepmarkpy.plugins.attacks.vae.inference import Engine
```

## Stable API (from v1.0.0)

- Module paths `deepmarkpy.plugins.{attacks,models}.<name>.inference` and the
  `Engine` signatures (`__init__(config, device=None)`; attacks `apply`,
  models `embed`/`detect`) are semver-stable. Anything else in plugin
  directories is internal.
- The consumable set (15): models `audio_seal`, `aware`, `perth`,
  `silent_cipher`, `timbrewm`, `wavmark`; attacks `vae`, `diffusion`,
  `neural_vocoder`, `speech_enhancement_1`, `speech_enhancement_2`,
  `speech_tokenization`, `opus_codec`, `encodec`, `descript_audio_codec`.
- Each plugin directory ships its `config.json`, `requirements.txt`,
  captured `constraints*.txt`, and reference `Dockerfile` as package data.

## Per-plugin build requirements

To be completed in P3.1: Python/torch constraints, system packages
(e.g. `opus-tools`), upstream clone pins and patches, weights acquisition,
env vars, licensing notes — per consumable plugin.
