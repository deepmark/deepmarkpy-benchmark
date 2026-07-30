# Consuming deepmarkpy plugin engines

The private consumer installs the package at a pinned tag and wraps each
plugin's `Engine` in its own serving layer and images. It never imports this
repo's `app.py` and never depends on the HTTP layer.

```bash
pip install "deepmarkpy @ git+https://github.com/deepmark/deepmarkpy-benchmark@v1.0.0"
```

```python
from deepmarkpy.plugins.attacks.vae.inference import Engine
engine = Engine(config)                       # config: the plugin's config.json dict
out = engine.apply(audio, sampling_rate)      # attacks
# models: engine.embed(audio, watermark_data, sampling_rate) / engine.detect(audio, sampling_rate)
```

## Stable API (from v1.0.0)

- Module paths `deepmarkpy.plugins.{attacks,models}.<name>.inference` and the
  `Engine` signatures (`__init__(config, device=None)`; attacks `apply`,
  models `embed`/`detect`) are semver-stable. Anything else in plugin
  directories is internal.
- Each plugin directory ships its `config.json`, `requirements.txt`, captured
  `constraints*.txt` (the exact dependency set of this repo's verified
  images), and reference `Dockerfile` as package data — use them as the
  ground truth for building your own images.
- `Engine.__init__` loads weights (except `speech_enhancement_2`, which
  constructs ClearVoice per `apply` call — an intentional, frozen behavior).
- Engines are deterministic per identical input except `vae`, `diffusion`,
  and `speech_enhancement_2` (see `tests/fixtures/contracts/README.md`), and
  a number of behavior quirks are preserved intentionally — do not "fix" them downstream if you need benchmark parity.

## Per-plugin build requirements

Baseline for every torch plugin: Python 3.11 (slim), `libsndfile1` +
`libgomp1` apt packages, `pip install deepmarkpy` (`--no-deps` if you manage
dependencies via the shipped constraint files), then the plugin's
`requirements.txt` constrained by its `constraints.txt`. The torch version
each image resolved is authoritative in its constraints file.

### Models

| Plugin | torch | Weights | Notes |
|---|---|---|---|
| `audio_seal` | 2.7.1 | downloaded at startup (`audioseal_wm_16bits`, `audioseal_detector_16bits`) | — |
| `aware` | 2.7.0 | loaded by the `aware` package | clone `github.com/deepmarkpy/aware` @ `fea9c49e3dfc57a421705ae411adedd80bcc6d09`, `pip install -e .` **constrained by `constraints.stage1.txt`**, then `requirements.txt` with `constraints.txt` — the two-step order matters (the editable install resolves aware's own exact pins, e.g. `pydantic==2.5.0`, which the second step upgrades). Needs `git` at build. |
| `perth` | 2.7.1 | bundled in the `resemble-perth` wheel | fully pinned `requirements.txt` |
| `silent_cipher` | 2.0.0 | 44.1k checkpoint downloaded at startup | config declares 16 kHz while the 44.1k model loads (intentional, preserved behavior). `numpy<2` per its requirements. |
| `timbrewm` | 2.0.0 | checkpoints inside the upstream clone | clone `github.com/TimbreWatermarking/TimbreWatermarking` @ `c41e7d75637f162d462ef2159acc5149b6c8071a`; then apply the upstream adaptations exactly as in the shipped `Dockerfile` (normative): rename `watermarking_model/model` → `watermarking_model/wm_model`, then the five `sed` edits (relative-import fix in `mel_transform.py`, torchaudio-import removal and hifigan config/checkpoint path + CPU-map fixes in `conv2_mel_modules.py`). Serve from a working directory containing the `TimbreWatermarking/` clone. |
| `wavmark` | 2.7.1 | downloaded from Hugging Face at startup | — |

### Attacks

| Plugin | torch | Weights | Notes |
|---|---|---|---|
| `vae` | 2.7.1 | `Intelligent-Instruments-Lab/rave-models` from HF at startup (model per `model_name_vae`) | stochastic (RAVE samples latents) |
| `diffusion` | 2.7.1 | `teticio/audio-diffusion-256` from HF at startup | stochastic (fresh OS-entropy seed per call); keep torchaudio matched to torch (an unmatched torchaudio wheel can dlopen CUDA on CPU-only hosts) |
| `neural_vocoder` | 2.7.1 | BigVGAN checkpoint from HF at startup (per `model_name_neural_vocoder`) | clone `github.com/NVIDIA/BigVGAN` @ `7d2b454564a6c7d014227f635b7423881f14bdac`; install `big_vgan_requirements.txt` (shipped) constrained by `constraints.builder.txt`; run with the clone directory as the working directory (`bigvgan`/`meldataset` import from cwd) |
| `speech_enhancement_1` | 2.7.1 | SpeechBrain models downloaded at startup (`mtl-mimic-voicebank` or `metricgan-plus-voicebank` per `type_se1`) | CPU wheels via the extra index URL in its `requirements.txt` |
| `speech_enhancement_2` | 2.13.0 | ClearVoice weights downloaded on first request | ClearVoice is constructed **per request** (frozen behavior); its env resolved torch 2.13.0 — do not force-match the 2.7.1 baseline |
| `speech_tokenization` | 2.4.1 | `HKUST-Audio/xcodec2` — bake at build (as the shipped Dockerfile does) or accept a slow first start | install order matters: `requirements.txt` (with constraints), then `xcodec2==0.1.3 --no-deps`, then `xcodec_requirements.txt` (with constraints) |
| `opus_codec` | none | none | apt: `opus-tools`, `libopus0`, `libsndfile1`; pure subprocess round-trip, works on `python:3.10-slim` |
| `encodec` | 2.7.1 | `encodec_24khz` downloaded at startup | `encodec==0.1.1` |
| `descript_audio_codec` | 2.7.1 | DAC `44khz` model downloaded at startup | `descript-audio-codec==1.0.0`; pulls `matplotlib` and friends transitively |

Licensing: each upstream model/repo carries its own license (Meta AudioSeal,
NVIDIA BigVGAN, TimbreWatermarking, Resemble Perth, SpeechBrain, ClearVoice,
Descript, upstream RAVE exports); review them before redistribution of
images or weights.

## Consumer validation exercise (for the first consumer)

The first consumer should validate the contract by
building two images **from the installed wheel and this document alone**
(no checkout of this repo):

1. one model image: `audio_seal` — serve
   `deepmarkpy.plugins.models.audio_seal.inference.Engine` behind your own
   transport, embed and detect a 1 s, 16 kHz test signal;
2. one attack image: `vae` — serve
   `deepmarkpy.plugins.attacks.vae.inference.Engine`, apply it to the same
   signal.

Any friction (missing package data, undocumented system dependency,
incorrect pin) is a bug in this document or the package data — report it
against this repo.
