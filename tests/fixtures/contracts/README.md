# HTTP-contract freeze fixtures

Recorded and verified by `scripts/contract_check.py` (see its docstring for
the request-construction rules and verification criteria) against the
frozen images. One directory per
compose service: `contract.json` (endpoint metadata, raw-response SHA-256,
double-call evidence, classification) + `arrays.npz` (request/response
audio and watermark arrays).

## Designated machine

macOS arm64 (Apple Silicon; Darwin 25.3.0), Docker Desktop, images built
2026-07-30 from the pinned Dockerfiles with the committed per-image
constraint files. Byte-identity claims are same-machine, same-image-lineage
only.

## Classification (double-call recorded 2026-07-30)

| Service | Kind | Classification | Basis |
|---|---|---|---|
| audioseal | model | deterministic | byte-identical double call (embed + detect) |
| aware | model | deterministic | byte-identical double call |
| perth | model | deterministic | byte-identical double call |
| silentcipher | model | deterministic | byte-identical double call |
| timbrewm | model | deterministic | byte-identical double call |
| wavmark | model | deterministic | byte-identical double call (2 s input — 1 s yields zero usable chunks in `wavmark.encode_watermark`) |
| neural_vocoder | attack | deterministic | byte-identical double call |
| encodec | attack | deterministic | byte-identical double call |
| descript_audio_codec | attack | deterministic | byte-identical double call |
| opus_codec | attack | deterministic | byte-identical double call |
| speech_tokenization | attack | deterministic | byte-identical double call |
| speech_enhancement1 | attack | deterministic | byte-identical double call **with `noise_strength=0.0` in the request** (the request-controlled noise default 0.01 is stochastic) |
| vae | attack | **stochastic** | double call differs (same length 14336, RMS 0.320 vs 0.460) — the RAVE export samples latents internally. The double-call check, not assumption, settled this. |
| speech_enhancement2 | attack | stochastic | double call differs — unseeded server-side noise (`noise_strength_se2=0.01` from config, not request-overridable) |
| network_transmission | attack | stochastic | same-container double call was byte-identical, but a fresh-container call differs (timing/instance-dependent RTP path) — classification forced stochastic with the evidence recorded in `contract.json`. `tc netem` does engage here: the qdisc is installed per request and torn down in a `finally`, so it exists only for the ~1.5 s a request takes — polling `tc qdisc show dev lo` at idle shows `noqueue` and says nothing about whether impairment ran. The response now carries `netem_active` so a run does not have to infer it. |
| diffusion | attack | stochastic | double call differs (RMS 0.222 vs 0.243, same length) — `inference.py` draws a fresh OS-entropy seed per request. `speech_enhancement2`'s environment intentionally keeps its own torchaudio pin (its env froze at torch 2.13.0). |

## Re-recording / verifying

```bash
python scripts/contract_check.py verify           # all recorded services
python scripts/contract_check.py record <svc>     # re-record (deliberate, reviewed decision)
```

Deterministic services must return the exact recorded raw body (SHA-256);
stochastic services are checked structurally (keys, finite values, length
range, RMS tolerance).
