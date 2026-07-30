# HTTP-contract freeze fixtures (P0.4)

Recorded and verified by `scripts/contract_check.py` (see its docstring for
the request-construction rules and verification criteria) against the
post-P0.2 frozen images, per REORG_PLAN.md §4.3/§6 P0.4. One directory per
compose service: `contract.json` (endpoint metadata, raw-response SHA-256,
double-call evidence, classification) + `arrays.npz` (request/response
audio and watermark arrays).

## Designated machine (§4.3)

macOS arm64 (Apple Silicon; Darwin 25.3.0), Docker Desktop, images built
2026-07-30 from the P0.2-pinned Dockerfiles with the committed per-image
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
| speech_enhancement1 | attack | deterministic | byte-identical double call **with `noise_strength=0.0` in the request** (§4.3 prescription; default 0.01 is stochastic) |
| vae | attack | **stochastic** | double call differs (same length 14336, RMS 0.320 vs 0.460) — the RAVE export samples latents internally. §4.3 anticipated this had to come from the double-call check. **P1.1 consequence: the extraction template plugin is `wavmark`, not `vae`.** |
| speech_enhancement2 | attack | stochastic | double call differs — unseeded server-side noise (`noise_strength_se2=0.01` from config, not request-overridable) |
| network_transmission | attack | stochastic | same-container double call was byte-identical, but a fresh-container call differs (timing/instance-dependent RTP path) — classification forced stochastic with the evidence recorded in `contract.json`. `tc netem` itself does not engage under Docker Desktop's VM (`tc qdisc show` stays `noqueue`). |
| diffusion | attack | stochastic | double call differs (RMS 0.222 vs 0.243, same length) — `ddpm.py:27-28` draws a fresh OS-entropy seed per request. Recording initially blocked by a frozen `torchaudio==2.11.0`/`torch==2.7.1` mismatch in the base lineage (torchaudio 2.11's aarch64 wheel dlopens `libcudart.so.13`); fixed 2026-07-30 with owner approval by repinning torchaudio 2.7.1 in the torch-2.7.1 constraint files and rebuilding (`speech_enhancement2` keeps its historical, runtime-inert 2.11.0 — its env froze at torch 2.13.0). |

## Re-recording / verifying

```bash
python scripts/contract_check.py verify           # all recorded services
python scripts/contract_check.py record <svc>     # re-record (owner sign-off)
```

Deterministic services must return the exact recorded raw body (SHA-256);
stochastic services are checked structurally (keys, finite values, length
range, RMS tolerance).
