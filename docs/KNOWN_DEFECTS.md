# Known-defect register (frozen)

Status: **documented, frozen, deferred** — per `docs/REORG_PLAN.md` §4
(decided 2026-07-29). Every entry below is a real defect that is
**deliberately not fixed** during the packaging & plugin reorganization
effort. The effort's guarantee is bit-for-bit behavior preservation —
including behaviors that are wrong. Fixing, improving, or "tidying" any entry
below (or anything of the same character: adding timeouts or
`raise_for_status` to existing clients, changing error shapes or message
text, fixing kwarg/sample-rate mismatches, removing dead code, normalizing
logging, hoisting per-request instantiation, seeding stochastic paths)
breaks that guarantee and is out of scope. **Target release for all entries:
the deferred-fix release, scoped separately after the v1.0.0 packaging
release** (REORG_PLAN §P3.3).

If preserving an entry exactly ever conflicts with an assigned extraction,
stop and escalate per `docs/AGENT_PROMPT.md` — do not resolve the conflict
yourself.

| # | Symptom | Location | Why frozen |
|---|---------|----------|-----------|
| D1 | `mixing` passes `gains=` where `EqualizerAttack` reads `gains_equalizer` → configured gains silently ignored | `src/deepmarkpy/plugins/attacks/mixing/attack.py:242` | Fixing changes attacked audio, hence accuracy numbers |
| D2 | `inverted_time_stretch` passes `stretch_rate=` where `TimeStretchAttack` reads `stretch_rate_time_stretch` → configured value ignored | `src/deepmarkpy/plugins/attacks/inverted_time_stretch/attack.py:43,48` | Fixing changes attacked audio, hence accuracy numbers |
| D3 | `silent_cipher` config declares 16 kHz while app.py loads the 44.1k checkpoint | `src/deepmarkpy/plugins/models/silent_cipher/inference.py:35` | Config values are a published red line; runtime behavior must not change |
| D4 | `perth` packs watermark bits then passes `watermark=None` (payload discarded). **Never deletable as "dead code"** | `src/deepmarkpy/plugins/models/perth/inference.py:44-48` | The packing call sequence is preserved verbatim in the P1 extraction; removing it is a behavior change in disguise |
| D5 | `speech_enhancement_2` returns HTTP 200 with `{"error": …, "audio": null}`; client guard passes → 0-d object array downstream | `src/deepmarkpy/plugins/attacks/speech_enhancement_2/app.py:38`, `attack.py:39-44` | Error-in-200 shape is observable client behavior; must survive extraction byte-identically |
| D6 | `aware` returns errors as HTTP-200 bodies | `src/deepmarkpy/plugins/models/aware/app.py:58,76` | Same as D5 |
| D7 | Five dockerized attack clients (`vae`, `diffusion`, `neural_vocoder`, `speech_enhancement_1`, `speech_enhancement_2`) use bare `requests.post` with no timeout and no `raise_for_status`; `BaseAttack` lacks a `_make_request` helper. (`opus_codec` 120 s, `network_transmission` 180 s, `speech_tokenization`, `encodec`, and `descript_audio_codec` 600 s already have both — two co-existing client patterns) | respective `attack.py` files | Adding timeouts/status checks changes failure behavior (hang vs raise) |
| D8 | `audio_seal` `detect` annotated `-> np.ndarray` but returns a tuple | `src/deepmarkpy/plugins/models/audio_seal/model.py:26,50` | Annotation-only today, but callers dispatch on the tuple; freeze until the deferred pass |
| D9 | `speech_tokenization` resamples redundantly in both app.py and xcodec.py | `src/deepmarkpy/plugins/attacks/speech_tokenization/app.py:48`, `inference.py:30` | Removing a resample round-trip changes audio numerics |
| D10 | `silent_cipher` bare `except:` swallows decode failures → `null` | `src/deepmarkpy/plugins/models/silent_cipher/inference.py:66` | Error shape is observable behavior |
| D11 | `resample_audio` fed raw list vs ndarray inconsistently, even within one file | `src/deepmarkpy/plugins/models/audio_seal/inference.py:40,67`; `wavmark/inference.py:42 (embed raw list) vs detect ndarray` | Normalizing dtypes can change numerics; preserved verbatim in extraction |
| D12 | Port defaults triplicated (.env.example / Dockerfile ENV / client fallback); `.env` never loaded by Python | repo-wide | De-duplication touches user-visible configuration surface |
| D13 | `timbrewm` requirements downgrade base torch 2.7.1→2.0.0; `speech_tokenization` →2.4.1 | per-plugin `requirements.txt` | Dependency versions are frozen in place (P0.2); changing them changes numerics |
| D14 | `pywt`, `pyrubberband` missing from `requirements.txt` → `wavelet`, `pitch_shift`, `time_stretch` silently absent on fresh installs (paper promises 40 attacks). **Empirical amendment (2026-07-29): `inverted_time_stretch` also vanishes** — it imports `plugins.attacks.time_stretch.attack`, so four attack classes are absent in the canonical environment, not three (see `tests/test_discovery_lock.py`) | `requirements.txt` | Adding dependencies changes the canonical environment and the locked discovery sets mid-effort; deferred per REORG_PLAN §8/D5 |
| D15 | Requested-but-missing attacks are warn-and-skipped (exit 0, results silently incomplete); only missing models raise | `src/deepmarkpy/benchmark.py:368` vs `:122` | Exit-code and log behavior are user-visible; raising is a behavior change |
| D16 | `Codec2VocoderAttack` output depends on in-process invocation history: pycodec2 carries C encoder state across instantiations within one process (fresh-process first call is reproducible; same-process re-runs differ on ~99% of samples), so benchmark codec2 results depend on how many codec2 invocations preceded them. Discovered 2026-07-29 during P0.3; excluded from goldens | `src/deepmarkpy/plugins/attacks/codec2_vocoder/attack.py:47` (pycodec2 usage) | Isolating/resetting the codec per call changes attacked audio and accuracy numbers |

Locations re-verified 2026-07-30 against the packaged tree (P3.3 close-out);
re-verify line numbers before relying on them in future PRs.
