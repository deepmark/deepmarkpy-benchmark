# Code Review Round 2 — POSTOJEĆI kod (prije ove grane)

Nalazi koji se odnose na kod koji je **već bio u `origin/dev` prije ove grane**. Ove probleme **nije uveo DPM-1576 PR**, ali su uočeni tokom pregleda.

Pregled obuhvata:
- `metrics.py`: stare funkcije `trim_audio_to_match`, `psnr`, `si_sdr`, `stoi_wrapper`, `pesq_wrapper`
- `report_generator.py` (basic): postojao u dev-u
- `benchmark.py`: `show_available_plugins`, `compare_watermarks`, osnovni tok `run()`
- `run.py`: osnovni argparse setup, `run_single_model` (stub)
- Testovi pisani na osnovu starog koda

---

## Sumarna tabela — POSTOJEĆI kod

| ID | Fajl | Tip | Opis | Status |
|---|---|---|---|---|
| **R2** | metrics.py | Redundancy | Dupliran `min_samples = fs // 4` check u `stoi_wrapper` i `pesq_wrapper` | ✅ Fixed |
| **R4** | metrics.py | Redundancy | `trim_audio_to_match` pozvan u svakoj metric funkciji (dupli poziv kroz `compute_metrics`) | ✅ Addressed (`_safe_metric` dekorator centralizuje trim za nove metrike; postojece funkcije zadrzavaju inline trim jer se pozivaju i direktno iz spoljnog koda) |
| **R6** | report_generator.py | Redundancy | `_display_name` dupliran (između `report_generator` i novih generatora) | ✅ Fixed |
| **R11** | benchmark.py | Redundancy | `show_available_plugins` dupliran blok for models/attacks | ✅ Fixed |
| **R13** | benchmark.py | Redundancy | `compare_watermarks` ima 5 odvojenih grana koje sve vraćaju `50.00` | ✅ Fixed |
| **C1** | metrics.py | Readability | `stoi_wrapper` 4-space višak indentacije (pogrešno formatiran) | ✅ Fixed |
| **C4** | metrics.py | Readability | `pesq_wrapper` tiho vraća None za neispravan SR, dok ostali logaju | ✅ Fixed |
| **C5** | metrics.py | Readability | Magic `1e-8` u `si_sdr` vs `_EPSILON = 1e-12` drugde — bez objašnjenja | ✅ Fixed |
| **C10** | benchmark.py | Readability | Magic `50.00` bez imenovane konstante | ✅ Fixed |
| **C12** | benchmark.py | Readability | Docstring `compute_mean_accuracy` pominje nepostojeći `confidence_threshold` | ✅ Fixed |
| **K5** | metrics.py | Style | Naming signala: `(audio1, audio2)`, `(original, watermarked)`, `(reference, estimate)`, `(reference, degraded)` — 4 konvencije | Nije diran |
| **K6** | metrics.py | Style | SR parametar: `sr` vs `fs` — dva imena | Nije diran |
| **K7** | run.py | Style | Single quotes vs double quotes pomiješane | Nije diran |
| **K8** | tests | Style | `Benchmark()` instanciran u svakoj test klasi umjesto session fixture | Nije diran |

---

## Detalji po grupama

### Metrike (`metrics.py`) — postojeći kod

**R2, R4** — dupli `trim + min_samples` checkovi:
```python
# stoi_wrapper
reference, degraded = trim_audio_to_match(reference, degraded)
min_samples = fs // 4
if len(reference) < min_samples or len(degraded) < min_samples: ...

# pesq_wrapper
reference, degraded = trim_audio_to_match(reference, degraded)
min_samples = fs // 4
if len(reference) < min_samples or len(degraded) < min_samples: ...
```

Helper:
```python
def _check_min_length(ref, deg, fs, name, min_seconds=0.25):
    ...
```

**C1** — `stoi_wrapper` ima cijelo tijelo indentovano za 4 razmaka više nego ostale funkcije. Ne utiče na funkcionalnost, vizualno odudara.

**C4** — `pesq_wrapper` na `fs not in [8000, 16000]` vraća `None` bez logger poruke. STOI, SII, MCD, NCM svi logaju kad odbiju SR.

**C5** — SI-SDR koristi `1e-8` kao epsilon (što je standardno za SI-SDR formule), ostale funkcije koriste `_EPSILON = 1e-12`. Komentar o toj razlici bi pomogao.

**K5, K6** — naming inkonzistencije potiču iz različitih autora/vremena:
- `trim_audio_to_match(audio1, audio2)` — generički imeni
- `psnr(original, watermarked)` — kontekstualni
- `si_sdr(reference, estimate)` — signal-processing konvencija
- `stoi_wrapper(reference, degraded)` — speech konvencija
- `pesq_wrapper(reference, degraded)` — speech konvencija

Preporuka: `(reference, degraded)` svuda (odgovara docstring-ovima).

Za SR: `fs` je standardni akronim u akustici (frekvencija sampling-a), `sr` je kraća varijanta iz `librosa`. Odabrati jedno i primijeniti.

---

### Basic report (`report_generator.py`)

**R6** — `_display_name` u basic generatoru radi različito od onog u novim generatorima:
```python
# basic: CamelCase → Space separated → strip
display_name = attack_name.replace("Attack", "").strip()
display_name = ''.join([' ' + c if c.isupper() and i > 0 else c for i, c in enumerate(display_name)]).strip()

# detailed, comparative: samo strip "Attack" (bez razmaka)
return attack_name.replace("Attack", "")
```

Preporuka: jednoobrazna implementacija — najbolje kroz `re.sub(r'(?<!^)(?=[A-Z])', ' ', name.replace("Attack", ""))`.

---

### Benchmark orkestracija (`benchmark.py`)

**R11** — `show_available_plugins`:
```python
for model_name, model_entry in self.models.items():
    model_cls = model_entry["class"]
    config = model_entry.get("config") or {}
    signature = inspect.signature(model_cls.__init__)
    params = [p for p in signature.parameters.values() if p.name != "self"]
    init_params = {...}
    logger.info(f"\nModel: {model_name}")
    ...

# identičan blok:
for attack_name, attack_entry in self.attacks.items():
    attack_cls = attack_entry["class"]
    ...
```

Helper:
```python
def _log_plugin(self, name, entry, kind_label):
    ...
```

**R13** — `compare_watermarks` ima 5 grana koje vraćaju `50.00`:
```python
if detected is None:
    return 50.00
if isinstance(detected, np.ndarray) and detected.ndim == 0:
    return 50.00
if isinstance(detected, (list, np.ndarray)) and len(detected) == 0:
    return 50.00
if np.any(detected == np.array(None)):
    return 50.00
if len(original) != len(detected):
    return 50.00
```

Može se skupiti u jedan guard sa helperom `_is_invalid_detection(detected, original)`.

**C10, C12** — dokumentacija i magic number:
- `50.00` u `compare_watermarks` — "random-guess baseline" komentar ili imenovana konstanta
- Docstring `compute_mean_accuracy` pominje `confidence_threshold` parametar koji je uklonjen

---

### Runner (`run.py`)

**K7** — `run.py` kombinuje single i double quotes (`'File not found'` pored `"Audio directory"`). `black` ili `ruff format` riješio bi.

---

### Testovi

**K8** — Svaka test klasa u `test_benchmark.py` poziva `self.bench = Benchmark()` kroz `autouse=True` fixture:
```python
class TestCompareWatermarks:
    @pytest.fixture(autouse=True)
    def _create_benchmark(self):
        self.bench = Benchmark()
```

`Benchmark()` poziva `PluginManager()` koji skenira sve plugin-e (spor). Session-scoped fixture u `conftest.py` ubrzao bi suite:
```python
# conftest.py
@pytest.fixture(scope="session")
def benchmark_instance():
    return Benchmark()
```

---

## Preporučeni redoslijed (POSTOJEĆI nalazi)

Ovi problemi **nisu prioritet za ovu granu** — treba ih uvrstiti u zasebne tickete kao tehnički dug.

**Nisko — kad bude dostupno vremena:**
1. C1 (formatiranje `stoi_wrapper` — pokreni `black`)
2. C10 + C12 (imenovana konstanta + docstring fix)
3. K7 (quote style — `black`)
4. K8 (session fixture za `Benchmark`)

**Srednje:**
5. R11 (`show_available_plugins` helper)
6. R13 (`compare_watermarks` guard)
7. C4, C5 (SR logging + epsilon komentar)
8. K5, K6 (naming harmonizacija)
9. R2, R4 (min_length helper + dupli trim)

**Veće:**
10. R6 (ujednači `_display_name` — ali ovo se rješava zajedno sa novim modulom `latex_helpers.py` iz *NEW_CODE* review-a)

---

*Generisan: 2026-04-29.*
