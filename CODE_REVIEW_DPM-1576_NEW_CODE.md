# Code Review Round 2 — NOVE promjene na grani DPM-1576

Nalazi koji se **odnose na kod napisan u ovoj grani** (`DPM-1576-add-metrics-to-benchmark` vs `origin/dev`).

Pregled obuhvata:
- Novi fajlovi: `attack_groups.py`, `detailed_report_generator.py`, `comparative_report_generator.py`
- Novi dijelovi u `metrics.py`: `visqol_wrapper`, `mcd`, `sii`, `ncm`, `_ansi_bands`, `compute_metrics`, ANSI konstante, magic-number konstante
- Prepisani dijelovi `benchmark.py`: `run()` tok sa metrikama, `compute_mean_accuracy`, nova struktura rezultata (`results[fajl]["attacks"]`)
- Prepisani dijelovi `run.py`: multi-model mod, attack groups, `to_json_safe`, `run_single_model`/`run_multiple_models`

---

## Sumarna tabela — NOVE promjene

| ID | Fajl | Tip | Opis |
|---|---|---|---|
| **R1** | metrics.py | Redundancy | Dupliran skelet SII/NCM (STFT + band loop) |
| **R3** | metrics.py | Redundancy | `try/except (RuntimeError, ValueError, IndexError)` + `logger.warning` dupliran 6× |
| **R5** | ×3 generatora | Redundancy | `_preamble` dupliran 3× (sa minimalnim razlikama) |
| **R7** | ×3 generatora | Redundancy | Longtable boilerplate kopiran 5× |
| **R8** | ×3 generatora | Redundancy | Pdflatex kompilacija duplirana 3× |
| **R9** | detailed + comparative | Redundancy | `QUALITY_METRICS / INTELLIGIBILITY_METRICS / METRIC_LABELS` duplirane |
| **R10** | detailed + comparative | Redundancy | `aggregate_results` / `aggregate_metrics` prolaze istu strukturu sa istim filterom |
| **R12** | benchmark.py | Redundancy | Accuracy dispatch (zero_bit/confidence/regular) pokrenut za glavni + cross-model odvojeno |
| **R14** | attack_groups.py | Redundancy | `_QUALITY_METRICS_BY_GROUP` / `_INTELLIGIBILITY_METRICS_BY_GROUP` dict-ovi dupliraju ono što završi u `ATTACK_GROUPS` |
| **R15** | run.py | Redundancy | `args_dict = vars(args)` u `main()` nikada ne čita se |
| **C2** | metrics.py | Readability | Netačan return hint `-> float` na `visqol_wrapper / sii / mcd / ncm` — sve mogu vratiti `None` |
| **C3** | metrics.py | Readability | `pesq_wrapper(mode=...)` parametar izložen ali nikad iskorišten iz `compute_metrics` |
| **C6** | report_generator.py | Readability | F-string sa indentacijom ide direktno u `.tex` (preamble, LaTeX body) |
| **C7** | comparative | Readability | `create_radar_chart` 95 linija — treba podjela |
| **C8** | detailed | Readability | `_metrics_table` ima međusobno zavisne parametre (`include_baseline` + `baseline_data`) |
| **C9** | benchmark.py | Readability | `Benchmark.run()` 200 linija — pre-dugo, treba dekompozicija |
| **C11** | benchmark.py | Readability | Naziv `different_model_*` je nejasan — bolje `cross_model_*` |
| **K1** | benchmark.py | Style | `if (attack_name =="CrossModelAttack"):` — PEP8 (zagrade, razmak, elif) |
| **K2** | ×3 generatora | Style | Naming nekonzistentan: `BenchmarkReportGenerator` vs `DetailedReportGenerator` vs `ComparativeReportGenerator` |
| **K3** | ×3 generatora | Style | `generate_full_report` vraća različite oblike `(tuple/None/scalar)` |
| **K4** | ×3 generatora | Style | `report_dir` default različit: `"report"` vs `"results/comparison"` |

---

## Detalji najvažnijih NOVIH problema

### R3+C3 — dekorator za metrike (NOVI kod)

Skoro svi novi wrappers (`visqol_wrapper`, `sii`, `mcd`, `ncm`) imaju identičan patern:
```python
def X_wrapper(reference, degraded, fs):
    reference, degraded = trim_audio_to_match(reference, degraded)
    try:
        ... actual work ...
    except (RuntimeError, ValueError, IndexError) as e:
        logger.warning(f"X calculation failed: {e}")
        return None
```

Dekorator:
```python
def _safe_metric(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(reference, degraded, *args, **kwargs):
            reference, degraded = trim_audio_to_match(reference, degraded)
            try:
                return fn(reference, degraded, *args, **kwargs)
            except (RuntimeError, ValueError, IndexError) as e:
                logger.warning(f"{name} calculation failed: {e}")
                return None
        return wrapper
    return deco
```

---

### R5-R8 — `utils/latex_helpers.py` (NOVI kod)

Svi report generatori su novi (detailed, comparative), pa je ovo direktno problem grane. Kreirati zajednički modul:
```python
def make_preamble(title, author, extra_packages=()): ...
def display_attack_name(raw): ...
def build_longtable(col_spec, header, rows, caption, label): ...
def compile_latex(report_dir, tex_basename): ...
```

Skida ~200 linija duplikacije iz `detailed_report_generator.py` i `comparative_report_generator.py`.

---

### R9 — metric konstante (NOVI kod)

Obje dupli­raju iste liste:
```python
# detailed_report_generator.py
QUALITY_METRICS = ["pesq", "psnr", "si_sdr", "mcd", "visqol"]
INTELLIGIBILITY_METRICS = ["stoi", "sii", "ncm"]
METRIC_LABELS = {...}

# comparative_report_generator.py
QUALITY_METRICS = ["pesq", "psnr", "si_sdr", "mcd", "visqol"]
INTELLIGIBILITY_METRICS = ["stoi", "sii", "ncm"]
METRIC_LABELS = {...}
```

`ALL_METRICS` već postoji u `metrics.py`. Dodati tamo i `QUALITY_METRICS / INTELLIGIBILITY_METRICS / METRIC_LABELS`, oba generatora importuju.

---

### R12 — cross-model accuracy dispatch (NOVI način, stara logika)

Glavni accuracy dispatch je postojao u dev-u, ali cross-model grana je nova:
```python
# glavni model (benchmark.py:289) — postojao u dev-u
if is_zero_bit: ...
elif returns_confidence: ...
else: accuracy = self.compare_watermarks(...)

# different model (benchmark.py:263) — NOVO u grani
if diff_is_zero_bit: ...
elif diff_returns_confidence: ...
else: ...
```

Helper:
```python
def _accuracy_from_detection(detected, watermark, is_zero_bit, returns_confidence):
    if is_zero_bit:
        return detected.tolist() if isinstance(detected, np.ndarray) else detected
    if returns_confidence:
        detected, _ = detected
    return self.compare_watermarks(watermark, detected)
```

---

### R14 — privatne metric mape (NOVI kod)

`attack_groups.py:13-30` (moja izmjena iz S4/D3 fix-a):
```python
_QUALITY_METRICS_BY_GROUP = {...}
_INTELLIGIBILITY_METRICS_BY_GROUP = {...}

ATTACK_GROUPS = {
    "process_disruption": {
        ...,
        "quality_metrics": _QUALITY_METRICS_BY_GROUP["process_disruption"],
        "intelligibility_metrics": _INTELLIGIBILITY_METRICS_BY_GROUP["process_disruption"],
    },
    ...
}
```

Module-level private dict-ovi su nepotrebni — vrijednosti se samo prepisuju u `ATTACK_GROUPS`. Rješenje: ukloniti oba dict-a, inline vrijednosti direktno u `ATTACK_GROUPS`.

---

### C9 — `Benchmark.run()` dekompozicija (NOVO tijelo)

Tijelo `run()` je prepisano u ovoj grani (calculate_quality_metrics, cross-model accuracy dispatch, nova rezultat struktura). Trenutno 200 linija, previše grananja.

```python
def run(self, ...):
    self._init_model_state(...)
    for filepath in filepaths:
        results[filepath] = self._run_file(filepath, ...)
    return results

def _run_file(self, filepath, model_instance, attack_types, ...):
    audio, watermarked = self._embed(filepath, model_instance)
    file_data = {"attacks": {}}
    if calculate_quality_metrics:
        file_data["watermarked_audio_quality"] = compute_metrics(...)
    for attack_name in attack_types:
        file_data["attacks"][attack_name] = self._run_attack(...)
    return file_data
```

---

## Preporučeni redoslijed (NOVI nalazi)

**Brze ispravke:**
1. R15 (obriši `args_dict` u main)
2. C3 (ukloni `mode` parametar iz `pesq_wrapper` ili izloži kroz compute_metrics)
3. C2 (`Optional[float]` u return hint-ima novih wrappera)
4. K1 (PEP8 `CrossModelAttack` grane)
5. R14 (ukloni privatne metric mape u `attack_groups.py`)
6. C11 (preimenuj `different_model_*` → `cross_model_*`)

**Refaktoringi:**
7. R3+R4 (dekorator `_safe_metric`)
8. R5-R8 (modul `latex_helpers.py`)
9. R9 (centralizuj metric konstante u `metrics.py`)
10. R12 (helper `_accuracy_from_detection`)
11. C9 (razbij `Benchmark.run()`)

**Stilske:**
12. K2, K3, K4 (unifikuj naming i API report generatora)

---

*Generisan: 2026-04-29.*
