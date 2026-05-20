# Code Review: DPM-1576-add-metrics-to-benchmark — finalna iteracija

**Grana:** `DPM-1576-add-metrics-to-benchmark` (HEAD: `c68498c Resolve bugs`)
**Poredjeno sa:** `origin/main`
**Obim:** 16 fajlova izmijenjeno, +2326 / −282 linija
**Datum:** 2026-05-19

Ovaj dokument:
1. provjerava status nalaza iz prethodna tri review-a (`CODE_REVIEW_DPM-1576.md`, `CODE_REVIEW_DPM-1576_NEW_CODE.md`, `CODE_REVIEW_DPM-1576_EXISTING_CODE.md`),
2. nabraja nalaze koji **i dalje stoje**,
3. dokumentuje **nove probleme** uocene u trenutnom snapshot-u koda.

Oznake:
- ✅ Fixed — problem otklonjen u trenutnom stanju
- ⚠️ Partial — djelimicno rijeseno
- ❌ Open — i dalje prisutan
- 🆕 New — nije bio u prethodnim review-ima

---

## Sadrzaj

1. [Status ranijih kriticnih nalaza](#1-status-ranijih-kritičnih-nalaza)
2. [Status ostalih ranijih nalaza](#2-status-ostalih-ranijih-nalaza)
3. [Otvoreni problemi (i dalje stoje)](#3-otvoreni-problemi-i-dalje-stoje)
4. [Novi nalazi](#4-novi-nalazi)
5. [Sumarna tabela svih otvorenih nalaza](#5-sumarna-tabela-svih-otvorenih-nalaza)
6. [Preporuceni redoslijed rjesavanja](#6-preporuceni-redoslijed-rjesavanja)

---

## 1. Status ranijih kriticnih nalaza

| ID | Opis | Status | Komentar |
|---|---|---|---|
| **K1** | SII FFT samo prvih 2048 uzoraka | ✅ Fixed | `sii` koristi `librosa.stft` sa `_SPECTRAL_N_FFT=2048` i hop, prosjekuje po vremenskim okvirima ([metrics.py:318-325](src/utils/metrics.py#L318-L325)). |
| **K2** | SII pogresna definicija suma | ✅ Fixed | Sum se sad racuna iz `noise_signal = degraded - reference` pa STFT ([metrics.py:322-325](src/utils/metrics.py#L322-L325)). |
| **K3** | SNR uklonjen bez zamjene | ❌ Open | SNR se i dalje **nigdje ne racuna** — funkcija `snr` postoji u [utils/utils.py:65](src/utils/utils.py#L65) ali nije pozvana iz `compute_metrics` niti `benchmark.py`. Detalji u sekciji 3. |
| **K4** | `to_json_safe` `None → "N/A"` razbija JSON round-trip | ⚠️ Partial | Dodat je `from_json_safe` ([run.py:41-52](src/run.py#L41-L52)) kao "inverz", ali izvorni problem ostaje: JSON fajlove citaju i alati van ovog projekta (pandas, jq, drugi Python skripti) koji ne znaju za `"N/A"` sentinel. Kompromis je krhka konvencija; cisto rjesenje je vratiti `null`. Detalji u sekciji 3 (problem **O1**). |

---

## 2. Status ostalih ranijih nalaza

### Visok prioritet

| ID | Opis | Status |
|---|---|---|
| **V1** | `VisqolApi` instancira po pozivu | ✅ Fixed (cache po modu, [metrics.py:238-269](src/utils/metrics.py#L238-L269)) |
| **V2** | PESQ za `process_disruption` | ✅ Fixed (`quality_metrics: []` za grupu, [attack_groups.py:22-23](src/utils/attack_groups.py#L22-L23)) |
| **V4** | MCD bez standardnog scaling faktora | ✅ Fixed (`_MCD_FACTOR ≈ 6.1413` + iskljucen MFCC[0], [metrics.py:348-379](src/utils/metrics.py#L348-L379)) |
| **V5** | Help tekst `--calculate_quality_metrics` zastareo | ✅ Fixed ([run.py:114-118](src/run.py#L114-L118)) |
| **V6** | `except Exception` preteoko | ⚠️ Partial — u `metrics.py` sve metrike koriste `(RuntimeError, ValueError, IndexError)`, ali u `run.py` su jos uvijek prisutne sirine: [run.py:173](src/run.py#L173), [run.py:253-254](src/run.py#L253-L254), [run.py:264-265](src/run.py#L264-L265), [run.py:337-338](src/run.py#L337-L338). |
| **V7** | ViSQOL top-level import | ✅ Fixed (lazy import + cache za "unavailable" stanje) |

### Srednji prioritet

| ID | Opis | Status |
|---|---|---|
| **S1** | Uniformne tezine vs ANSI | ✅ Fixed (`_ANSI_BAND_IMPORTANCE`, normalizovano nakon Nyquist filtera) |
| **S2** | Razliciti freq opsezi SII vs NCM | ✅ Fixed (`_ansi_bands(fs)` zajednicki) |
| **S3** | SII/NCM forsirano resamp na 16 kHz | ✅ Fixed — samo PESQ/STOI sada resamp ([metrics.py:494-503](src/utils/metrics.py#L494-L503)) |
| **S4** | Racuna se 8 metrika, prikazuje manje | ✅ Fixed (`compute_metrics(metrics=...)` i `get_metrics_for_attack`) |
| **S5** | `watermarked_audio_quality` duplirano | ✅ Fixed (file-level u [benchmark.py:154](src/benchmark.py#L154)) |
| **S6** | Semanticka promjena nije dokumentovana | ✅ Fixed (docstring `compute_metrics` + README "What is measured") |
| **S7** | Multi-model pad nije handlovan | ⚠️ Partial — postoji try/except, ali samo za `(MemoryError, ConnectionError, OSError)` ([run.py:291](src/run.py#L291)). U praksi modeli najcesce padaju sa `RuntimeError` (PyTorch CUDA OOM), `subprocess.CalledProcessError`, `requests.RequestException`, `httpx.HTTPError` — svi prolaze i obaraju cijeli run. Detalji u sekciji 4 (**N5**). |

### Nizak prioritet (stari M-ovi)

Skoro sve sitnice iz prvog review-a su rijesene:

| ID | Opis | Status |
|---|---|---|
| M1 | NCM clip(0,1) | ✅ Resen — koristi `np.abs(...)` |
| M2 | Magic numbers | ✅ Resen — sve konstante imenovane u modulu |
| M3 | `compute_metrics` bez type hints | ✅ Resen |
| M4 | "visqol (optional)" docstring | ✅ Resen |
| M5 | MCD bez DTW | ✅ Resen — eksplicitno u docstring-u |
| M6 | `create_radar_chart` dead code | ✅ Resen — pozvan u comparative report |
| M7 | `total_attacks` parametar nekoriscen | ✅ Resen — argument uklonjen |
| M8 | Hardkodirana imena bez validacije | ✅ Resen — [test_attack_groups.py:22-27](tests/test_attack_groups.py#L22-L27) |
| M9 | `group_attacks` shadowing | ✅ Resen — preimenovano u `attacks_from_groups` |
| M11 | `requirements.txt` trailing newline | ❌ Open — i dalje nedostaje (provjereno na kraju fajla) |
| M12 | Imena modela u README primjerima | ✅ OK — `AwareModel`, `PerthModel`, `AudioSealModel` postoje |
| M13 | ViSQOL 0.0.4 | ✅ Resen — `visqol-python==3.4.0` |
| M14 | `.gitignore` bez komentara | ✅ Resen ([.gitignore:38-40](.gitignore#L38-L40)) |

### Stari "EXISTING_CODE" (R/C/K)

Sve duplikacije (R2, R4, R6, R11, R13) i citljivosti (C1, C4, C5, C10, C12) su rijesene preko helper-a (`_check_min_length`, `_log_plugin_entry`, `_is_invalid_detection`, `RANDOM_GUESS_ACCURACY`, `display_attack_name`).

K5 (naming `audio1`/`reference`/...), K6 (`sr` vs `fs`), K7 (quote style), K8 (Benchmark per-test fixture) — i dalje otvoreni, nizak prioritet, vidi sekciju 3.

### Stari "NEW_CODE" (R/C/K)

| ID | Opis | Status |
|---|---|---|
| **R3** | `_safe_metric` dekorator | ✅ Implementiran |
| **R5–R8** | LaTeX duplikacije | ✅ `latex_helpers.py` izvuden, koriste ga `detailed` i `comparative` generatori |
| **R9** | Metric konstante duplirane | ✅ Centralizovane u `metrics.py` |
| **R12** | Cross-model accuracy dispatch | ❌ Open — dva odvojena bloka u [benchmark.py:243-257](src/benchmark.py#L243-L257) i [benchmark.py:265-271](src/benchmark.py#L265-L271). |
| **R14** | `_QUALITY_METRICS_BY_GROUP` privatne mape | ✅ Inline-ovano u `ATTACK_GROUPS` |
| **R15** | `args_dict` u `main` nekoriscen | ✅ N/A — koristi se u `run_single_model` |
| **C2** | `Optional[float]` return hints | ✅ Fixed |
| **C3** | `pesq_wrapper(mode=...)` parametar nekoriscen | ✅ Fixed — sad ga `compute_metrics` prosljedjuje |
| **C6** | F-string preamble sa indentacijom | ⚠️ Partial — `latex_helpers.make_preamble` cist, ali `BenchmarkReportGenerator._preamble` u [report_generator.py:26-51](src/utils/report_generator.py#L26-L51) **nije migriran** i jos uvijek ima 8-prostorne f-string indente. |
| **C7** | `create_radar_chart` 95 linija | ✅ Fixed — razlozeno na `_draw_radar_axes` / `_draw_model_legend` / `_draw_attack_legend` |
| **C8** | `_metrics_table` `baseline_data` | ⚠️ Partial — i dalje opcioni parametar; nije razdvojen, ali docstring sad opisuje semantiku |
| **C9** | `Benchmark.run()` >200 linija | ❌ Open — funkcija je i dalje monolitna ([benchmark.py:84-285](src/benchmark.py#L84-L285)) |
| **C11** | `different_model_*` naming | ❌ Open — i dalje se koristi taj naziv u [benchmark.py:201-204](src/benchmark.py#L201-L204) |
| **K1 (new)** | PEP8: `if (attack_name =="CrossModelAttack"):` | ❌ Open — ([benchmark.py:199](src/benchmark.py#L199), [benchmark.py:243](src/benchmark.py#L243)) — viskak parenteza, fali razmak prije `==`, drugi blok je `elif` po smislu ali pisan kao novi `if` |

---

## 3. Otvoreni problemi (i dalje stoje)

### O1. SNR i dalje uopste nije izracunat (K3)

**Fajl:** `src/utils/metrics.py`, `src/benchmark.py`
**Tip:** Funkcionalna regresija
**Prioritet:** Visok

`compute_metrics` ne racuna SNR. Funkcija `snr` u [utils/utils.py:65](src/utils/utils.py#L65) postoji ali se ne uvozi u `metrics.py` niti u `benchmark.py`. Niti README, niti `--calculate_quality_metrics` help, niti `ALL_METRICS` ne pominju SNR.

**Posljedica:** prije ove grane, korisnici su imali SNR u izvjestaju. Sada ne. Ako je odluka **bila** da se ukloni, treba je eksplicitno zabiljeziti (CHANGELOG / README "Migration"). Ako nije bila — vratiti.

**Predlog:** dodati `"snr"` u `QUALITY_METRICS` (ili `INTELLIGIBILITY_METRICS`), uvesti `from utils.utils import snr` u `metrics.py`, dodati lambdu u `compute_metrics.computations`, dodati labelu u `METRIC_LABELS`, i navesti SNR u relevantnim grupama u `attack_groups.py`.

---

### O2. `to_json_safe` `None → "N/A"` (K4) — kompromisno rjesenje, jos krhko

**Fajl:** `src/run.py:21-38` i `src/run.py:41-52`
**Tip:** API/podaci
**Prioritet:** Visok

```python
def to_json_safe(obj):
    if obj is None:
        return "N/A"
    ...
```

Dodatak `from_json_safe` (linije 41-52) djeluje kao paliative ali ne rjesava sustinski problem:

1. **Spoljni potrosaci JSON-a** (pandas, jq, drugi Python skripti) ne znaju za `"N/A"` sentinel. `pd.read_json("benchmark_results.json")` ce dobiti `"N/A"` stringove tamo gdje treba `NaN`.
2. **Test `test_plain_python_types_unchanged`** ([test_run.py:65-68](tests/test_run.py#L65-L68)) **fiksira** ovo ponasanje kao zeljeno:
   ```python
   assert result == {"a": 1, "b": 2.0, "c": "hello", "d": True, "e": "N/A"}
   ```
   Test sad lockuje bug-as-feature.
3. **Workaround u downstream-u**: `aggregate_results` u [detailed_report_generator.py:217](src/utils/detailed_report_generator.py#L217) i [:220](src/utils/detailed_report_generator.py#L220) ima:
   ```python
   if wm_quality and wm_quality != "N/A":
       for m in all_metrics:
           val = wm_quality.get(m)
           if val is not None and val != "N/A":
               watermark_values[m].append(val)
   ```
   Pravilno bi bilo jednostavno `if val is not None` — duplo provjeravanje pokazuje da konvencija curi.

**Predlog:**
- Ukloniti `if obj is None: return "N/A"` iz `to_json_safe`. JSON `null` je vec idiomski.
- Ukloniti `from_json_safe` (postaje nepotreban).
- U `_format_val` (LaTeX displej) zadrzati `"N/A"` — to je samo formatiranje za prikaz, nije serijalizacija podataka.
- Azurirati `test_plain_python_types_unchanged` da ocekuje `None` umjesto `"N/A"`.
- Pojednostaviti `aggregate_results` workaround.

---

### O3. Cross-model accuracy dispatch je dupliran (R12)

**Fajl:** `src/benchmark.py:243-257` i `src/benchmark.py:265-271`
**Tip:** Duplikacija
**Prioritet:** Srednji

Iste tri grane (`is_zero_bit` / `returns_confidence` / regular) postoje za glavni model i za "different" model. Helper:

```python
def _accuracy_from_detection(self, detected, watermark, is_zero_bit, returns_confidence):
    if is_zero_bit:
        return detected.tolist() if isinstance(detected, np.ndarray) else detected
    if returns_confidence:
        detected, _ = detected
    return self.compare_watermarks(watermark, detected)
```

Onda:
```python
accuracy = self._accuracy_from_detection(
    detected_message, file_watermark, is_zero_bit, returns_confidence,
)
if attack_name == "CrossModelAttack":
    different_accuracy = self._accuracy_from_detection(
        different_detected_message, different_watermark,
        diff_is_zero_bit, diff_returns_confidence,
    )
```

---

### O4. PEP8 / stilske greske u CrossModel granama (K1 new)

**Fajl:** `src/benchmark.py:199-201, 211, 243`
**Tip:** Stil
**Prioritet:** Nizak

```python
if (attack_name =="CrossModelAttack"):       # nepotrebne zagrade, fali razmak prije ==
    
    different_model_name = kwargs.get("different_model_name")
    ...

#in case of the collusion mod attack
elif (attack_name == "ZeroBitCollusionAttack"):

else:
    ...

# pa NIZE, novi if:
if (attack_name =="CrossModelAttack"):
    different_detected_message = ...
```

Drugi `if` u liniji 243 bi trebao biti `if attack_name == "CrossModelAttack":` (bez zagrada, sa razmakom). Pored toga, dva `if` na isti uslov bi mogla biti spojena, ali to zahtijeva preuredivanje toka.

---

### O5. `Benchmark.run()` je preopterecen (C9)

**Fajl:** `src/benchmark.py:84-285`
**Tip:** Citljivost
**Prioritet:** Srednji

Funkcija ima >200 linija, 4 nivoa ugnjezdenja, mijesa: parsiranje argumenata, embedovanje, save audio, dispatch napada, racunanje accuracy-ja, racunanje kvaliteta, akumulaciju rezultata. Predlozena dekompozicija (`_run_file`, `_run_attack`) je iz prethodnog review-a — ostaje validno.

---

### O6. `BenchmarkReportGenerator._preamble` nije migriran (C6)

**Fajl:** `src/utils/report_generator.py:26-51`
**Tip:** Konzistentnost / duplikacija
**Prioritet:** Nizak

Detailed i comparative koriste `latex_helpers.make_preamble`. Basic generator zadrzava sopstveni `_preamble` sa f-stringovima koji ukljucuju 8 razmaka indentacije (preuzimaju se kao `        \\usepackage{...}` u .tex). Cisto kozmeticki, ali narusava jednoobraznost.

```python
# umjesto manuelne implementacije:
def _preamble(self, title, author):
    return make_preamble(title, author, self._has_deepmark_cls)
```

Tada uklniti i inline pdflatex blok ([report_generator.py:253-267](src/utils/report_generator.py#L253-L267)) zamjenom sa `compile_latex(self.report_dir, "benchmark_report")`.

---

### O7. Stari stil/citljivost: K5, K6, K7, K8

I dalje neresenо iz prethodnog review-a:

- **K5** — `(audio1, audio2)` vs `(reference, degraded)` u `metrics.py` (mix konvencija)
- **K6** — `sr` vs `fs` parametar (mix)
- **K7** — `'single'` vs `"double"` quotes u `run.py`
- **K8** — `Benchmark()` instanciran u svakoj test klasi preko autouse fixture-a (svaki put PluginManager skenira diskove). Session-scoped fixture u `conftest.py` ubrzao bi suite (74 testa).

---

### O8. `requirements.txt` bez trailing newline (M11)

Zadnja linija `audiocomplib==0.2.0` bez `\n`. Cisto kozmeticki.

---

## 4. Novi nalazi

### N1. Sirok `except Exception` u `run.py` na 4 mjesta

**Fajl:** `src/run.py:173, 253-254, 264-265, 337-338`
**Tip:** Anti-pattern (slicno V6 ali za `run.py`)
**Prioritet:** Srednji

```python
# linija 173:
except Exception as e:
    logger.error(f"Error accessing audio directory {args.wav_files_dir}: {e}")
    return

# linija 253:
except Exception as e:
    logger.error(f"Failed to generate benchmark report: {e}")

# linija 264:
except Exception as e:
    logger.error(f"Failed to generate detailed report: {e}")

# linija 337:
except Exception as e:
    logger.error(f"Failed to generate comparative report: {e}")
```

`metrics.py` je rijesio ovo (sirenje na konkretne tipove), `run.py` nije. Posebno u (253/264/337) — guta i `KeyError`, `AttributeError`, `TypeError` koji su znakovi bug-ova, ne valid runtime fail-ova.

**Predlog:** suziti na `(OSError, IOError, ValueError, RuntimeError, subprocess.SubprocessError)` zavisno od kontextsa, ili koristiti `logger.exception(...)` koji loguje stack trace pa ne progutava informaciju o bug-u.

---

### N2. `run_multiple_models` exception filter ne hvata uobicajene model-fail tipove

**Fajl:** `src/run.py:291`
**Tip:** Robusnost
**Prioritet:** Visok

```python
except (MemoryError, ConnectionError, OSError) as e:
```

Komentar iznad kaze da treba propustiti samo "infrastrukturne" greske, ali u praksi modeli najcesce padaju sa:

- **`RuntimeError`** — PyTorch CUDA OOM (`torch.cuda.OutOfMemoryError` izvedeno iz `RuntimeError`), torch dispatcher greske
- **`subprocess.CalledProcessError`** — Docker `up -d` failures
- **`requests.RequestException`** / **`httpx.HTTPError`** — HTTP timeouts, 5xx, connection refused (NIJE `ConnectionError`)
- **`json.JSONDecodeError`** — pogresan response od FastAPI
- **`KeyError`** — model konfigracija nedostaje neki kljuc u JSON response-u

Ako bilo koji od ovih izleti, cijeli multi-model run pada. To je tacno onaj scenario koji je S7 trebao da rijesi.

**Predlog:**
```python
except Exception as e:
    logger.exception(f"Model {model_name} failed; skipping. Error:")
    failed_models.append(model_name)
    continue
```
`logger.exception` loguje pun stack trace, tako da se code-bugovi i dalje vide u log-u (a ne tihog skipping-a kao u prethodnoj verziji). Pratece: dodati `--strict` flag koji propagira sve da se mogu eksplicitno hvatati u testovima.

---

### N3. `test_plain_python_types_unchanged` lockuje "N/A" kao zeljeno ponasanje

**Fajl:** `tests/test_run.py:65-68`
**Tip:** Test-as-spec
**Prioritet:** Srednji (zavisno od resenja O2)

```python
def test_plain_python_types_unchanged(self):
    data = {"a": 1, "b": 2.0, "c": "hello", "d": True, "e": None}
    result = to_json_safe(data)
    assert result == {"a": 1, "b": 2.0, "c": "hello", "d": True, "e": "N/A"}
```

Ime testa kaze "plain types unchanged" — ali test eksplicitno tvrdi da se `None` MIJENJA u `"N/A"`. Ovo je internal-konvencija pretvorena u tvrdo pravilo testovima. Ako se O2 prihvati, ovaj test se mora azurirati (ocekivano `"e": None`).

---

### N4. Lazy import `from utils.metrics import ALL_METRICS` u `attack_groups.py`

**Fajl:** `src/utils/attack_groups.py:111`
**Tip:** Stil / mrtav kod kompleksnosti
**Prioritet:** Nizak

```python
def get_metrics_for_attack(attack_name):
    ...
    if group_key is None:
        from utils.metrics import ALL_METRICS
        return list(ALL_METRICS)
```

Lazy import obicno sluzi za izbjegavanje cirkularnih importa. U ovom slucaju **nema cirkularnog importa**: `metrics.py` ne uvozi nista iz `attack_groups`. Lazy import je nepotrebna kompleksnost.

**Predlog:** prebaciti import na vrh fajla.

---

### N5. Klasni atribut shadow: `INTELLIGIBILITY_METRICS = INTELLIGIBILITY_METRICS`

**Fajl:** `src/utils/detailed_report_generator.py:153-155`
**Tip:** Citljivost / fragility
**Prioritet:** Nizak

```python
class DetailedReportGenerator:
    BASE_QUALITY_METRICS = QUALITY_METRICS
    INTELLIGIBILITY_METRICS = INTELLIGIBILITY_METRICS   # shadow
    ALL_METRIC_LABELS = METRIC_LABELS
```

Drugi red doslovno pise `X = X` na klasnom nivou — to radi jer Python razrjesava desnu stranu u modulskom scope-u prije pravljenja klasnog atributa, ali je krhko: ako se ikada zapise `from utils.metrics import INTELLIGIBILITY_METRICS as IM`, klasna definicija puca.

**Predlog:** preimenovati klasne atribute (`BASE_INTELLIGIBILITY_METRICS`) ili sasvim ukinuti — koristiti modulske konstante direktno: `aggregate_results` i ostali metodi mogu citati `metrics.QUALITY_METRICS` direktno.

---

### N6. Dvostruki `trim_audio_to_match` u toku `compute_metrics → wrapper`

**Fajl:** `src/utils/metrics.py`
**Tip:** Performansa (mala)
**Prioritet:** Nizak

`compute_metrics` na liniji 489 radi `ref_trimmed, deg_trimmed = trim_audio_to_match(...)` i prosljedjuje rezultate. Svaki dekorisani wrapper (`@_safe_metric`) **ponovo** poziva `trim_audio_to_match` u dekoratoru ([metrics.py:24](src/utils/metrics.py#L24)).

Drugi trim je no-op (signali su vec iste duzine), ali svaki put kreira tuple. Beznacajno za male signale, mjerljivo za 4000 poziva × 8 metrika.

**Predlog:** dodati flag `_pre_trimmed` u dekorator ili izbaciti trim iz wrapper-a kad zna se da ga je `compute_metrics` vec uradio. Alternativno: vjerovati da je idempotentno i ostaviti.

---

### N7. `_safe_metric` ne hvata Tensorflow/PyTorch greske (ViSQOL)

**Fajl:** `src/utils/metrics.py:27, 273-291`
**Tip:** Robusnost
**Prioritet:** Srednji

`@_safe_metric("ViSQOL")` lovi `(RuntimeError, ValueError, IndexError)`. ViSQOL interno koristi TensorFlow, koji moze podici:

- `tensorflow.errors.InvalidArgumentError` (izvedeno iz `Exception` direktno, NE iz `RuntimeError`)
- `tensorflow.errors.OutOfRangeError`
- Drugi C-extension errori

Ako se podigne neki od tih, dekorator ne hvata, izuzetak izleti i obara kompletan `compute_metrics` poziv — pa onda i cijeli benchmark fajl jer ne postoji vise fail-tolerant nivo.

**Predlog:** za ViSQOL specificno, prosiriti listu na `(RuntimeError, ValueError, IndexError, Exception)` ili (eksplicitnije) na `(RuntimeError, ValueError, IndexError, TypeError)` + `try/except Exception` samo unutar `visqol_wrapper`.

---

### N8. `Benchmark.run()` `attack_kwargs` polujeva CLI smece u napade

**Fajl:** `src/benchmark.py:141-147`
**Tip:** Slabo enkapsulisanje
**Prioritet:** Srednji

```python
attack_kwargs = {
    **kwargs,                    # <- sadrzi wav_files_dir, wm_models, attack_group, verbose, calculate_quality_metrics, ...
    "model": model_instance,
    "watermark_data": watermark_data,
    "sampling_rate": sampling_rate,
    "models": self.models,
}
```

Pozivac (`run_single_model`) prosljeduje `**args_dict` u `benchmark.run`, koji ga zatim prosljeduje u svaki `attack_instance.apply(...)`. Svaki napad dobija `wav_files_dir`, `wm_model`, `wm_models`, `attack_group`, `verbose`, `calculate_quality_metrics` kao kwargs.

Vecina napada koristi `**kwargs` pa ovo trenutno ne pada — ali:
1. Ako se neki napad refaktorise da eksplicitno lista svoje argumente i ne prima `**kwargs`, padace.
2. Iz koda nije jasno koji argumenti su namijenjeni napadu, a koji su CLI smece.

**Predlog:** u `run_single_model` napraviti whitelistu attack_kwargs (filter `args_dict` na samo kljuceve koji su iz `valid_args`). Ili `Benchmark.run` da prima eksplicitno `attack_params=dict(...)` i ne propusta `**kwargs`.

---

### N9. `comparative_report_generator.generate_full_report` ima dead arguments

**Fajl:** `src/utils/comparative_report_generator.py:320-333`
**Tip:** API / dokumentacija
**Prioritet:** Nizak

```python
def generate_full_report(self, all_results, all_stats,
                          calculate_quality_metrics=False):
    """..."""
    del all_results, calculate_quality_metrics  # unused, see docstring
```

Argumenti su deklarisani, prima ih `run.py`, pa se odmah `del`-uju. Docstring kaze "kept for API compatibility", ali ovo nije public API — samo `run.py` poziva. Predlog: ukloniti argumente iz signature i poziva u `run.py:332-335`.

---

### N10. `compute_metrics` koristi lambde za sve, kreira ih i kad se ne koriste

**Fajl:** `src/utils/metrics.py:508-519`
**Tip:** Mikro-optimizacija
**Prioritet:** Vrlo nizak

```python
computations = {
    "pesq": lambda: pesq_wrapper(ref_nb, deg_nb, nb_sr, pesq_mode),
    ...
    "ncm": lambda: ncm(ref_trimmed, deg_trimmed, sr),
}
return {name: computations[name]() for name in ALL_METRICS if name in requested}
```

Sve lambde se prave i kad nisu zatrazene. Brzo na njihovo kreiranje, ali capture varijabli (`ref_nb`, `deg_nb`, `nb_sr`, `pesq_mode`) u closure-u za PESQ je nepotrebno ako se PESQ ne racuna.

**Predlog:** `if/elif` ili dispatch dict van funkcije sa direktnim pozivom. Ne mora ako benchmark dominira drugim cost-ovima.

---

### N11. `sr_scalar` defensive cast je nejasan

**Fajl:** `src/benchmark.py:179`
**Tip:** Citljivost
**Prioritet:** Nizak

```python
sr_scalar = int(sampling_rate) if isinstance(sampling_rate, (np.ndarray, list)) else sampling_rate
```

`sampling_rate` u ovom toku dolazi iz `load_audio` (linija 164) koji vraca `int` ili float skalar (`librosa.load`). Lista/array je veoma necesto. Ako je `np.ndarray` sa vise elemenata, `int(...)` baca `TypeError`. Ako je sa jednim, prolazi.

Ako se branilo od edge case-a koji se desavao, dodati komentar (i test). Ako se "samo za svaki slucaj" — obrisati.

---

### N12. `display_attack_name` ne vodi racuna o pravoj granici rijeci

**Fajl:** `src/utils/latex_helpers.py:84`
**Tip:** Edge case
**Prioritet:** Vrlo nizak

```python
stripped = attack_name.replace("Attack", "").strip()
```

`.replace("Attack", "")` zamjenjuje SVE pojave. Ako bi neko nazvao klasu `AttackOnSomethingAttack`, dobio bi `OnSomething`. Trenutno nista u `attack_groups.py` ne pati od ovog, ali je krhko ako neko imenuje `ReplayAttackV2`.

**Predlog:** `re.sub(r'Attack$', '', attack_name).strip()` — strip samo trailing `Attack`.

---

### N13. `RANDOM_GUESS_ACCURACY` definisan u sredini klase

**Fajl:** `src/benchmark.py:355-358`
**Tip:** Stil
**Prioritet:** Vrlo nizak

```python
class Benchmark:
    def __init__(self): ...
    ...
    def compute_mean_accuracy(self, results): ...

    # Accuracy returned when detection produces no usable watermark.
    RANDOM_GUESS_ACCURACY = 50.00

    @staticmethod
    def _is_invalid_detection(...): ...
```

Klasna konstanta je deklarisana **izmedu** dva metoda. Idiomski bi bila na vrhu klase (odmah ispod docstring-a `class Benchmark:`), uz ostale konstante.

---

### N14. `ALL_METRICS` ne ukljucuje SNR pa `get_metrics_for_attack("FakeAttack")` ga necese vratiti

**Fajl:** `src/utils/attack_groups.py:111-112` + `metrics.py:439`
**Tip:** Posljedica O1
**Prioritet:** Visok (vezano za O1)

Ako se O1 (vratiti SNR) usvoji, ne smije se zaboraviti azurirati `ALL_METRICS = QUALITY_METRICS + INTELLIGIBILITY_METRICS` da SNR udje u skup, plus relevantne grupe u `attack_groups.py` (`audio_distortion` posebno).

---

### N15. `_clean_report_dir` ne radi nista ako `report_dir` ne postoji

**Fajl:** `src/run.py:190-201`
**Tip:** Logicka rupa
**Prioritet:** Vrlo nizak

```python
def _clean_report_dir(report_dir):
    if not os.path.exists(report_dir):
        return
    for item in os.listdir(report_dir):
        ...
```

Ako `report_dir` ne postoji, funkcija izadje. Ali pozivaci ([run.py:183](src/run.py#L183), [run.py:273](src/run.py#L273)) zatim ocekuju da fajlovi mogu biti pisani u njega — a zapravo `os.makedirs(report_dir, exist_ok=True)` se desi tek kasnije u `run_single_model`. Trenutno radi jer se `os.makedirs` poziva poslije. Stavlja se u kategoriju "ne pada slucajno", ali bi cisto bilo `os.makedirs(report_dir, exist_ok=True)` na samom vrhu `_clean_report_dir`.

---

### N16. `_DEEPMARK_ASSETS` se ne sinhronizuje sa stvarnim assetima

**Fajl:** `src/run.py:187`
**Tip:** Hardkodirana lista
**Prioritet:** Vrlo nizak

```python
_DEEPMARK_ASSETS = {"deepmark.cls", "deepmark-logo.png", "deepmark-logo.pdf", "deepmark-logo.jpg"}
```

Lista je hardkodirana. Ako se doda novi asset (npr. `deepmark-banner.svg`), `_clean_report_dir` ce ga obrisati. Trenutno radi. Predlog: `os.listdir` i filtriranje po prefiksu `deepmark*`, ili regex. Nije hitno.

---

### N17. `psnr` vraca `float('inf')` umjesto None za savrseno poklapanje

**Fajl:** `src/utils/metrics.py:144-147`
**Tip:** API ne-uniformnost
**Prioritet:** Nizak

```python
if mse == 0:
    return float('inf')
```

Ostale metrike na "ne moze da racuna" vracaju `None`. PSNR vraca `inf`, koji je validan u float matematici, ali u JSON-u nije validan (`json.dump(float('inf'))` baca `ValueError: Out of range float values are not JSON compliant`).

**Provjera:** `to_json_safe` ne handluje `inf`. Ako `psnr` vrati `inf`, `json.dump(...)` ce baciti gresku.

**Predlog:** vratiti veliku konstantu (`100.0` dB) ili `None` umjesto `inf`. Aktuelan predlog: `return None` i pustiti drugu logiku da to interpretira.

---

### N18. `compare_watermarks` ne handluje `np.array(None) == ...` numpy upozorenja

**Fajl:** `src/benchmark.py:369`
**Tip:** numpy DeprecationWarning
**Prioritet:** Nizak

```python
if np.any(detected == np.array(None)):
    return True
```

`np.array(None)` pravi 0-d object array. `detected == np.array(None)` radi element-wise poredjenje, sto u novijim numpy verzijama (>=1.25) baca `DeprecationWarning` ako su tipovi nekompatibilni. U `numpy 2.x` (koji se koristi: `numpy==2.2.6` u requirements.txt), ova putanja je rizik.

**Predlog:** umjesto toga koristiti `pandas.isna`-style provjeru ili iterirati: `if isinstance(detected, np.ndarray) and detected.dtype == object and any(x is None for x in detected.flatten())`.

---

### N19. `metrics.py:209` — visak praznog reda izmedu `stoi_wrapper` i `pesq_wrapper`

**Fajl:** `src/utils/metrics.py:208-209`
**Tip:** Stil
**Prioritet:** Trivijalno

Dvostruki blank prije `pesq_wrapper`. PEP8 dozvoljava 2 prazna reda izmedu top-level definicija — OK. Ali `_visqol_cache = {}` na liniji 238 ima samo jedan blank prije sebe, a poslije module-konstanti (`_MIN_METRIC_DURATION_S`) ima dva. Nekonzistentno.

---

### N20. Test fajlovi nemaju test za `compute_metrics`

**Fajl:** `tests/`
**Tip:** Pokrivenost
**Prioritet:** Srednji

Postoji `test_attack_groups.py`, `test_benchmark.py`, `test_report_generator.py`, `test_run.py`, ali nema `test_metrics.py`. Sve nove metrike (`sii`, `mcd`, `ncm`, `visqol_wrapper`, `compute_metrics`) su uvedene u ovoj grani i nisu pokrivene unit testovima. Konkretno bi dobro doslo:

- `compute_metrics` sa `metrics=["pesq"]` vraca samo PESQ
- `compute_metrics` sa nepostojecom metrikom (graceful degradation)
- `mcd(ref, ref) ≈ 0`
- `sii(ref, ref) ≈ 1.0`
- `ncm(ref, ref) ≈ 1.0`
- `visqol_wrapper` vraca None kad ViSQOL nije instaliran (mock-ovati `_get_visqol_api`)
- `_ansi_bands(fs=4000)` graceful pad (ispod nyquist-a sve)

---

### N21. `__pycache__` ili stub `__init__.py` za `tests/` paket

**Fajl:** `tests/conftest.py:10`
**Tip:** Robusnost / pytest discovery
**Prioritet:** Nizak

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
```

Ovaj hack u `conftest.py` radi, ali se nesinhronizuje sa CI-jevim `pytest`. Cisce: `tests/__init__.py` (nema), ili `pytest.ini` / `pyproject.toml` sa `[tool.pytest.ini_options] pythonpath = ["src"]`.

---

### N22. `comparative_report_generator` izmenjivost je krhka

**Fajl:** `src/utils/comparative_report_generator.py:148-151`
**Tip:** Magic numbers / nedoslednost sa basic generator-om
**Prioritet:** Nizak

```python
_MODEL_COLORS = [
    "#039FAC", "#E74C3C", "#2ECC71", "#F39C12",
    ...
]
```

Basic generator (`report_generator.py:69`) koristi `bar_color = '#469CA9'` — slicna paleta ali nedosledno (drugacija nijansa). Bilo bi cisto deliti paletu izmedu generatora ili je centralizovati u `latex_helpers.py` / `theme.py`.

---

### N23. Docker requirements bez pinova → CUDA dependency drift

**Fajlovi:**
- `Dockerfile.base:11-21`
- `src/plugins/attacks/speech_enhancement_1/requirements.txt`
- `src/plugins/attacks/diffusion/requirements.txt`
- `src/plugins/attacks/vae/requirements.txt`

**Tip:** Build determinizam / dependency drift
**Prioritet:** **Visok** (blokira reproducibilan build)

**Simptom otkriven 2026-05-19:** Nakon reinstalacije laptopa i fresh `docker compose build`-a, kontejner `speech_enhancement1` puca pri startu sa:
```
OSError: libcudart.so.13: cannot open shared object file: No such file or directory
```
Stari build je radio iz cache-a — novi build nije. Servis nikada ni ne podigne FastAPI, port 10005 je `Connection refused`.

**Uzrok:** `Dockerfile.base` instalira `torch==2.7.1` ali `torchaudio` **bez pina**, i bez `--extra-index-url https://download.pytorch.org/whl/cpu`. Default PyTorch wheel se u međuvremenu prešalo sa CUDA 12 na CUDA 13. Onda u inner Dockerfile-u `pip install speechbrain` (koji nije pinovan) povlači transitive dependency koja upgrade-uje `torchaudio` na verziju linkovanu sa CUDA 13, koja na CPU-only kontejneru ne uspijeva da učita `libcudart.so.13`.

```dockerfile
# Dockerfile.base — TRENUTNO
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchaudio \             # ← nije pinovano
    numpy \
    ...
```

```
# speech_enhancement_1/requirements.txt — TRENUTNO
pydantic
fastapi
uvicorn
librosa
speechbrain                  # ← nije pinovano, povlači novije torch/torchaudio
```

**Fix (Dockerfile.base):**
```dockerfile
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1 \
    torchaudio==2.7.1 \
    ...
```

**Fix (svaki inner requirements.txt koji instalira speechbrain/diffusers/transformers):**
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.7.1
torchaudio==2.7.1
speechbrain==<konkretna_verzija>
...
```

**Slican rizik (potvrđen pregledom):**
- `diffusion/requirements.txt` — `torch` bez verzije, `diffusers`/`accelerate` bez verzije
- `vae/requirements.txt` — `torch` bez verzije

Ti servisi trenutno rade samo zato što PyPI još uvijek serve-uje kompatibilne kombinacije, ali su ranjivi na isti drift.

**Why it matters:** Build prestaje biti reproducibilan. Build koji radi danas može pasti za nekoliko nedjelja kada upstream package release neki novi major. Ovo je classic Python/Docker antipattern za ML servise.

---

## 5. Sumarna tabela svih otvorenih nalaza

| ID | Fajl | Tip | Prioritet | Kratak opis |
|---|---|---|---|---|
| **O1** | `metrics.py`, `benchmark.py` | Funkcionalnost | **Visok** | SNR i dalje nije izracunat |
| **O2** | `run.py` | Podaci | **Visok** | `to_json_safe None → "N/A"` razbija JSON kompatibilnost |
| **N2** | `run.py:291` | Robusnost | **Visok** | Multi-model `except` ne hvata `RuntimeError`/HTTP/subprocess |
| **N14** | `metrics.py`, `attack_groups.py` | Posljedica O1 | **Visok** | Kad se SNR vrati, treba ga ukljuciti u `ALL_METRICS` i grupe |
| **O3** | `benchmark.py:243-271` | Duplikacija | Srednji | Cross-model accuracy dispatch dupliran |
| **O5** | `benchmark.py:84-285` | Citljivost | Srednji | `Benchmark.run()` 200+ linija, monolitan |
| **N1** | `run.py:173,253,264,337` | Anti-pattern | Srednji | `except Exception` siroko |
| **N3** | `tests/test_run.py:65-68` | Test-as-spec | Srednji | Test lockuje N/A bug |
| **N5** | `detailed_report_generator.py:153-155` | Citljivost | Nizak | Klasni atribut shadowuje import |
| **N7** | `metrics.py:_safe_metric` | Robusnost | Srednji | Ne hvata TensorFlow greske u ViSQOL |
| **N8** | `benchmark.py:141-147` | Enkapsulacija | Srednji | `attack_kwargs` propagira CLI smece |
| **N17** | `metrics.py:144-147` | API | Srednji | `psnr` vraca `inf` koji nije JSON-validno |
| **N20** | `tests/` | Test pokrivenost | Srednji | Nema `test_metrics.py` za nove metrike |
| **O4** | `benchmark.py:199,243` | PEP8 | Nizak | Visak zagrada, fali razmak `==` |
| **O6** | `report_generator.py:26-51` | Konzistentnost | Nizak | `_preamble` nije migriran na `latex_helpers` |
| **O7** | `metrics.py`, testovi | Stil | Nizak | K5/K6/K7/K8 (naming, quotes, fixture) |
| **O8** | `requirements.txt` | Stil | Nizak | Nedostaje trailing newline |
| **N4** | `attack_groups.py:111` | Stil | Nizak | Lazy import `ALL_METRICS` bez razloga |
| **N6** | `metrics.py` | Performansa | Nizak | Dvostruki `trim_audio_to_match` |
| **N9** | `comparative_report_generator.py:320` | API | Nizak | Dead arguments u `generate_full_report` |
| **N10** | `metrics.py:508-519` | Mikro-opt | Vrlo nizak | Lambde se prave i kad nisu trazene |
| **N11** | `benchmark.py:179` | Citljivost | Nizak | `sr_scalar` defensive cast bez objasnjenja |
| **N12** | `latex_helpers.py:84` | Edge case | Vrlo nizak | `display_attack_name` zamjenjuje SVE "Attack" |
| **N13** | `benchmark.py:355` | Stil | Vrlo nizak | Klasna konstanta usred klase |
| **N15** | `run.py:190-201` | Logicka rupa | Vrlo nizak | `_clean_report_dir` ne pravi dir |
| **N16** | `run.py:187` | Hardkod | Vrlo nizak | `_DEEPMARK_ASSETS` lista |
| **N18** | `benchmark.py:369` | numpy DeprecationWarning | Nizak | `np.array(None)` poredjenje |
| **N19** | `metrics.py:208-238` | Stil | Trivijalno | Nekonzistentni blank lines |
| **N21** | `tests/conftest.py:10` | CI robustnost | Nizak | `sys.path.insert` hack umjesto `pyproject.toml` |
| **N22** | `comparative_report_generator.py:148` | Konzistentnost | Nizak | Boja paleta nedoslednog stila |

---

## 6. Preporuceni redoslijed rjesavanja

### Prije merge-a u `dev` (high priority, blocking)

1. **O1 + N14** — vratiti SNR (ili eksplicitno dokumentovati da je trajno uklonjen, npr. u CHANGELOG-u i README "Migration" sekciji)
2. **O2 + N3** — odluciti da li `to_json_safe(None)` treba da vraca `null` ili `"N/A"`. Preporuka: vratiti na `null`, ukloniti `from_json_safe`, azurirati test, pojednostaviti workaround u `aggregate_results`.
3. **N2** — prosiriti exception filter u `run_multiple_models` na `Exception` sa `logger.exception` (ili eksplicitno pobrojati `RuntimeError`, `subprocess.CalledProcessError`, `requests.RequestException`, `httpx.HTTPError`)
4. **N17** — `psnr` da vraca `None` (ili konacnu vrijednost) umjesto `float('inf')` da JSON ne pada

### U narednom sprintu

5. **O3** — helper `_accuracy_from_detection` (skida ~30 linija duplikacije)
6. **N7** — prosiriti `_safe_metric` ili wraps-ovati ViSQOL specijalno
7. **N8** — whitelist `attack_kwargs` u `run_single_model`
8. **N1** — suziti `except Exception` u `run.py` ili koristiti `logger.exception`
9. **O5** — dekomponovati `Benchmark.run()` na `_run_file` / `_run_attack`
10. **N20** — dodati `tests/test_metrics.py` (mcd/sii/ncm/compute_metrics)

### Tehnicki dug

11. **O4** — PEP8 cleanup `(attack_name =="...")` u `benchmark.py`
12. **O6** — migrirati `BenchmarkReportGenerator._preamble` na `make_preamble`, `compile_latex`
13. **N5** — preimenovati klasne atribute u `DetailedReportGenerator`
14. **N9** — ukloniti dead args iz `ComparativeReportGenerator.generate_full_report`
15. **N6, N10, N11, N12, N13** — sitne ciscenja
16. **O7 (K5/K6/K7/K8)** — naming harmonizacija + session-scoped Benchmark fixture
17. **O8, N16, N19, N21, N22** — kozmetika

---

## Konacna ocjena

**Velika napreduje od prvog review-a.** Skoro svi kriticni i visoko-prioritetni problemi iz prvog round-a su rijeseni:

- ✅ SII/NCM matematicki ispravljeni i koriste ANSI standard
- ✅ MCD scaling, ViSQOL caching, narrowband resampling, per-grupa metric selection — sve adresovano
- ✅ Duplikacije izvuene u `latex_helpers.py`, `_safe_metric`, `_check_min_length`, `_log_plugin_entry`, `_is_invalid_detection`
- ✅ Single source of truth za relevantne metrike po grupi (`attack_groups.py`)

**Ostala 4 visok-prioritetna gleam-a** (SNR regresija, JSON `None` semantika, multi-model exception filter, PSNR `inf`) su ono sto sprjecava ciste merge-a u `dev`.

**Citljivost i pokrivenost testovima** su sledeci najveci dug. `Benchmark.run()` je i dalje monolitno, a nove metrike (~50% novog koda) su bez unit testova.

---

*Generisano: 2026-05-19, na osnovu HEAD `c68498c` (Resolve bugs).*
