# Code Review: DPM-1576-add-metrics-to-benchmark

**Grana:** `DPM-1576-add-metrics-to-benchmark` vs `origin/dev`
**Commit:** `17d491d DPM-1576-add-metrics-and-reports-to-benchmark`
**Obim:** 12 fajlova, +2049 / −150 linija, 2 nova fajla (attack_groups.py, detailed_report_generator.py, comparative_report_generator.py)

Ovaj dokument sadrzi sve identifikovane probleme u promjenama, grupisane po prioritetu i fajlu, sa predlozima resenja.

---

## Sadrzaj

1. [Kriticni problemi](#kriticni-problemi)
2. [Visok prioritet](#visok-prioritet)
3. [Srednji prioritet](#srednji-prioritet)
4. [Nizak prioritet (minor)](#nizak-prioritet-minor)
5. [Semanticka pitanja (potrebna diskusija)](#semanticka-pitanja-potrebna-diskusija)
6. [Sumarna tabela](#sumarna-tabela)

---

## Kriticni problemi

Problemi koji proizvode **pogresne rezultate** ili dovode do **gubitka funkcionalnosti**.

### K1. `sii` racuna FFT samo prvih 2048 uzoraka (ignorise ostatak signala)

**Fajl:** `src/utils/metrics.py:197-199`
**Tip:** Bug

```python
ref_spec = np.abs(np.fft.rfft(reference, n=n_fft)) ** 2
deg_spec = np.abs(np.fft.rfft(degraded, n=n_fft)) ** 2
```

**Problem:**
`np.fft.rfft(signal, n=2048)` uzima samo prvih 2048 uzoraka signala. Za signal duzine 5 s na 16 kHz (80000 uzoraka), ignorise se 77952 uzoraka (97.4% signala). Rezultat nije reprezentativan.

**Rjesenje:**
Koristiti STFT framing kao u `ncm`:
```python
ref_stft = librosa.stft(reference, n_fft=n_fft, hop_length=n_fft//2)
ref_spec = np.mean(np.abs(ref_stft) ** 2, axis=1)
```

---

### K2. `sii` pogresna definicija suma

**Fajl:** `src/utils/metrics.py:220-222`
**Tip:** Matematicka greska

```python
signal_power = np.mean(ref_spec[mask])
noise = deg_spec[mask] - ref_spec[mask]
noise_power = np.mean(np.maximum(noise, 0) + 1e-12)
```

**Problem:**
Razlika **spektralnih snaga** dva signala nije "sum" u signalnom smislu. Pravi sum je magnitudni spektar razlike signala: `noise_signal = degraded - reference`, pa tek onda spektar.

**Rjesenje:**
```python
noise_signal = degraded - reference
noise_stft = librosa.stft(noise_signal, n_fft=n_fft, hop_length=n_fft//2)
noise_power_per_band = np.mean(np.abs(noise_stft) ** 2, axis=1)
```

---

### K3. SNR metrika je uklonjena bez zamjene

**Fajl:** `src/benchmark.py` (uklonjeno iz `dev`)
**Tip:** Regresija funkcionalnosti

Prije:
```python
snr_val = snr(audio, attacked_audio)
results[filepath][attack_name] = {"accuracy": ..., "stoi": ..., "pesq": ..., "snr": snr_val}
```

Sada: SNR se **uopste ne racuna** — niti u `compute_metrics`, niti u `benchmark.py`.

**Rjesenje:**
Dodati SNR u `compute_metrics`:
```python
from utils.utils import snr as snr_func

def compute_metrics(reference, degraded, sr):
    ...
    result = {
        "snr": snr_func(ref_trimmed, deg_trimmed),
        "pesq": ...,
        ...
    }
```

---

### K4. `to_json_safe` mijenja `None` u `"N/A"` — razbija JSON round-trip

**Fajl:** `src/run.py:26-27`
**Tip:** Bug sa posljedicama

```python
def to_json_safe(obj):
    if obj is None:
        return "N/A"
    ...
```

**Problem:**
Kada metrika nije izracunata (neuspjeh wrappera), interno se cuva `None`. Nakon serializacije, `None` postaje string `"N/A"`. Ako neko kasnije ucita taj JSON nazad u Python (npr. za agregaciju), dobija **string umjesto None-a**. Kod koji radi `if value is None` prestaje da radi; kod koji racuna statistike puca jer `"N/A"` nije broj.

JSON standard vec ima `null` za None; ne treba ga mijenjati.

**Rjesenje:**
Ukloniti `if obj is None: return "N/A"` iz `to_json_safe`. Pustiti Python `None` da se serijalizuje u JSON `null`. Formatiranje u `"N/A"` raditi **samo u LaTeX generatoru** na mjestu prikazivanja:

```python
# u _format_val metodu:
def _format_val(self, stats):
    if stats is None or stats.get("mean") is None:
        return "N/A"   # OK — samo za displej
    return f"{stats['mean']:.2f}"
```

---

## Visok prioritet

### V1. `VisqolApi()` se instancira pri svakom pozivu — znacajno usporava benchmark

**Fajl:** `src/utils/metrics.py:161-163`
**Tip:** Performansni problem

```python
def visqol_wrapper(reference, degraded, fs=16000):
    ...
    try:
        api = VisqolApi()
        if fs >= 48000:
            api.create(mode="audio")
        else:
            api.create(mode="speech")
        result = api.measure_from_arrays(reference, degraded, fs)
```

**Problem:**
`VisqolApi.create()` ucitava TensorFlow model. Za benchmark sa 100 fajlova × 40 napada = 4000 poziva, ucitavanje se ponavlja svaki put.

**Rjesenje:**
Cache-ovati API instancu po modu:
```python
_visqol_cache = {}

def _get_visqol(mode):
    if mode not in _visqol_cache:
        api = VisqolApi()
        api.create(mode=mode)
        _visqol_cache[mode] = api
    return _visqol_cache[mode]

def visqol_wrapper(reference, degraded, fs=16000):
    reference, degraded = trim_audio_to_match(reference, degraded)
    try:
        mode = "audio" if fs >= 48000 else "speech"
        api = _get_visqol(mode)
        result = api.measure_from_arrays(reference, degraded, fs)
        return result.moslqo
    except (RuntimeError, ValueError) as e:
        logger.warning(f"ViSQOL calculation failed: {e}")
        return None
```

---

### V2. PESQ se prikazuje za `process_disruption` grupu — stvara laznu sigurnost

**Fajl:** `src/utils/detailed_report_generator.py:21`
**Tip:** Metodoloska greska (vec diskutovana sa korisnikom)

```python
GROUP_QUALITY_METRICS = {
    "process_disruption": ["pesq", "psnr", "si_sdr", "mcd", "visqol"],
    ...
}
```

**Problem:**
Grupa `process_disruption` obuhvata `CrossModelAttack`, `CollusionAttack`, `ZeroBitCollusionAttack`, `SameModelAttack`. PESQ se **moze** izracunati jer je duzina i poravnanje signala ocuvano, ali **ne mjeri ono sto je sustina napada** (preziveci watermark vs. kvalitet audia). Visok PESQ tu moze znaciti da watermark vise ne postoji — zavaravajuci je.

**Rjesenje:**
Isprazniti PESQ (i druge quality metrike) za ovu grupu:
```python
"process_disruption": [],   # PESQ/PSNR/SI-SDR ne mjere sustinu ovih napada
```
ili zamijeniti komentarom u izvjestaju koji eksplicitno kaze da quality metrike nisu primjenjive.

---

### V4. `mcd` nema standardni scaling faktor

**Fajl:** `src/utils/metrics.py:262`
**Tip:** Nekompatibilnost sa literaturom

```python
return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))
```

**Problem:**
Standardna MCD formula je:
```
MCD = (10 / ln(10)) × sqrt(2 × Σ(c_ref - c_deg)²)
```
Nedostaje faktor `10/ln(10) × sqrt(2) ≈ 6.14`. Rezultati su priblizno 6.14× manji nego objavljeni MCD u literaturi — ne mogu se direktno uporedjivati sa radovima.

**Rjesenje:**
Ili dodati standardni faktor:
```python
MCD_FACTOR = 10.0 / np.log(10.0) * np.sqrt(2.0)  # ≈ 6.1413
...
return float(MCD_FACTOR * np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))
```
Ili eksplicitno u docstring-u navesti: *"Ova implementacija ne koristi standardni MCD scaling; vrijednosti su ~6× nize od literaturnih."*

Takodje razmotriti iskljucivanje MFCC[0] (DC/energy komponenta) koja se cesto preskace u MCD:
```python
mfcc_ref = librosa.feature.mfcc(y=reference, sr=sr, n_mfcc=n_mfcc)[1:, :]
```

---

### V5. Help tekst za `--calculate_quality_metrics` je zastareo

**Fajl:** `src/run.py:100`
**Tip:** Dokumentacija

```python
parser.add_argument(
    "--calculate_quality_metrics",
    ...
    help="Calculate audio quality metrics (PESQ, STOI, PSNR, SI-SDR, SNR) and generate detailed report",
)
```

**Problem:**
- Pominje **SNR** koji je uklonjen
- Propusta **MCD, ViSQOL, SII, NCM** (4 nove metrike)

**Rjesenje:**
```python
help=(
    "Calculate audio quality metrics (PESQ, PSNR, SI-SDR, MCD, ViSQOL) "
    "and speech intelligibility metrics (STOI, SII, NCM); "
    "generates a detailed report."
),
```

---

### V6. `except Exception` je preteoko u svim wrapper funkcijama

**Fajl:** `src/utils/metrics.py:172, 236, 263, 332`
**Tip:** Error handling anti-pattern

```python
try:
    ...
except Exception as e:
    logger.warning(f"... failed: {e}")
    return None
```

**Problem:**
`Exception` hvata sve osim `SystemExit`/`KeyboardInterrupt` — ukljucujuci bug-ove u samom kodu (TypeError, AttributeError iz logickih gresaka). Takve greske se pretvaraju u "izracunavanje nije uspjelo" umjesto da se vide i poprave.

**Rjesenje:**
Suziti exception opseg:
```python
except (RuntimeError, ValueError, IndexError) as e:
    logger.warning(f"ViSQOL calculation failed: {e}")
    return None
```
Ili dodati `logger.exception(...)` koji loguje i stack trace (korisno za debug, ali ne stvara tihi failure).

---

## Srednji prioritet

### S1. SII i NCM koriste uniformne tezine umjesto ANSI band importance

**Fajl:** `src/utils/metrics.py:216, 307`
**Tip:** Nekompatibilnost sa standardom

```python
weights = np.ones(n_bands) / n_bands
```

**Problem:**
ANSI S3.5-1997 propisuje **razlicite tezine po pojasu** (tzv. band importance function). Uniformne tezine daju fundamentalno drugaciji rezultat. Ovo NIJE standardni SII/NCM.

**Rjesenje:**
- Opcija A: implementirati ANSI band importance tabelu
- Opcija B: koristiti postojecu biblioteku (npr. `pysiib`)
- Opcija C: eksplicitno prefiks u docstring-u: *"SIMPLIFIED SII approximation — NOT standard ANSI S3.5"*

---

### S2. SII i NCM koriste razlicite opsege frekvencija bez objasnjenja

**Fajl:** `src/utils/metrics.py:204-205 (SII)` i `:298-299 (NCM)`
**Tip:** Nekonzistentnost

```python
# SII
min_freq = 150.0
max_freq = min(fs / 2.0, 8500.0)

# NCM
min_freq = 100.0
max_freq = min(fs / 2.0, 8000.0)
```

**Problem:**
SII koristi 150–8500 Hz, NCM koristi 100–8000 Hz. Razlog nije dokumentovan.

**Rjesenje:**
Komentar koji objasnjava ili harmonizovati opsege. Ako postoji literatura za ove izbore (ANSI vs Loizou za NCM), link u komentaru.

---

### S3. `compute_metrics` forsira resampling na 16 kHz i za metrike kojima nije potreban

**Fajl:** `src/utils/metrics.py:353-370`
**Tip:** Gubitak kvaliteta

```python
metrics_sr = 16000 if sr not in [8000, 16000] else sr
...
result = {
    ...
    "sii": sii(ref_resampled, deg_resampled, metrics_sr),
    "ncm": ncm(ref_resampled, deg_resampled, metrics_sr),
}
```

**Problem:**
- PESQ/STOI zahtijevaju 8/16 kHz (OK)
- SII/NCM su sami implementirani i mogu raditi na bilo kom SR
- Za input od 44.1 kHz, forsirano resamplovanje na 16 kHz **gubi high-frequency informaciju** koju bi SII/NCM mogli iskoristiti

**Rjesenje:**
Prepustiti SII/NCM originalni SR:
```python
"sii": sii(ref_trimmed, deg_trimmed, sr),
"ncm": ncm(ref_trimmed, deg_trimmed, sr),
```

---

### S4. Redundantno racunanje — svih 8 metrika se racuna, samo dio se prikazuje

**Fajl:** `src/benchmark.py:273` i `src/utils/detailed_report_generator.py:20-104`
**Tip:** Performansa

**Problem:**
`compute_metrics` uvijek racuna svih 8 metrika, ali generator prikazuje selektivno po grupi (npr. `desynchronization` prikazuje samo mcd+visqol). Za 40 napada × 100 fajlova, znacajno vrijeme se trosi na metrike koje se nece prikazati.

**Rjesenje:**
Proslijediti listu trazenih metrika:
```python
def compute_metrics(reference, degraded, sr, metrics=None):
    if metrics is None:
        metrics = ALL_METRICS
    result = {}
    if "pesq" in metrics:
        result["pesq"] = pesq_wrapper(...)
    ...
```

Ili jos cistije — deklarisati relevantne metrike u `config.json` svakog napada i konsultovati ih:
```json
{
    "relevant_quality_metrics": ["pesq", "psnr"],
    "relevant_intelligibility_metrics": ["stoi"]
}
```

---

### S5. `watermarked_audio_quality` se duplicira u svakom attack zapisu

**Fajl:** `src/benchmark.py:285-287`
**Tip:** Struktura podataka

```python
results[filepath][attack_name] = {
    "accuracy": accuracy,
    "watermarked_audio_quality": watermarked_audio_quality,  # ista vrijednost za sve napade u fajlu!
    "attacked_audio_quality_wm": attacked_audio_quality_wm,
}
```

**Problem:**
`watermarked_audio_quality` je identicna za sve napade unutar istog fajla (zavisi samo od fajla i modela). Duplikacija od 40 puta po fajlu u JSON-u.

**Rjesenje:**
Pomjeriti na file-level:
```python
results[filepath] = {
    "watermarked_audio_quality": watermarked_audio_quality,
    "attacks": {
        attack_name: {
            "accuracy": ...,
            "attacked_audio_quality_wm": ...,
        }
    }
}
```
Potrebno azurirati `compute_mean_accuracy` i `aggregate_results` metode koje citaju strukturu.

---

### S6. Semanticka promjena mjerenja napada nije dokumentovana

**Fajl:** `src/benchmark.py:273`
**Tip:** Dokumentacija

Prije: metrike su mjerile `(original_clean, napadnuti_clean)` — "koliko je napad oslabio signal"
Sada: metrike mjere `(original_clean, napadnuti_watermarkovani)` — "koliko watermark + napad zajedno slabe signal"

Ova semanticka promjena **nigdje nije dokumentovana**. Citalac JSON/LaTeX rezultata ne zna sta tacno vrijednosti znace.

**Rjesenje:**
- Komentar u kodu iznad `compute_metrics` poziva
- Napomena u `README.md` ispod tabele metrika
- Napomena u docstring-u `compute_metrics`

---

### V7. Lazy import `visqol` — modul pada ako paket nije instaliran

**Fajl:** `src/utils/metrics.py:8`
**Tip:** Robusnost

```python
from visqol import VisqolApi
```

**Problem:**
Top-level import. Ako `visqol` nije instaliran, `metrics.py` se **ne moze uopste uciati**, sto obara i PESQ, STOI, PSNR, SI-SDR, MCD — sve.

**Rjesenje:**
Lazy import:
```python
def visqol_wrapper(reference, degraded, fs=16000):
    try:
        from visqol import VisqolApi
    except ImportError:
        logger.warning("visqol package not installed, skipping")
        return None
    ...
```

---

### S7. Multi-model mode ne handluje pad jednog modela

**Fajl:** `src/run.py:263-274`
**Tip:** Robusnost

```python
for model_name in model_names:
    ...
    results, flattened_stats = run_single_model(...)
    all_results[model_name] = results
```

**Problem:**
Ako `run_single_model` padne (npr. model crash, OOM, Docker container problem), prekida se cijeli multi-model run. Sve sledeci modeli se preskacu.

**Rjesenje:**
```python
for model_name in model_names:
    try:
        results, flattened_stats = run_single_model(...)
        all_results[model_name] = results
        all_stats[model_name] = flattened_stats
    except Exception as e:
        logger.error(f"Model {model_name} failed: {e}, continuing with remaining models")
        continue

if not all_results:
    logger.error("All models failed; skipping comparative report")
    return
```

---

## Nizak prioritet (minor)

### M1. NCM `clip(0, 1)` gubi informaciju o negativnoj korelaciji

**Fajl:** `src/utils/metrics.py:326`

```python
norm_cov = np.clip(norm_cov, 0.0, 1.0)
```

Pearson korelacija je u `[-1, 1]`. Klipovanje na `[0, 1]` pretvara negativnu korelaciju (npr. -0.3) u 0 — gubi se razlika.

**Rjesenje:** `abs(norm_cov)` ili eksplicitno dokumentovati ponasanje.

---

### M2. Magic numbers svuda u `metrics.py`

**Primjeri:**
- `n_fft = 2048` (SII, NCM)
- `min_freq = 150.0`, `max_freq = 8500.0` (SII)
- `min_freq = 100.0`, `max_freq = 8000.0` (NCM)
- `-15.0, 15.0` (SII SNR clamp range)
- `1e-12` (epsilon)

**Rjesenje:** izdvojiti u module-level konstante sa komentarima:
```python
SII_FFT_SIZE = 2048
SII_MIN_FREQ_HZ = 150.0
SII_MAX_FREQ_HZ = 8500.0   # ANSI S3.5 koristi do 8 kHz; 8.5 kHz kao gornja granica
SII_SNR_CLAMP_DB = 15.0
```

---

### M3. `compute_metrics` nema type hints dok ostale funkcije imaju

**Fajl:** `src/utils/metrics.py:336`

```python
def compute_metrics(reference, degraded, sr):
```

Ostale funkcije imaju `ref: np.ndarray, deg: np.ndarray, fs: int`. Nekonzistentno.

**Rjesenje:**
```python
def compute_metrics(
    reference: np.ndarray, degraded: np.ndarray, sr: int,
) -> dict:
```

---

### M4. Docstring "visqol (optional)" je zbunjujuc

**Fajl:** `src/utils/metrics.py:347`

```python
Quality: pesq, psnr, si_sdr, mcd, visqol (optional)
```

Nista u kodu nije opcionalno — wrapper uvijek vraca `None` na gresku, nema zastavice. "Optional" suggests konfigurabilnost koja ne postoji.

**Rjesenje:** ukloniti "(optional)" ili razjasniti: *"visqol (may be None if library not installed or fails)"*.

---

### M5. `mcd` uzima samo kratak vremenski prozor ako signali imaju razlicitu duzinu

**Fajl:** `src/utils/metrics.py:260-261`

```python
min_len = min(mfcc_ref.shape[1], mfcc_deg.shape[1])
diff = mfcc_ref[:, :min_len] - mfcc_deg[:, :min_len]
```

Nema DTW (Dynamic Time Warping) poravnanja. Za `time_stretch` i `pitch_shift` napade, frame-to-frame poredjenje po istom indeksu daje varljiv rezultat.

**Rjesenje:** u docstring navesti da MCD koristi indeks-poravnanje (ne DTW) i da je zato nepouzdan za napade koji mijenjaju tajming. (DTW implementacija je znacajniji rad.)

---

### M6. `DetailedReportGenerator.create_radar_chart` je definisan ali se ne koristi

**Fajl:** `src/utils/detailed_report_generator.py:436-499`
**Tip:** Dead code

Metod postoji ali `generate_latex_report` umjesto njega koristi `benchmark_chart.png` (bar chart) iz osnovnog generatora.

**Rjesenje:** obrisati ili povezati u izvjestaj.

---

### M7. `total_attacks` parametar u `generate_full_report` se ne koristi

**Fajl:** `src/utils/detailed_report_generator.py:696-697`

```python
def generate_full_report(self, results, model_name="DeepMark", total_attacks=None):
    """
    ...
    total_attacks: Total number of available attacks (unused, kept for API compatibility)
    """
```

**Rjesenje:** ako nema planiranog use case-a, obrisati argument (API nije public). Ako je held over za kompatibilnost, komentar ne objasnjava ko ga zove sa tim argumentom.

---

### M8. Hardkodirana imena napada u `ATTACK_GROUPS` bez validacije

**Fajl:** `src/utils/attack_groups.py:3-77`

Imena napada su stringovi. Ako se klasa napada preimenuje (npr. `GaussianNoiseAttack` → `GaussianNoiseAtk`), rucno treba azurirati `attack_groups.py`. Nema test-a koji provjerava.

**Rjesenje:** dodati unit test:
```python
def test_all_grouped_attacks_exist_as_plugins():
    pm = PluginManager()
    available = set(pm.get_attacks().keys())
    for group_key, group in ATTACK_GROUPS.items():
        for attack in group["attacks"]:
            assert attack in available, f"{attack} in {group_key} not found in plugins"
```

---

### M9. `group_attacks` shadowanje imena (lokalna varijabla ista kao importovana funkcija)

**Fajl:** `src/run.py:125`

```python
group_attacks = get_attacks_for_groups(args.attack_group)
```

Ako se u istom scope-u ikada importuje `from utils.attack_groups import group_attacks` (funkcija), doslo bi do konflikta. Trenutno nije problem, ali zbunjujuce.

**Rjesenje:** preimenovati lokalnu varijablu:
```python
attacks_from_groups = get_attacks_for_groups(args.attack_group)
```

---

### M10. Komentar `# Exit if no files found` je trivijalan, obrisan — ali primjer siri diskusiju o komentarima

**Fajl:** `src/run.py:152`

Obrisan je nepotreban komentar, ali `run.py` i dalje ima komentare poput `# --- Multi-model mode ---` koji bi bili bolji kao docstring-ovi funkcija.

---

### M11. `requirements.txt` — zadnja linija bez newline-a

**Fajl:** `requirements.txt`

Zadnja linija (`audiocomplib==0.2.0`) nema trailing `\n`. POSIX konvencija + neki alati upozoravaju.

**Rjesenje:** dodati prazan red na kraj.

---

### M12. Neki napadi u README primjerima mozda imaju pogresna imena

**Fajl:** `README.md`

Primjer `--wm_models AudioSealModel AwareModel PerthModel` — provjeriti da li se klasa zaista zove `AwareModel` (mozda je `AWAREModel` ili drugacije).

---

### M13. ViSQOL 0.0.4 je vrlo rana verzija

**Fajl:** `requirements.txt`

`visqol==0.0.4` — provjeriti postoji li stabilnija/novija verzija. Rana verzija = moguci bug-ovi, nesigurna API.

---

### M14. `.gitignore` dodaje `generate_comparative.py` bez komentara

**Fajl:** `.gitignore`

```
generate_comparative.py
```

Nova osoba u projektu ne zna zasto je ovaj fajl ignorisan.

**Rjesenje:**
```
# Local helper script for manual comparative report testing
generate_comparative.py
```

---

## Semanticka pitanja (potrebna diskusija)

Ovo su odluke koje nisu "bug" ali ocjena zavisi od namjere. Treba eksplicitno razgovarati.

### D1. Da li MCD i ViSQOL ima smisla za `desynchronization` grupu?

**Fajl:** `src/utils/detailed_report_generator.py:23`

```python
"desynchronization": ["mcd", "visqol"],
```

Za `TimeStretchAttack`, `PitchShiftAttack`:
- **MCD**: bez DTW daje ogromne vrijednosti jer su frame-ovi pomjereni
- **ViSQOL**: ima neku toleranciju na poravnanje ali ne za velika desync-a

Trenutno se prikazuju ali vrijednosti su varljive. Alternativa: isprazniti listu kao za `temporal_editing` (Cut/Crop).

---

### D2. Da li je nova semantika mjerenja napada (watermarkovani+napadnuti) dobar izbor?

Prije: `metric(original, attack(clean_audio))` — cisti efekat napada
Sada: `metric(original, attack(watermarked_audio))` — kombinovani efekat embedding+napad

- **Za** (realizam): to je ono sto korisnik cuje
- **Protiv** (dekompozicija): ne vidi se koliko od degradacije je od watermarka, koliko od napada

Za svrhu benchmarkinga watermark **robustnosti**, trenutni pristup je OK. Za dijagnostiku embedding-a, razlika je korisna. Razmotriti da li je vrijedno implementirati oboje (skupo).

---

### D3. Koji je autoritet o tome koje metrike su relevantne za koju grupu?

Trenutno se relevantnost definise u **dva mjesta**:
1. `GROUP_QUALITY_METRICS` / `GROUP_INTELLIGIBILITY_METRICS` (srednje-grupa)
2. `AUDIO_EDITING_SUBGROUPS` (pod-grupe audio_editing-a)

Oba su u `detailed_report_generator.py` i importuju se u `comparative_report_generator.py`. Ako se neko odluci refaktorisati, lako se rasinkronizuju.

**Rjesenje:** premjestiti u `config.json` svakog napada — pojedinacni napad sam deklarise svoje relevantne metrike. Generatori citaju iz config-a. Single source of truth.

---

## Sumarna tabela

| ID | Fajl | Kategorija | Prioritet | Opis |
|---|---|---|---|---|
| K1 | metrics.py | Bug | Kriticno | SII FFT samo prvih 2048 uzoraka |
| K2 | metrics.py | Matematika | Kriticno | SII pogresna definicija suma |
| K3 | benchmark.py | Regresija | Kriticno | SNR uklonjen bez zamjene |
| K4 | run.py | Bug | Kriticno | `to_json_safe` None → "N/A" razbija JSON round-trip |
| V1 | metrics.py | Performanse | Visok | VisqolApi() instancira pri svakom pozivu |
| V2 | detailed_report_generator.py | Metodologija | Visok | PESQ za process_disruption stvara laznu sigurnost |
| V4 | metrics.py | Kompatibilnost | Visok | MCD bez standardnog scaling faktora |
| V5 | run.py | Dokumentacija | Visok | Zastareli help tekst za --calculate_quality_metrics |
| V6 | metrics.py | Anti-pattern | Visok | `except Exception` preteoko |
| V7 | metrics.py | Robusnost | Visok | ViSQOL top-level import |
| S1 | metrics.py | Standard | Srednji | SII/NCM uniformne tezine umjesto ANSI |
| S2 | metrics.py | Konzistentnost | Srednji | Razliciti freq opsezi SII vs NCM |
| S3 | metrics.py | Kvalitet | Srednji | SII/NCM forsirano resamplovano na 16 kHz |
| S4 | benchmark.py | Performanse | Srednji | Racuna se 8 metrika, prikazuje manje |
| S5 | benchmark.py | Struktura | Srednji | watermarked_audio_quality duplirano |
| S6 | benchmark.py | Dokumentacija | Srednji | Semanticka promjena nije dokumentovana |
| S7 | run.py | Robusnost | Srednji | Multi-model pad se ne handluje |
| M1 | metrics.py | Preciznost | Nizak | NCM clip(0,1) gubi negativnu korelaciju |
| M2 | metrics.py | Stil | Nizak | Magic numbers |
| M3 | metrics.py | Stil | Nizak | compute_metrics bez type hints |
| M4 | metrics.py | Dokumentacija | Nizak | "visqol (optional)" zbunjujuce |
| M5 | metrics.py | Nepreciznost | Nizak | MCD bez DTW |
| M6 | detailed_report_generator.py | Dead code | Nizak | create_radar_chart nije pozvan |
| M7 | detailed_report_generator.py | API | Nizak | total_attacks parametar nekoriscen |
| M8 | attack_groups.py | Test coverage | Nizak | Hardkodirana imena bez validacije |
| M9 | run.py | Stil | Nizak | group_attacks shadowing |
| M11 | requirements.txt | Stil | Nizak | Nedostaje trailing newline |
| M12 | README.md | Dokumentacija | Nizak | Provjeriti imena modela u primjerima |
| M13 | requirements.txt | Zavisnost | Nizak | ViSQOL 0.0.4 rana verzija |
| M14 | .gitignore | Dokumentacija | Nizak | Nedostaje komentar za ignored fajl |
| D1 | detailed_report_generator.py | Diskusija | — | MCD/ViSQOL za desynchronization |
| D2 | benchmark.py | Diskusija | — | Nova semantika mjerenja napada |
| D3 | detailed_report_generator.py | Dizajn | — | SSOT za "koje metrike su relevantne" |

---

## Preporuceni redoslijed rjesavanja

**Prije merge-a u `dev`:**
1. K3 (vratiti SNR)
2. K4 (to_json_safe None handling)
3. V5 (help tekst)
4. V7 (lazy visqol import)

**U narednom sprintu:**
5. K1, K2 (SII bug-ovi)
6. V1 (ViSQOL cache)
7. V2 (prilagoditi metrike po grupi)
8. V4 (MCD scaling)
9. S4 (selektivno racunanje metrika)
10. S5 (restrukturiranje results-a)
11. S7 (multi-model resilience)

**Tehnicki dug (kasnije):**
- Svi S i M problemi
- Diskusije D1, D2, D3 sa timom

---

*Generisao: code review za `DPM-1576-add-metrics-to-benchmark` na datum 2026-04-27.*
