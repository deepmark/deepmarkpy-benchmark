import inspect
import logging
import os

import numpy as np
import soundfile as sf
import librosa

from deepmarkpy.plugin_manager import PluginManager
from deepmarkpy.utils.utils import load_audio
from deepmarkpy.utils.metrics import ALL_METRICS, compute_metrics
from deepmarkpy.utils.attack_groups import get_metrics_for_attack


logger = logging.getLogger(__name__)


# Attacks whose calling convention differs from the plain apply(audio, **kwargs).
_TUPLE_RETURNING_ATTACKS = {"CrossModelAttack"}
_NEEDS_CLEAN_ORIGINAL = {"ZeroBitCollusionAttack"}
_CODEC2_SUPPORTED = {700, 1200, 1300, 1400, 1600, 2400, 3200}

_BENCHMARK_INTERNAL_KEYS = frozenset({
    "sampling_rate", "model", "models", "watermark_data",
    "orig_audio", "original_audio_collusion",
})


def apply_attack(attack_instance, attack_class_name, target_audio, clean_audio, attack_kwargs):
    """Apply an attack, honoring the conventions some attacks require.

    ``target_audio`` is the signal being attacked; ``clean_audio`` is the
    un-watermarked reference, which collusion-style attacks splice from and
    which must never be the same array as ``target_audio`` (that would make
    the attack a no-op).

    Returns ``(attacked_audio, extra)`` where ``extra`` is the second element
    for tuple-returning attacks and ``None`` otherwise.

    Both the benchmark run loop and the detection-reliability pass call this,
    so an attack's calling convention is defined once.
    """
    kwargs = dict(attack_kwargs)
    extra = None

    if attack_class_name in _NEEDS_CLEAN_ORIGINAL:
        kwargs["original_audio_collusion"] = clean_audio

    if attack_class_name in _TUPLE_RETURNING_ATTACKS:
        attacked_audio, extra = attack_instance.apply(target_audio, **kwargs)
    else:
        attacked_audio = attack_instance.apply(target_audio, **kwargs)
        if isinstance(attacked_audio, tuple):
            # Unpacking is driven by the registry above, so an unregistered
            # tuple would otherwise reach the metrics as a 0-d object array.
            raise TypeError(
                f"{attack_class_name}.apply() returned a tuple; attacks must "
                "return the attacked audio unless their class name is listed "
                "in benchmark._TUPLE_RETURNING_ATTACKS"
            )

    if isinstance(attacked_audio, np.ndarray):
        attacked_audio = np.squeeze(attacked_audio)
    return attacked_audio, extra


def require_attacks_available(attack_types, attacks_registry, plugin_failures=None):
    """Raise when a requested attack is absent from ``attacks_registry``.

    A missing model already raises; a missing attack used to warn and skip, so
    a run finished with exit 0 and a report whose attack count had silently
    dropped. Most often the plugin failed to import because an optional
    dependency is not installed, so the import error is quoted when known.
    """
    missing = [name for name in attack_types if name not in attacks_registry]
    if not missing:
        return

    message = [f"Requested attacks are not available: {sorted(missing)}"]
    failures = plugin_failures or {}
    if failures:
        message.append("Plugins that failed to import:")
        message += [f"  {module}: {error}" for module, error in sorted(failures.items())]
    else:
        message.append(f"Discovered attacks: {sorted(attacks_registry)}")

    raise ValueError("\n".join(message))


def expand_attacks(attack_types, attacks_registry):
    """Expand attacks whose config carries a bitrate list into separate entries.

    E.g. Codec2VocoderAttack with ``bitrate_codec2=[700, 1300, 2400]`` becomes
    ``Codec2VocoderAttack_700``, ``_1300``, ``_2400``. Returns a list of
    ``(class_name, display_name, kwargs_override)`` triples. Shared so every
    mode reports the same attack set under the same labels.
    """
    expanded = []
    for atk_name in attack_types:
        if atk_name not in attacks_registry:
            expanded.append((atk_name, atk_name, {}))
            continue
        config = attacks_registry[atk_name].get("config") or {}
        bitrate_key = next(
            (k for k in config if k == "bitrate_codec2" and isinstance(config[k], list)),
            None,
        )
        if bitrate_key:
            for val in config[bitrate_key]:
                if val not in _CODEC2_SUPPORTED:
                    logger.warning(
                        f"Skipping unsupported Codec2 bitrate: {val}. "
                        f"Supported: {sorted(_CODEC2_SUPPORTED)}"
                    )
                    continue
                expanded.append((atk_name, f"{atk_name}_{val}", {bitrate_key: val}))
        else:
            expanded.append((atk_name, atk_name, {}))
    return expanded


class Benchmark:
    """
    A class to perform various attacks on watermarking models and benchmark their performance.
    """

    def __init__(self, external_plugins_dir=None):
        """
        Initialize Benchmark class with PluginManager.

        Args:
            external_plugins_dir: Optional directory of third-party plugin
                directories, forwarded to PluginManager (defaults to the
                DEEPMARK_PLUGINS_DIR environment variable).
        """
        self.plugin_manager = PluginManager(external_plugins_dir=external_plugins_dir)
        # Now these are dicts of the form { "class_name": {"class": ActualClass, "config": {...}} }
        self.attacks = self.plugin_manager.get_attacks()
        self.models = self.plugin_manager.get_models()

    def get_available_args(self):
        valid_args = {}
        models = self.models.keys()
        attacks = self.attacks.keys()
        for attack in attacks:
            config = self.attacks[attack]["config"]
            if config is not None:
                for key, value in config.items():
                    if key in valid_args and valid_args[key] != value:
                        logger.warning(
                            f"Config parameter '{key}' defined by multiple attacks with "
                            f"different defaults. Last value wins. Consider using unique "
                            f"parameter names (e.g., '{key}_{attack.lower()}')."
                        )
                    valid_args[key] = value
        return list(models), list(attacks), valid_args

    @staticmethod
    def _log_plugin_entry(kind_label: str, name: str, entry: dict) -> None:
        """Log constructor params + config defaults for a single plugin."""
        plugin_cls = entry["class"]
        config = entry.get("config") or {}

        signature = inspect.signature(plugin_cls.__init__)
        params = [p for p in signature.parameters.values() if p.name != "self"]
        init_params = {
            p.name: (None if p.default is inspect.Parameter.empty else p.default)
            for p in params
        }

        logger.info(f"\n{kind_label}: {name}")
        logger.info(f"  - Constructor parameters: {init_params}")
        logger.info("  - Argument defaults:")
        if config:
            for key, val in config.items():
                logger.info(f"    {key}: {val}")
        else:
            logger.info("    (none found)")

    def show_available_plugins(self):
        """
        Print out all discovered models and attacks, including any __init__ parameters
        and key-value pairs from config.json (defaults).
        """
        logger.info("===== Available Models =====")
        for name, entry in self.models.items():
            self._log_plugin_entry("Model", name, entry)

        logger.info("\n===== Available Attacks =====")
        for name, entry in self.attacks.items():
            self._log_plugin_entry("Attack", name, entry)

    def run_no_attacks(
        self,
        filepaths,
        wm_model,
        watermark_data=None,
        sampling_rate=None,
        verbose=False,
        calculate_quality_metrics=False,
        save_audio=False,
        output_dir=None,
        **kwargs,
    ):
        """Embed and detect without applying any attacks.

        Returns per-file accuracy (and confidence where available).
        Used with ``--no_attacks`` to measure baseline model fidelity.

        When ``calculate_quality_metrics`` is True, also computes the full
        metric suite (PESQ, PSNR, SI-SDR, MCD, ViSQOL, STOI, SII, NCM,
        NISQA x5) on the watermarked-vs-original pair so the no-attacks
        report can show how much the watermark itself perturbs the audio.

        When ``save_audio`` is True, the watermarked audio for each file is
        written to ``output_dir``. Since this mode applies no attacks, only
        the watermarked signal is saved (no attacked variants).
        """
        if save_audio and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        if wm_model not in self.models:
            raise ValueError(
                f"Model '{wm_model}' not found. Available: {list(self.models.keys())}"
            )

        model_cls = self.models[wm_model]["class"]
        model_instance = model_cls()
        model_config = self.models[wm_model]["config"] or {}
        returns_confidence = model_config.get("returns_confidence", False)
        is_zero_bit = model_config.get("is_zero_bit", False)

        if sampling_rate is None:
            sampling_rate = model_config["sampling_rate"]
            logger.info(f"Using default sampling rate {sampling_rate} for model {wm_model}")

        results = []
        for filepath in filepaths:
            if verbose:
                logger.info(f"Processing file: {filepath}")

            audio, sampling_rate = load_audio(filepath, target_sr=sampling_rate)

            file_watermark = (
                watermark_data
                if watermark_data is not None
                else model_instance.generate_watermark()
            )

            watermarked_audio = model_instance.embed(
                audio=audio, watermark_data=file_watermark, sampling_rate=sampling_rate,
            )

            # Save the watermarked audio. No attacks run in this mode, so the
            # watermarked signal is the only variant worth writing.
            if save_audio and output_dir:
                base_filename = os.path.splitext(os.path.basename(filepath))[0]
                watermarked_path = os.path.join(
                    output_dir, f"{base_filename}_watermarked.wav"
                )
                sf.write(watermarked_path, watermarked_audio, sampling_rate)

            confidence = None
            if returns_confidence:
                detected_message, confidence = model_instance.detect(
                    watermarked_audio, sampling_rate,
                )
            else:
                detected_message = model_instance.detect(
                    watermarked_audio, sampling_rate,
                )

            if is_zero_bit:
                raw = detected_message.tolist() if isinstance(detected_message, np.ndarray) else detected_message
                accuracy = float(raw) * 100
            else:
                accuracy = self.compare_watermarks(file_watermark, detected_message)

            entry = {
                "file": os.path.basename(filepath),
                "accuracy": accuracy,
            }
            if confidence is not None:
                entry["confidence"] = confidence

            if calculate_quality_metrics:
                # Compare original vs watermarked (no attack between).
                # Captures how much the watermark itself perturbs the
                # signal -- the same baseline that ``run()`` records as
                # ``watermarked_audio_quality``.
                entry["watermarked_audio_quality"] = compute_metrics(
                    audio, watermarked_audio, sampling_rate,
                    metrics=ALL_METRICS,
                )

            results.append(entry)

        return {
            "is_zero_bit": is_zero_bit,
            "returns_confidence": returns_confidence,
            "files": results,
        }

    def run(
        self,
        filepaths,
        wm_model,
        watermark_data=None,
        attack_types=None,
        sampling_rate=None,
        verbose=False,
        save_audio=False,
        output_dir="audio_processed",
        calculate_quality_metrics=True,
        crop_before_attack=None,
        on_file_complete=None,
        **kwargs,
    ):
        """
        Benchmark the watermarking models against selected attacks.

        Args:
            filepaths (str or list): Path(s) to the audio file(s) to benchmark.
            wm_model (str): The model to benchmark (e.g., 'AudioSeal', 'WavMark', 'SilentCipher').
            watermark_data (np.ndarray, optional): The binary watermark data to embed. Defaults to random message.
            attack_types (list, optional): A list of attack types to perform. Defaults to all available attacks.
            sampling_rate (int, optional): Target sampling rate for loading audio. Defaults to None.
            verbose (bool, optional): Print verbose info. Defaults to False.
            save_audio (bool, optional): Whether to save processed audio files. Defaults to False.
            output_dir (str, optional): Directory to save processed audio. Defaults to "audio_processed".
            **kwargs: Additional parameters for specific attacks.

        Returns:
            dict: A dictionary containing benchmark results for each file and attack.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        # Create output directory if it doesn't exist
        if save_audio:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Audio will be saved to: {output_dir}")

        # If user doesn't specify attacks, use them all. An explicitly empty
        # request is not the same thing: falling back there would silently run
        # the whole registry for a caller who asked for a specific, and
        # entirely unavailable, set.
        if attack_types is None:
            attack_types = list(self.attacks.keys())
        else:
            if not attack_types:
                raise ValueError(
                    "No attacks to run: the requested attack set is empty."
                )
            self._require_attacks_available(attack_types)

        expanded_attacks = expand_attacks(attack_types, self.attacks)

        results = {}

        if wm_model not in self.models:
            raise ValueError(
                f"Model '{wm_model}' not found. Available: {list(self.models.keys())}"
            )

        model_cls = self.models[wm_model]["class"]
        model_instance = model_cls()
        model_config = self.models[wm_model]["config"] or {}
        returns_confidence = model_config.get("returns_confidence", False)
        is_zero_bit = model_config.get("is_zero_bit", False)

        if sampling_rate is None:
            sampling_rate = self.models[wm_model]["config"]["sampling_rate"]
            logger.info(f"Using default sampling rate {sampling_rate} for model {wm_model}")

        all_attack_config_keys = set()
        for atk_entry in self.attacks.values():
            if atk_entry.get("config"):
                all_attack_config_keys.update(atk_entry["config"].keys())

        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in all_attack_config_keys or k in _BENCHMARK_INTERNAL_KEYS
        }

        attack_kwargs = {
            **filtered_kwargs,
            "model": model_instance,
            "watermark_data": watermark_data,
            "sampling_rate": sampling_rate,
            "models": self.models,
        }

        for filepath in filepaths:
            if verbose:
                logger.info(f"\nProcessing file: {filepath}")
            # File-level container: watermark-only quality is stored here
            # once, and per-attack data lives under the "attacks" key.
            results[filepath] = {"attacks": {}}

            # Get base filename without extension
            base_filename = os.path.splitext(os.path.basename(filepath))[0]

            # Generate a fresh watermark for each file if none was supplied by the user
            file_watermark = watermark_data if watermark_data is not None else model_instance.generate_watermark()
            attack_kwargs["watermark_data"] = file_watermark

            # Load audio
            audio, sampling_rate = load_audio(filepath, target_sr=sampling_rate)
            logger.info(f"Sampling rate is: {sampling_rate}")

            # Embed watermark
            watermarked_audio = model_instance.embed(
                audio=audio, watermark_data=file_watermark, sampling_rate=sampling_rate
            )

            # Optionally crop the beginning of the watermarked audio right
            # after embedding, so every downstream attack sees the cropped
            # signal. Apply the same crop to the original audio so quality
            # metrics and attacks that splice samples by index between the
            # two (CollusionAttack, ZeroBitCollusionAttack) stay length-
            # matched and time-aligned.
            if crop_before_attack is not None:
                if "CropBeginningAttack" in self.attacks:
                    pre_crop = self.attacks["CropBeginningAttack"]["class"]()
                    watermarked_audio = pre_crop.apply(
                        watermarked_audio,
                        sampling_rate=sampling_rate,
                        crop_percentage_beginning=crop_before_attack,
                    )
                    audio = pre_crop.apply(
                        audio,
                        sampling_rate=sampling_rate,
                        crop_percentage_beginning=crop_before_attack,
                    )
                else:
                    samples_to_crop = int(len(watermarked_audio) * (crop_before_attack / 100.0))
                    watermarked_audio = watermarked_audio[samples_to_crop:]
                    audio = audio[samples_to_crop:]
            attack_kwargs["orig_audio"] = audio

            # Save watermarked audio
            if save_audio:
                watermarked_filename = f"{base_filename}_watermarked.wav"
                watermarked_path = os.path.join(output_dir, watermarked_filename)
                sf.write(watermarked_path, watermarked_audio, sampling_rate)

            sr_scalar = int(sampling_rate) if isinstance(sampling_rate, (np.ndarray, list)) else sampling_rate

            # Watermark-only quality: computed once per file, not per attack.
            # Stored at file level to avoid duplicating the same values 40x.
            # Always-on metrics (PESQ/ViSQOL/STOI) are computed even when
            # the full metric flag is off, so the detailed report always
            # has a baseline row to compare attack rows against.
            wm_metrics = set(self.ALWAYS_ON_METRICS)
            if calculate_quality_metrics:
                wm_metrics.update(ALL_METRICS)
            results[filepath]["watermarked_audio_quality"] = compute_metrics(
                audio, watermarked_audio, sr_scalar, metrics=wm_metrics,
            )

            # Apply each attack and compute metrics
            for attack_class_name, attack_display_name, attack_overrides in expanded_attacks:
                if attack_class_name not in self.attacks:
                    logger.warning(f"Attack '{attack_class_name}' not found. Skipping.")
                    continue

                if verbose:
                    logger.info(f"  Applying attack: {attack_display_name}")

                attack_instance = self.attacks[attack_class_name]["class"]()
                attack_name = attack_display_name

                # Merge bitrate overrides into kwargs for this attack
                current_attack_kwargs = {**attack_kwargs, **attack_overrides}

                if attack_class_name == "CrossModelAttack":
                    different_model_name = kwargs.get("different_model_name_cross_model")
                    logger.info(f"Different model is chosen and it's {different_model_name}")
                    different_model_cls = self.models[different_model_name]["class"]
                    different_model_instance = different_model_cls()

                attacked_audio, different_watermark = apply_attack(
                    attack_instance,
                    attack_class_name,
                    target_audio=watermarked_audio,
                    clean_audio=audio,
                    attack_kwargs=current_attack_kwargs,
                )

                # Ensure consistent shape for all attacks
                if isinstance(attacked_audio, np.ndarray):
                    attacked_audio = np.squeeze(attacked_audio)

                # Save attacked audio. Use a separate variable so the 2D
                # reshape required by sf.write doesn't leak into detect(),
                # which expects a 1D signal.
                if save_audio:
                    attacked_to_save = (
                        np.expand_dims(attacked_audio, axis=1)
                        if attacked_audio.ndim == 1
                        else attacked_audio
                    )
                    attacked_filename = f"{base_filename}_{attack_name}.wav"
                    attacked_path = os.path.join(output_dir, attacked_filename)
                    sf.write(attacked_path, attacked_to_save, sampling_rate)
                    if verbose:
                        logger.info(f"Saved attacked audio: {attacked_filename}")
                
                confidence = None
                if returns_confidence:
                    detected_message, confidence = model_instance.detect(attacked_audio, sampling_rate)
                else:
                    detected_message = model_instance.detect(attacked_audio, sampling_rate)

                if attack_class_name == "CrossModelAttack":
                    different_detected_message = different_model_instance.detect(attacked_audio, sampling_rate)
                    diff_model_config = self.models.get(different_model_name, {}).get("config") or {}
                    diff_is_zero_bit = diff_model_config.get("is_zero_bit", False)
                    diff_returns_confidence = diff_model_config.get("returns_confidence", False)
                    if diff_is_zero_bit:
                        if isinstance(different_detected_message, np.ndarray):
                            different_accuracy = different_detected_message.tolist()
                        else:
                            different_accuracy = different_detected_message
                    elif diff_returns_confidence:
                        different_watermark_detected, _ = different_detected_message
                        different_accuracy = self.compare_watermarks(different_watermark, different_watermark_detected)
                    else:
                        different_accuracy = self.compare_watermarks(different_watermark, different_detected_message)
                

                attacked_audio_quality_wm = self._compute_attack_quality(
                    calculate_quality_metrics, attack_name,
                    audio, attacked_audio, sr_scalar,
                )

                if is_zero_bit:
                    raw = detected_message.tolist() if isinstance(detected_message, np.ndarray) else detected_message
                    accuracy = float(raw) * 100
                    detection_valid = True
                else:
                    # False when the detector returned nothing usable, in which
                    # case `accuracy` is the RANDOM_GUESS_ACCURACY sentinel
                    # rather than a measured bit-agreement.
                    detection_valid = not self._is_invalid_detection(
                        detected_message, file_watermark
                    )
                    accuracy = self.compare_watermarks(file_watermark, detected_message)

                results[filepath]["attacks"][attack_name] = {
                    "accuracy": accuracy,
                    "detection_valid": detection_valid,
                    "attack_snr_db": self._attack_snr_db(
                        watermarked_audio, attacked_audio
                    ),
                    "attacked_audio_quality_wm": attacked_audio_quality_wm,
                }

                # Add confidence for models that return it
                if confidence is not None:
                    results[filepath]["attacks"][attack_name]["confidence"] = confidence

                if attack_class_name == "CrossModelAttack":
                    results[filepath]["attacks"][attack_name]["accuracy_cross_model"] = different_accuracy

            # A long run used to hold every result in memory until the last
            # file finished, so an interruption at file N of M kept nothing.
            if on_file_complete is not None:
                on_file_complete(filepath, results[filepath])

        return results

    # Metrics computed for every attack, regardless of group definitions
    # or the ``calculate_quality_metrics`` flag. PESQ/ViSQOL/STOI are
    # core robustness signals and must always appear in the detailed
    # report; everything else is opt-in via the per-group whitelist.
    ALWAYS_ON_METRICS = ("pesq", "visqol", "stoi")

    @staticmethod
    def _compute_attack_quality(enabled, attack_name, original, attacked, sr):
        """Return per-attack quality metrics for the detailed report.

        Always computes PESQ, ViSQOL and STOI so the core robustness
        signals are present in every run. When ``enabled`` is True (the
        ``--calculate_quality_metrics`` flag is set) the per-group
        metric whitelist (see ``attack_groups.get_metrics_for_attack``)
        is also computed. The comparison is always ``original`` vs the
        watermarked-then-attacked signal.
        """
        relevant = set(Benchmark.ALWAYS_ON_METRICS)
        if enabled:
            relevant.update(get_metrics_for_attack(attack_name))
        if not relevant:
            return None
        return compute_metrics(original, attacked, sr, metrics=relevant)

    def compute_mean_accuracy(self, results):
        """
        Compute mean accuracy per attack (plus cross-model accuracy where available).

        Args:
            results: Dictionary of results from ``run()``

        Returns:
            Dictionary mapping each attack name to ``accuracy_mean``,
            ``accuracy_std``, ``accuracy_n``, ``detection_failures`` (files
            whose detector returned nothing usable), optionally
            ``accuracy_cross_model_mean``/``_n``, and ``<metric>_mean`` plus
            ``<metric>_n`` for each always-on quality metric (PESQ, ViSQOL,
            STOI). The ``_n`` counts state how many files each mean covers,
            which varies when a metric is unavailable for some clips.
        """
        attack_accuracies = {}

        for _, file_data in results.items():
            attacks_dict = file_data.get("attacks", {})
            for attack_name, metrics in attacks_dict.items():
                if attack_name not in attack_accuracies:
                    attack_accuracies[attack_name] = {
                        "accuracy": [],
                        "accuracy_cross_model": [],
                        "confidence": [],
                        "detection_valid": [],
                        "metrics": {m: [] for m in self.ALWAYS_ON_METRICS},
                    }

                attack_accuracies[attack_name]["accuracy"].append(metrics["accuracy"])

                if "detection_valid" in metrics:
                    attack_accuracies[attack_name]["detection_valid"].append(
                        bool(metrics["detection_valid"])
                    )

                if "accuracy_cross_model" in metrics:
                    attack_accuracies[attack_name]["accuracy_cross_model"].append(
                        metrics["accuracy_cross_model"]
                    )

                if "confidence" in metrics:
                    attack_accuracies[attack_name]["confidence"].append(metrics["confidence"])

                quality = metrics.get("attacked_audio_quality_wm")
                if isinstance(quality, dict):
                    for m in self.ALWAYS_ON_METRICS:
                        v = quality.get(m)
                        if v is not None:
                            attack_accuracies[attack_name]["metrics"][m].append(v)

        mean_accuracies = {}

        for attack_name, acc in attack_accuracies.items():
            mean_accuracies[attack_name] = {}

            accuracies = [a for a in acc["accuracy"] if a is not None]
            mean_accuracies[attack_name]["accuracy_mean"] = float(np.mean(accuracies))
            mean_accuracies[attack_name]["accuracy_n"] = len(accuracies)
            mean_accuracies[attack_name]["accuracy_std"] = (
                float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0
            )

            # Files whose detector returned nothing usable: their accuracy is
            # the random-guess sentinel, not a measurement.
            validity = acc["detection_valid"]
            if validity:
                mean_accuracies[attack_name]["detection_failures"] = int(
                    len(validity) - sum(validity)
                )

            if acc["accuracy_cross_model"]:
                cross = [a for a in acc["accuracy_cross_model"] if a is not None]
                mean_accuracies[attack_name]["accuracy_cross_model_mean"] = float(
                    np.mean(cross)
                )
                mean_accuracies[attack_name]["accuracy_cross_model_n"] = len(cross)

            for m, vals in acc["metrics"].items():
                if vals:
                    mean_accuracies[attack_name][f"{m}_mean"] = float(np.mean(vals))
                    mean_accuracies[attack_name][f"{m}_n"] = len(vals)

        return mean_accuracies


    # Accuracy returned when detection produces no usable watermark. 50%
    # matches the random-guess baseline for a uniform binary message, so
    # comparisons against this value cleanly identify "detector failed".
    RANDOM_GUESS_ACCURACY = 50.00

    def _require_attacks_available(self, attack_types):
        """Fail when an explicitly requested attack is not loadable."""
        require_attacks_available(
            attack_types, self.attacks,
            getattr(self.plugin_manager, "failed", None),
        )

    @staticmethod
    def _attack_snr_db(reference, attacked):
        """Signal-to-noise ratio of what an attack added, in dB.

        Makes each attack's real strength visible per file. Attacks that fix an
        absolute noise amplitude rather than an SNR vary here with input
        loudness, so this is what shows whether two noise attacks were applied
        at comparable severity on a given file.

        Returns None when the two signals cannot be compared sample-wise (any
        attack that changes length) or when the attack made no difference.
        """
        if not isinstance(attacked, np.ndarray) or not isinstance(reference, np.ndarray):
            return None
        ref, att = np.squeeze(reference), np.squeeze(attacked)
        if ref.shape != att.shape:
            return None
        noise_power = float(np.mean(np.square(att - ref)))
        signal_power = float(np.mean(np.square(ref)))
        if noise_power <= 0 or signal_power <= 0:
            return None
        return float(10 * np.log10(signal_power / noise_power))

    @staticmethod
    def _is_invalid_detection(detected, original) -> bool:
        """Return True when ``detected`` can't be compared against ``original``."""
        if detected is None:
            return True
        if isinstance(detected, np.ndarray) and detected.ndim == 0:
            return True
        if isinstance(detected, (list, np.ndarray)) and len(detected) == 0:
            return True
        if np.any(detected == np.array(None)):
            return True
        if len(original) != len(detected):
            return True
        return False

    def compare_watermarks(self, original, detected):
        """
        Compare the original and detected watermarks.

        Args:
            original (np.ndarray): The original binary watermark.
            detected (np.ndarray): The detected binary watermark.

        Returns:
            float: Detection accuracy as a percentage, or
            ``RANDOM_GUESS_ACCURACY`` (50.0) when the detected payload is
            missing, empty, wrong-length, or otherwise unusable.
        """
        if self._is_invalid_detection(detected, original):
            return self.RANDOM_GUESS_ACCURACY
        matches = np.sum(original == detected)
        return (matches / len(original)) * 100