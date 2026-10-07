import inspect
import logging
import os
import re

import numpy as np
import soundfile as sf
import librosa

from deepmarkpy.core.base_model import implements_is_watermarked
from deepmarkpy.utils import efficiency
from deepmarkpy.plugin_manager import PluginManager
from deepmarkpy.utils.utils import load_audio
from deepmarkpy.utils.metrics import compute_metrics
from deepmarkpy.utils.metric_resolver import (
    EFFICIENCY_METRICS,
    MetricResolver,
    worst_case_of,
)


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


def resolve_cross_model_name(entry_kwargs, attacks_registry, models):
    """The second model ``CrossModelAttack`` re-embeds with.

    The attack reads this from its kwargs and, unlike every other plugin,
    has no fallback to its own config.json, so both run loops resolve it
    here -- the entry's own kwargs first, the plugin's config.json default
    second -- and hand it over. Raises before any audio is touched when the
    name is not a discovered model.
    """
    name = entry_kwargs.get(
        "different_model_name_cross_model",
        (attacks_registry["CrossModelAttack"].get("config") or {}).get(
            "different_model_name_cross_model"
        ),
    )
    if name not in models:
        raise ValueError(
            f"CrossModelAttack needs a second model, but "
            f"'{name}' is not among the "
            f"discovered models: {sorted(models)}. Set "
            f"attack_parameters.CrossModelAttack."
            f"different_model_name_cross_model in the config."
        )
    return name


def require_attacks_available(attack_types, attacks_registry, plugin_failures=None):
    """Raise when a requested attack is absent from ``attacks_registry``.

    A missing model raises too. Skipping the attack would finish the run with
    exit 0 and a report short of it. Most often the plugin failed to import
    because an optional dependency is not installed, so the import error is
    quoted when known.
    """
    missing = [
        name for name in attack_types
        if (name.split(":")[0] if ":" in name else name) not in attacks_registry
    ]
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


def declared_versions(attack_name, attacks_registry):
    """Version names an attack's own config.json declares.

    A single-version config has no named presets, so it reports the one
    implicit ``default``.
    """
    raw_config = (attacks_registry.get(attack_name) or {}).get("_raw_config")
    if raw_config and "default" in raw_config and isinstance(
        raw_config["default"], dict
    ):
        return [key for key in raw_config if not key.startswith("_")]
    return ["default"]


def is_multi_version(attack_name, attacks_registry):
    """Whether the attack declares more than one named preset."""
    return len(declared_versions(attack_name, attacks_registry)) > 1


def expand_attacks(attack_types, attacks_registry, parameters=None,
                   extra_versions=None):
    """Expand attacks into (class_name, display_name, kwargs_override, version) quads.

    A Codec2 bitrate list gives one entry per bitrate
    (``Codec2VocoderAttack_700``, ...), ``AttackName:version`` one entry for
    that version, and a bare multi-version attack one entry per version.

    Args:
        attack_types: attack specs, optionally ``AttackName:version``.
        attacks_registry: ``Benchmark.attacks``.
        parameters: ``(attack_name, version) -> params`` for one entry,
            queried by resolved version, or None for no overrides.
        extra_versions: ``{attack: {version: params}}`` for versions the
            config defines; they load the plugin's default preset and take
            every parameter from ``params``.

    ``version`` in each quad is the preset to load: None for a
    config-defined version, whose name appears in ``display_name``.
    """
    parameters = parameters or (lambda name, version: {})
    extra_versions = extra_versions or {}
    expanded = []

    def add(entry):
        """Append unless a row with this display name exists.

        A version listed explicitly can arrive again through its group;
        the explicit spec comes first and wins.
        """
        if any(existing[1] == entry[1] for existing in expanded):
            logger.debug(f"Already expanded, skipping duplicate: {entry[1]}")
            return
        expanded.append(entry)

    for atk_spec in attack_types:
        # Parse version suffix
        version = None
        atk_name = atk_spec
        if ":" in atk_spec:
            # At the first colon, as config validation and availability
            # checks split it: a class name cannot hold one, a version name
            # can.
            atk_name, _, version = atk_spec.partition(":")

        if atk_name not in attacks_registry:
            add((atk_name, atk_spec, {}, version))
            continue

        added = extra_versions.get(atk_name, {})

        config = attacks_registry[atk_name].get("config") or {}
        bitrate_key = next(
            (k for k in config if k == "bitrate_codec2" and isinstance(config[k], list)),
            None,
        )
        if bitrate_key:
            # Versions first, exactly as below, then one row per bitrate of
            # each: a bare name runs every version here too.
            raw = attacks_registry[atk_name].get("_raw_config") or {}
            is_multi = is_multi_version(atk_name, attacks_registry)
            if version:
                versions = [version]
            elif is_multi or added:
                versions = declared_versions(atk_name, attacks_registry) + list(added)
            else:
                versions = [None]
            for name in versions:
                # As in _version_entry: a config-defined version loads the
                # plugin's default and takes every parameter from the config.
                overrides = (
                    dict(added[name]) if name in added
                    else parameters(atk_name, name)
                )
                # Each declared version runs at its own preset's bitrates. An
                # override may replace the list itself, in which case it
                # decides how many runs there are.
                preset = raw.get(name) if isinstance(raw.get(name), dict) else config
                bitrates = overrides.get(
                    bitrate_key, preset.get(bitrate_key, config[bitrate_key]),
                )
                if not isinstance(bitrates, list):
                    bitrates = [bitrates]
                for val in bitrates:
                    if val not in _CODEC2_SUPPORTED:
                        logger.warning(
                            f"Skipping unsupported Codec2 bitrate: {val}. "
                            f"Supported: {sorted(_CODEC2_SUPPORTED)}"
                        )
                        continue
                    display = f"{atk_name}_{val}"
                    # Labelled as _version_entry labels: only when there is
                    # more than one version, so ':default' and a bare name
                    # on a single-version plugin merge into one row.
                    if name and (is_multi or added):
                        display += f" ({name})"
                    entry_kwargs = {**overrides, bitrate_key: val}
                    add((atk_name, display, entry_kwargs,
                         None if name in added else name))
            continue

        is_multi = is_multi_version(atk_name, attacks_registry)

        if version:
            add(_version_entry(
                atk_name, version, is_multi or bool(added), added, parameters,
            ))
        elif is_multi or added:
            # No version specified but presets exist: run every one, the
            # plugin's own and any the config defined.
            for name in declared_versions(atk_name, attacks_registry) + list(added):
                add(_version_entry(
                    atk_name, name, True, added, parameters,
                ))
        else:
            add((
                atk_name, atk_name, parameters(atk_name, None), None,
            ))
    return expanded


def instantiate_attack(attack_cls, class_name, version):
    """Construct an attack, passing ``version`` only when it accepts one.

    Decided by signature, so a ``TypeError`` raised inside a constructor
    that takes a version propagates instead of triggering a retry without it.
    """
    try:
        parameters = inspect.signature(attack_cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover -- exotic callables
        parameters = {}

    takes_version = "version" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    if takes_version:
        return attack_cls(version=version)

    if version and version != "default":
        # Running the default preset would label its results as this version.
        raise ValueError(
            f"{class_name} does not support versions, so version "
            f"'{version}' cannot be loaded: its constructor takes no "
            f"'version' argument. Accept 'version' in {class_name}.__init__ "
            f"and pass it to super().__init__(version=version), or select "
            f"the attack without a version."
        )
    return attack_cls()


# Characters a filename cannot hold on some platform: the path separators,
# and the ones Windows reserves. A version name may contain any of them.
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def audio_filename_label(display_name):
    """``display_name`` as it may appear in a saved-audio filename; results and reports keep it verbatim."""
    return _UNSAFE_FILENAME_CHARS.sub("_", display_name)


def _version_entry(atk_name, version, label_version, added, parameters):
    """One expanded entry for a named version of an attack."""
    display = f"{atk_name} ({version})" if label_version else atk_name
    if version in added:
        # Not a preset the plugin knows, so load its default and let the
        # config supply every parameter.
        return (atk_name, display, dict(added[version]), None)
    return (atk_name, display, parameters(atk_name, version), version)


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
        metric_resolver=None,
        **kwargs,
    ):
        """Embed and detect without applying any attacks.

        Returns per-file accuracy (and confidence where available).
        Used by the ``no_attacks`` mode to measure baseline fidelity.

        Computes whichever metrics ``metric_resolver`` enables on the
        watermarked-vs-original pair, so the no-attacks report can show how
        much the watermark itself perturbs the audio. No attacks run here,
        so there are no attack groups: the resolver's defaults apply.

        When ``save_audio`` is True, the watermarked audio for each file is
        written to ``output_dir``. Since this mode applies no attacks, only
        the watermarked signal is saved (no attacked variants).
        """
        if save_audio and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        resolver = metric_resolver or MetricResolver.from_attack_groups(
            calculate_quality_metrics=calculate_quality_metrics,
        )
        baseline_metrics = resolver.signal_metrics_for_group(None)

        if wm_model not in self.models:
            raise ValueError(
                f"Model '{wm_model}' not found. Available: {list(self.models.keys())}"
            )

        model_cls = self.models[wm_model]["class"]
        model_instance = model_cls()
        model_config = self.models[wm_model]["config"] or {}
        returns_confidence = model_config.get("returns_confidence", False)
        is_zero_bit = model_config.get("is_zero_bit", False)
        supports_detection = implements_is_watermarked(model_instance)

        if sampling_rate is None:
            sampling_rate = model_config["sampling_rate"]
            logger.info(f"Using default sampling rate {sampling_rate} for model {wm_model}")

        results = []
        detection_errors = []
        for filepath in filepaths:
            if verbose:
                logger.info(f"Processing file: {filepath}")

            audio, sampling_rate = load_audio(filepath, target_sr=sampling_rate)

            file_watermark = (
                watermark_data
                if watermark_data is not None
                else model_instance.generate_watermark()
            )

            timings = {}
            with efficiency.measure(timings, "embed_latency",
                                    resolver.is_enabled(None, "embed_latency")):
                watermarked_audio = model_instance.embed(
                    audio=audio, watermark_data=file_watermark,
                    sampling_rate=sampling_rate,
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
            # ``is_watermarked()`` is defined over the raw return of
            # ``detect()`` -- AudioSeal's reads the confidence out of the
            # (watermark, confidence) pair -- so the unsplit value is kept
            # rather than reassembled from the parts below.
            with efficiency.measure(timings, "detect_latency",
                                    resolver.is_enabled(None, "detect_latency")):
                detect_output = model_instance.detect(
                    watermarked_audio, sampling_rate,
                )
            if returns_confidence:
                detected_message, confidence = detect_output
            else:
                detected_message = detect_output

            if is_zero_bit:
                raw = detected_message.tolist() if isinstance(detected_message, np.ndarray) else detected_message
                accuracy = float(raw) * 100
            else:
                accuracy = self.compare_watermarks(file_watermark, detected_message)

            entry = {
                "file": os.path.basename(filepath),
                "filepath": filepath,
                "accuracy": accuracy,
            }
            entry.update(timings)

            # Whether the watermark was found is the model's decision, not a
            # threshold applied to its output. Recorded only when the model
            # answers it; the report drops the column otherwise rather than
            # guessing on the model's behalf.
            if supports_detection:
                try:
                    entry["detected"] = bool(
                        model_instance.is_watermarked(detect_output)
                    )
                except Exception as exc:  # noqa: BLE001 - one file, not the run
                    detection_errors.append(str(exc))
                    logger.warning(
                        "is_watermarked() failed for %s: %s", filepath, exc,
                    )
            if confidence is not None:
                entry["confidence"] = confidence

            if baseline_metrics:
                # Compare original vs watermarked (no attack between).
                # Captures how much the watermark itself perturbs the
                # signal -- the same baseline that ``run()`` records as
                # ``watermarked_audio_quality``.
                entry["watermarked_audio_quality"] = compute_metrics(
                    audio, watermarked_audio, sampling_rate,
                    metrics=baseline_metrics,
                )

            results.append(entry)

        # A count of 0/N reads as "the watermark was never found", which is
        # a measurement. If the model's own decision raised on every file it
        # is not one, so the column is withdrawn rather than filled with a
        # number that means the opposite of what happened.
        if supports_detection and len(detection_errors) == len(results) \
                and detection_errors:
            logger.error(
                "%s.is_watermarked() failed on every file (%s); the detection "
                "count is omitted from the report.",
                wm_model, detection_errors[0],
            )
            supports_detection = False
            for entry in results:
                entry.pop("detected", None)

        return {
            "is_zero_bit": is_zero_bit,
            "returns_confidence": returns_confidence,
            "supports_detection": supports_detection,
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
        metric_resolver=None,
        attack_parameters=None,
        extra_attack_versions=None,
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
            metric_resolver (MetricResolver, optional): decides which metrics
                are computed for each attack, from the config file's
                per-group matrix. Defaults to the built-in matrix declared
                by ``ATTACK_GROUPS``, honouring ``calculate_quality_metrics``.
            attack_parameters (callable, optional): ``(attack, version) ->
                params``, applied per expanded entry so an override on one
                version cannot reach another. See ``expand_attacks``.
            extra_attack_versions (dict, optional): versions the config
                defines that the plugin does not; see ``expand_attacks``.
            **kwargs: Additional parameters for specific attacks.

        Returns:
            dict: A dictionary containing benchmark results for each file and attack.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        resolver = metric_resolver or MetricResolver.from_attack_groups(
            calculate_quality_metrics=calculate_quality_metrics,
        )

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

        expanded_attacks = expand_attacks(
            attack_types, self.attacks,
            parameters=attack_parameters,
            extra_versions=extra_attack_versions,
        )
        # Before any audio, as detection reliability does, so an unknown
        # second model stops the run before the first file is embedded.
        for class_name, _display, overrides, _version in expanded_attacks:
            if class_name == "CrossModelAttack":
                resolve_cross_model_name(
                    {**kwargs, **overrides}, self.attacks, self.models,
                )

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
            file_timings = {}
            with efficiency.measure(file_timings, "embed_latency",
                                    resolver.is_enabled(None, "embed_latency")):
                watermarked_audio = model_instance.embed(
                    audio=audio, watermark_data=file_watermark,
                    sampling_rate=sampling_rate
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
            # This is the "no attack" baseline row every group's table is
            # read against, so it carries every metric any group asks for --
            # a metric enabled for one group only would otherwise have no
            # baseline to compare against in that group's table.
            results[filepath]["watermarked_audio_quality"] = compute_metrics(
                audio, watermarked_audio, sr_scalar,
                metrics=set(resolver.all_signal_metrics()),
            )

            # Apply each attack and compute metrics
            for attack_class_name, attack_display_name, attack_overrides, attack_version in expanded_attacks:
                if attack_class_name not in self.attacks:
                    logger.warning(f"Attack '{attack_class_name}' not found. Skipping.")
                    continue

                if verbose:
                    logger.info(f"  Applying attack: {attack_display_name}")

                attack_instance = instantiate_attack(
                    self.attacks[attack_class_name]["class"],
                    attack_class_name, attack_version,
                )
                attack_name = attack_display_name

                # Merge bitrate overrides into kwargs for this attack
                current_attack_kwargs = {**attack_kwargs, **attack_overrides}

                if attack_class_name == "CrossModelAttack":
                    # Read the entry's own kwargs, not the run-level ones:
                    # per-attack parameters travel with each expanded entry.
                    different_model_name = resolve_cross_model_name(
                        current_attack_kwargs, self.attacks, self.models,
                    )
                    logger.info(f"Different model is chosen and it's {different_model_name}")
                    different_model_cls = self.models[different_model_name]["class"]
                    different_model_instance = different_model_cls()
                    current_attack_kwargs[
                        "different_model_name_cross_model"] = different_model_name

                attack_timings = {}
                with efficiency.measure(attack_timings, "attack_latency",
                                        resolver.is_enabled(None, "attack_latency")):
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
                    attacked_filename = f"{base_filename}_{audio_filename_label(attack_name)}.wav"
                    attacked_path = os.path.join(output_dir, attacked_filename)
                    sf.write(attacked_path, attacked_to_save, sampling_rate)
                    if verbose:
                        logger.info(f"Saved attacked audio: {attacked_filename}")
                
                confidence = None
                with efficiency.measure(attack_timings, "detect_latency",
                                        resolver.is_enabled(None, "detect_latency")):
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
                    resolver, attack_name, audio, attacked_audio, sr_scalar,
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
                    # Embedding happens once per file, not once per attack,
                    # so its time is repeated here rather than measured
                    # again -- the attack tables need a figure per row.
                    **attack_timings,
                    **file_timings,
                }

                # Add confidence for models that return it
                if confidence is not None:
                    results[filepath]["attacks"][attack_name]["confidence"] = confidence

                if attack_class_name == "CrossModelAttack":
                    results[filepath]["attacks"][attack_name]["accuracy_cross_model"] = different_accuracy

            # Each file's results go out as soon as the file finishes, so
            # an interruption at file N of M keeps the files already done.
            if on_file_complete is not None:
                on_file_complete(filepath, results[filepath])

        return results

    @staticmethod
    def _compute_attack_quality(resolver, attack_name, original, attacked, sr):
        """Return the metrics this attack's group asked for.

        The resolver decides, so a metric switched on for one group is
        genuinely computed there and a metric switched off costs nothing.
        The comparison is always ``original`` vs the watermarked-then-
        attacked signal.
        """
        relevant = set(resolver.metrics_for_attack(attack_name))
        if not relevant:
            return None
        return compute_metrics(original, attacked, sr, metrics=relevant)

    def compute_mean_accuracy(self, results, resolver=None, is_zero_bit=False):
        """
        Compute per-attack statistics, with each attack's group deciding
        which metrics and which statistics it gets.

        Args:
            results: Dictionary of results from ``run()``.
            resolver: ``MetricResolver`` from the config file. Defaults to
                the built-in matrix with quality metrics off, which yields
                accuracy plus the always-on trio.
            is_zero_bit: Accuracy represents watermark detection rather than
                payload bit agreement, so BER must not be computed.

        Returns:
            Dictionary mapping each attack name to computed statistics.
            Keys are ``<metric>_<statistic>`` plus ``<metric>_n``, so a
            report reads exactly the columns its config asked for.
        """
        resolver = resolver or MetricResolver.from_attack_groups(
            calculate_quality_metrics=False,
        )

        # Which metrics an attack gets depends on its group, so the
        # accumulator is built per attack rather than once up front.
        attack_groups = {}

        attack_accuracies = {}

        for _, file_data in results.items():
            attacks_dict = file_data.get("attacks", {})
            for attack_name, file_metrics in attacks_dict.items():
                if attack_name not in attack_accuracies:
                    group_key = resolver.group_for_attack(attack_name)
                    attack_groups[attack_name] = group_key
                    attack_accuracies[attack_name] = {
                        "accuracy": [],
                        "accuracy_cross_model": [],
                        "confidence": [],
                        "detection_valid": [],
                        # metrics_for_attack lists what compute_metrics
                        # produces; the enabled timings are added here.
                        "metrics": {
                            m: [] for m in (
                                list(resolver.metrics_for_attack(attack_name))
                                + [m for m in EFFICIENCY_METRICS
                                   if resolver.is_enabled(group_key, m)]
                            )
                        },
                    }

                attack_accuracies[attack_name]["accuracy"].append(file_metrics["accuracy"])

                if "detection_valid" in file_metrics:
                    attack_accuracies[attack_name]["detection_valid"].append(
                        bool(file_metrics["detection_valid"])
                    )

                if "accuracy_cross_model" in file_metrics:
                    attack_accuracies[attack_name]["accuracy_cross_model"].append(
                        file_metrics["accuracy_cross_model"]
                    )

                if "confidence" in file_metrics:
                    attack_accuracies[attack_name]["confidence"].append(file_metrics["confidence"])

                quality = file_metrics.get("attacked_audio_quality_wm")
                if isinstance(quality, dict):
                    for m in attack_accuracies[attack_name]["metrics"]:
                        v = quality.get(m)
                        if v is not None:
                            attack_accuracies[attack_name]["metrics"][m].append(v)

                # Timings sit beside accuracy on the entry, not inside the
                # quality dict: they are not a comparison of two signals.
                for m in EFFICIENCY_METRICS:
                    if m not in attack_accuracies[attack_name]["metrics"]:
                        continue
                    v = file_metrics.get(m)
                    if v is not None:
                        attack_accuracies[attack_name]["metrics"][m].append(v)

        computed = {}

        for attack_name, acc in attack_accuracies.items():
            computed[attack_name] = {}
            group_key = attack_groups[attack_name]

            accuracies = [a for a in acc["accuracy"] if a is not None]
            arr = np.array(accuracies)

            computed[attack_name]["accuracy_n"] = len(accuracies)
            self._apply_statistics(
                computed[attack_name], "accuracy", arr,
                resolver.statistics_for(group_key, "accuracy"),
            )

            if resolver.is_enabled(group_key, "emr"):
                exact_recovery_count = int(np.sum(arr == 100.0))
                computed[attack_name]["emr_count"] = exact_recovery_count
                computed[attack_name]["emr_rate"] = (
                    float(exact_recovery_count / len(arr)) if len(arr) else 0.0
                )

            if not is_zero_bit and resolver.is_enabled(group_key, "ber"):
                ber_arr = 1.0 - (arr / 100.0)
                self._apply_statistics(
                    computed[attack_name], "ber", ber_arr,
                    resolver.statistics_for(group_key, "ber"),
                )

            validity = acc["detection_valid"]
            if validity:
                computed[attack_name]["detection_failures"] = int(
                    len(validity) - sum(validity)
                )

            if acc["accuracy_cross_model"]:
                cross = [a for a in acc["accuracy_cross_model"] if a is not None]
                computed[attack_name]["accuracy_cross_model_mean"] = float(np.mean(cross))
                computed[attack_name]["accuracy_cross_model_n"] = len(cross)

            for m, vals in acc["metrics"].items():
                if vals:
                    m_arr = np.array(vals)
                    computed[attack_name][f"{m}_n"] = len(vals)
                    self._apply_statistics(
                        computed[attack_name], m, m_arr,
                        resolver.statistics_for(group_key, m),
                    )

        return computed

    @staticmethod
    def _apply_statistics(target, prefix, arr, statistics):
        """Apply selected statistics to an array and store with prefix."""
        if len(arr) == 0:
            return
        statistics = set(statistics)
        if "mean" in statistics:
            target[f"{prefix}_mean"] = float(np.mean(arr))
        if "std" in statistics:
            target[f"{prefix}_std"] = (
                float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            )
        if "median" in statistics:
            target[f"{prefix}_median"] = float(np.median(arr))
        if "p5" in statistics:
            target[f"{prefix}_p5"] = float(np.percentile(arr, 5))
        if "p10" in statistics:
            target[f"{prefix}_p10"] = float(np.percentile(arr, 10))
        if "p95" in statistics:
            target[f"{prefix}_p95"] = float(np.percentile(arr, 95))
        if "p99" in statistics:
            target[f"{prefix}_p99"] = float(np.percentile(arr, 99))
        if "worst_case" in statistics:
            # Direction-aware: the worst latency is the slowest, not the
            # fastest. The prefix is the metric name.
            target[f"{prefix}_worst_case"] = worst_case_of(arr, prefix)


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
