"""DeepMark Benchmark command-line entry point.

The command line carries operational settings only -- which config files
to run, where the audio is, where reports go, seeding, verbosity, audio
dumping, and the plugin directory. Every measurement decision (mode,
models, attacks, attack parameters, metrics, statistics, duration groups,
crop) lives in a config file, so a run is reproducible from its config
plus an audio directory.

Pass one ``--config`` file per mode. Several may be given and run in the
order listed; each must declare a different mode.
"""

import argparse
import datetime
import json
import platform
import subprocess
import sys
import os
import random
import shutil
from dataclasses import dataclass
from typing import Optional

import logging

from deepmarkpy import __version__
from deepmarkpy.benchmark import Benchmark, expand_attacks
from deepmarkpy.config import (
    ConfigError,
    MODE_KEYS,
    VALID_MODES,
    init_template,
    load_config_data,
    load_configs,
)
from deepmarkpy.utils.attack_groups import GROUP_ORDER
from deepmarkpy.utils.report_generator import generate_benchmark_report
from deepmarkpy.utils.metric_resolver import NISQA_METRICS
from deepmarkpy.utils.metrics import nisqa_status
from deepmarkpy.utils.utils import load_env_file

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


import numpy as np


# Exit codes, so a wrapper script can tell a bad config from a failed run.
EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CONFIG_ERROR = 2

# Applied when neither the command line nor the config file says otherwise.
DEFAULT_REPORT_DIR = "report"

AUDIO_SUFFIXES = (".wav", ".mp3")


def to_json_safe(obj):
    """
    Recursively convert numpy types to native Python types
    so json.dump does not crash.
    """
    if obj is None:
        return "N/A"
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj


def from_json_safe(obj):
    """
    Inverse of to_json_safe: recursively convert "N/A" sentinel strings
    back into None so numeric consumers do not crash.
    """
    if obj == "N/A":
        return None
    if isinstance(obj, dict):
        return {k: from_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Operational settings
# ---------------------------------------------------------------------------

@dataclass
class RunSettings:
    """Operational settings for one mode, after CLI/config/default resolution."""

    wav_files_dir: Optional[str]
    report_dir: str
    seed: Optional[int]
    verbose: bool
    save_audio: bool


def _resolve_settings(args, config) -> RunSettings:
    """Merge command line, config ``general`` block, and built-in defaults.

    The command line wins wherever it says anything, because it is the
    per-invocation override; the config file is the stored intent. Only
    these six keys exist in both places -- everything measurement-related
    lives in the config file alone, so there is nothing else to reconcile.
    """
    general = config.general

    def pick(cli_value, key, default=None):
        if cli_value is not None:
            return cli_value
        value = general.get(key)
        return default if value is None else value

    return RunSettings(
        wav_files_dir=pick(args.wav_files_dir, "wav_files_dir"),
        report_dir=pick(args.report_dir, "report_dir", DEFAULT_REPORT_DIR),
        seed=pick(args.seed, "seed"),
        verbose=bool(pick(args.verbose, "verbose", False)),
        save_audio=bool(pick(args.save_audio, "save_audio", False)),
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="deepmark-benchmark",
        description=(
            "Run the DeepMark audio-watermarking benchmark. Measurement "
            "settings live in a JSON config file; the flags below are "
            "operational only."
        ),
        epilog=(
            "Start from a template:  deepmark-benchmark --init benchmark > "
            "configs/benchmark.json"
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        # Repeating the flag adds files rather than replacing the earlier
        # ones, which silently skipped a requested mode.
        action="extend",
        default=None,
        metavar="PATH",
        help=(
            "Config file(s) to run, one per mode, in the order given. Each "
            "must declare a different \"mode\". Required unless --init is "
            "used."
        ),
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=list(VALID_MODES),
        default=None,
        metavar="MODE",
        help=(
            "Print a fresh, fully-commented config file for MODE to stdout "
            "and exit. Redirect it to a file to start from it. Modes: "
            + ", ".join(VALID_MODES)
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help=(
            "Validate the config file(s) against the discovered plugins and "
            "exit without running anything."
        ),
    )

    parser.add_argument(
        "--wav_files_dir",
        type=str,
        default=None,
        help=(
            "Directory of .wav/.mp3 files to evaluate. Overrides "
            "general.wav_files_dir; required if neither sets it."
        ),
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help=(
            "Where reports and saved audio go. Overrides general.report_dir "
            f"(default: ./{DEFAULT_REPORT_DIR}). Its contents are deleted at "
            "the start of every run."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed the host-side RNGs so a run can be reproduced. Overrides "
            "general.seed. Omitted by default, which keeps drawing a fresh "
            "watermark per file and fresh attack noise per run. Seeds "
            "host-side randomness only: the diffusion, vae, "
            "speech_enhancement_2 and network_transmission services stay "
            "stochastic server-side."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Enable verbose logging. Overrides general.verbose.",
    )
    parser.add_argument(
        "--save_audio",
        action="store_true",
        default=None,
        help=(
            "Save watermarked and attacked audio to <report_dir>/audio/ for "
            "manual inspection. Overrides general.save_audio."
        ),
    )
    parser.add_argument(
        "--plugins_dir",
        type=str,
        default=None,
        help=(
            "Directory containing third-party plugin directories "
            "(attack.py/model.py + config.json). Overrides "
            "general.plugins_dir; also settable via the DEEPMARK_PLUGINS_DIR "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"deepmarkpy {__version__}",
    )

    _add_legacy_arguments(parser)
    return parser


# The pre-2.0 flags. They are assembled into a config mapping and handed
# to the same validator and the same run loop, so there is no second code
# path below this point -- a flag run and the config file it corresponds
# to produce the same reports.
_LEGACY_FLAGS = (
    "wm_model", "wm_models", "attack_types", "attack_groups",
    "no_attacks", "detection_reliability", "calculate_quality_metrics",
    "crop_before_attack",
)


def _add_legacy_arguments(parser):
    """Restore the flag interface that predates the config file.

    Everything the config file added -- per-group metrics and statistics,
    the efficiency section, duration groups, per-version attack
    parameters -- has no flag, and never had one. These cover exactly
    what was expressible before, so an existing script keeps working.
    """
    group = parser.add_argument_group(
        "compatibility",
        "The pre-2.0 flags, for scripts written against the previous "
        "release. Ignored when --config is given, which is the fuller "
        "interface: these cannot express per-group metrics, statistics, "
        "efficiency or duration groups.",
    )
    models = group.add_mutually_exclusive_group()
    models.add_argument(
        "--wm_model", type=str, default=None,
        help="Single watermarking model to benchmark.",
    )
    models.add_argument(
        "--wm_models", type=str, nargs="+", default=None, metavar="MODEL",
        help="Multiple watermarking models to benchmark and compare.",
    )
    group.add_argument(
        "--attack_types", type=str, nargs="*", default=None, metavar="ATTACK",
        help="Attack class names to apply. Passing the flag with no names "
             "runs them all, as it always did.",
    )
    group.add_argument(
        "--attack_groups", type=str, nargs="+", default=None, metavar="GROUP",
        help=f"Attack families to apply: {', '.join(GROUP_ORDER)}.",
    )
    group.add_argument(
        "--no_attacks", action="store_true", default=False,
        help="Skip all attacks; embed and detect only.",
    )
    group.add_argument(
        "--detection_reliability", action="store_true", default=False,
        help="Measure false positive and false negative rates.",
    )
    group.add_argument(
        "--calculate_quality_metrics", action="store_true", default=False,
        help="Compute the full per-group metric set rather than accuracy "
             "plus PESQ, ViSQOL and STOI.",
    )
    group.add_argument(
        "--crop_before_attack", type=float, default=None, metavar="PERCENT",
        help="Crop this percentage from the beginning of the watermarked "
             "audio before each attack.",
    )


def _report_config_error(problem):
    """Print a config problem as a block, and say nothing ran.

    Straight to stderr rather than through the logger: a timestamp and a
    module name on every line is exactly the noise that makes the actual
    message hard to pick out, and this is the last thing the user sees
    before the process exits.
    """
    rule = "=" * 72
    print(f"\n{rule}\n CONFIGURATION ERROR - the benchmark did not start\n{rule}",
          file=sys.stderr)
    print(str(problem).rstrip(), file=sys.stderr)
    print(f"{rule}\n", file=sys.stderr)


def _peek_plugins_dir(paths):
    """Read ``general.plugins_dir`` before the plugins are loaded.

    Plugin discovery has to happen before the config can be fully
    validated (attack names are checked against it), but the directory to
    discover from may itself be set in the config. This reads just that
    one key, tolerating any malformed file -- real validation runs later
    and reports it properly.
    """
    found = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            value = (raw.get("general") or {}).get("plugins_dir")
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        if value:
            found[path] = value

    if not found:
        return None
    distinct = set(found.values())
    if len(distinct) > 1:
        logger.warning(
            f"Config files disagree on general.plugins_dir ({sorted(distinct)}); "
            f"using {next(iter(found.values()))}. Plugins are discovered once "
            f"for the whole invocation."
        )
    return next(iter(found.values()))


def _legacy_flags_used(args):
    """The pre-2.0 flags this invocation actually set."""
    used = []
    for name in _LEGACY_FLAGS:
        value = getattr(args, name, None)
        # Identity, not equality: --crop_before_attack 0 is a flag that was
        # set, and 0.0 == False.
        if value is not None and value is not False:
            used.append(f"--{name}")
    return used


def _legacy_attack_parameters(leftovers, benchmark, parser):
    """Route ``--<param> <value>`` flags onto the attacks that declare them.

    The old CLI built one flag per parameter found in any plugin's
    config.json and passed them as a flat mapping, so a name shared by
    two attacks reached both. That is reproduced here by keying the
    parameter under every attack whose config declares it -- the config
    file's own routing is per attack, and this is the only faithful way
    to express a flat namespace in it.
    """
    declared = {}
    for attack_name, entry in benchmark.attacks.items():
        for key, default in (entry.get("config") or {}).items():
            if not key.startswith("_"):
                declared.setdefault(key, []).append((attack_name, default))

    parameters = {}
    index = 0
    while index < len(leftovers):
        token = leftovers[index]
        index += 1
        if not token.startswith("--"):
            parser.error(f"unrecognized arguments: {token}")

        name, _, inline = token[2:].partition("=")
        negated = name.startswith("no-") and name[3:] in declared
        key = name[3:] if negated else name
        if key not in declared:
            parser.error(
                f"unrecognized arguments: {token}. No discovered attack has "
                f"a parameter called '{key}'."
            )

        default = declared[key][0][1]
        if isinstance(default, bool):
            value = not negated
            if inline:
                parser.error(f"{token}: '{key}' is a flag, so it takes no value")
        else:
            if inline:
                raw = inline
            elif index < len(leftovers) and not leftovers[index].startswith("--"):
                raw = leftovers[index]
                index += 1
            else:
                parser.error(f"argument --{key}: expected one argument")
            value = _coerce_like(raw, default, key, parser)

        for attack_name, _default in declared[key]:
            parameters.setdefault(attack_name, {})[key] = value

    return parameters


def _coerce_like(raw, default, key, parser):
    """Parse ``raw`` into the type of the plugin's own default."""
    try:
        if isinstance(default, bool):
            return raw.lower() in ("1", "true", "yes")
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, (list, dict)):
            return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        parser.error(
            f"argument --{key}: '{raw}' is not a "
            f"{type(default).__name__}, which is what this parameter takes"
        )
    return raw


def _configs_from_flags(args, leftovers, benchmark, parser):
    """Validate the pre-2.0 flags as though they were a config file.

    Returns one ``ModeConfig`` per mode asked for -- two when
    ``--no_attacks`` and ``--detection_reliability`` are combined, which
    the old CLI ran as a pair.
    """
    models = list(args.wm_models or ([args.wm_model] if args.wm_model else []))
    parameters = _legacy_attack_parameters(leftovers, benchmark, parser)

    attacks = {}
    if args.attack_types is not None:
        # The flag with no names has always meant "run them all", which in
        # a config file is an empty selection rather than an empty list.
        if args.attack_types or args.attack_groups:
            attacks["list"] = list(args.attack_types)
    if args.attack_groups:
        attacks["groups"] = list(args.attack_groups)

    general = {}
    if args.wav_files_dir:
        general["wav_files_dir"] = args.wav_files_dir

    modes = []
    if args.no_attacks:
        modes.append("no_attacks")
    if args.detection_reliability:
        modes.append("detection_reliability")
    if not modes:
        modes.append("benchmark")

    built = []
    for mode in modes:
        data = {"mode": mode, "models": models}
        if general:
            data["general"] = dict(general)
        if args.calculate_quality_metrics:
            data["calculate_quality_metrics"] = True
        if "attacks" in MODE_KEYS[mode] and attacks:
            data["attacks"] = dict(attacks)
        if "attack_parameters" in MODE_KEYS[mode] and parameters:
            data["attack_parameters"] = {
                name: dict(values) for name, values in parameters.items()
            }
        # A crop of 0% crops nothing, which the old CLI accepted; the
        # config file spells that null, so the flag's 0 means the same.
        if "crop_before_attack" in MODE_KEYS[mode] \
                and args.crop_before_attack not in (None, 0):
            data["crop_before_attack"] = args.crop_before_attack
        built.append(load_config_data(
            data, source="<command line>",
            attacks_registry=benchmark.attacks,
            models_registry=benchmark.models,
        ))
    return built


def main(argv=None):
    # Must precede Benchmark(): plugin clients read their service port in
    # __init__, so a port set in .env has to be in the environment by then.
    load_env_file()

    parser = _build_parser()
    # Unknown flags are the attack parameters the old CLI generated from
    # the plugin configs, which cannot be declared before the plugins are
    # imported. They stay an error whenever a config file is in play.
    args, leftovers = parser.parse_known_args(argv)

    if args.init:
        print(init_template(args.init))
        return EXIT_OK

    legacy = _legacy_flags_used(args)

    if args.config and leftovers:
        parser.error(f"unrecognized arguments: {' '.join(leftovers)}")

    if not args.config:
        if not legacy and not leftovers:
            parser.error(
                "--config is required: pass one config file per mode. "
                "Create one with 'deepmark-benchmark --init benchmark > "
                "configs/benchmark.json'."
            )
        benchmark = Benchmark(external_plugins_dir=args.plugins_dir)
        try:
            configs = _configs_from_flags(args, leftovers, benchmark, parser)
        except ConfigError as exc:
            _report_config_error(exc)
            return EXIT_CONFIG_ERROR
    else:
        if legacy:
            logger.warning(
                f"Ignoring {', '.join(legacy)}: --config was given, and the "
                f"config file carries every measurement setting. Drop "
                f"--config to use the flags instead."
            )

        # Pre-flight: everything that does not need the plugin registry --
        # JSON syntax, structure, types, metrics, statistics. Importing the
        # plugins logs dozens of lines, and burying a misplaced comma under
        # them is exactly what makes a broken config hard to find.
        try:
            load_configs(args.config, quiet=True)
        except ConfigError as exc:
            _report_config_error(exc)
            return EXIT_CONFIG_ERROR

        plugins_dir = args.plugins_dir or _peek_plugins_dir(args.config)
        benchmark = Benchmark(external_plugins_dir=plugins_dir)

        # Second pass, now that names can be checked against what was
        # discovered.
        try:
            configs = load_configs(args.config, benchmark.attacks,
                                   benchmark.models)
        except ConfigError as exc:
            _report_config_error(exc)
            return EXIT_CONFIG_ERROR

    plans = []
    for config in configs:
        settings = _resolve_settings(args, config)
        # CLI overrides do not pass through config validation. Reject an
        # unusable NumPy seed before validation succeeds or output is cleared.
        if settings.seed is not None and not 0 <= settings.seed < 2**32:
            _report_config_error(
                f"{config.source}: seed must be between 0 and 4294967295; "
                f"got {settings.seed}."
            )
            return EXIT_CONFIG_ERROR
        if not settings.wav_files_dir and not args.validate_only:
            _report_config_error(
                f"{config.source}: no audio directory.\n"
                f"    Set general.wav_files_dir in the config file, or pass "
                f"--wav_files_dir."
            )
            return EXIT_CONFIG_ERROR
        plans.append((config, settings))

    unreachable = _unreachable_model_services(
        benchmark, configs, wait=not args.validate_only,
    )

    if args.validate_only:
        # A stopped container is not a broken config file. Report it, but
        # let the check pass -- validating a config before starting Docker
        # is exactly when this flag is most useful.
        for config, settings in plans:
            logger.info(
                f"{config.source}: valid. mode={config.mode}, "
                f"models={', '.join(config.models)}, "
                f"attacks={_describe_attack_selection(config)}, "
                f"audio={settings.wav_files_dir}, report={settings.report_dir}"
            )
        for config, _settings in plans:
            _log_attack_parameters(benchmark, config, only_configured=False)
        for message in unreachable:
            logger.warning(f"Not reachable right now, but not a config error: "
                           f"{message}")
        _warn_about_metric_services(configs)
        logger.info(
            f"{len(plans)} config file(s) validated; nothing was run "
            f"(--validate-only)."
        )
        return EXIT_OK

    if unreachable:
        for message in unreachable:
            logger.error(message)
        return EXIT_CONFIG_ERROR

    _warn_about_metric_services(configs)

    filepaths_by_dir = {}
    for config, settings in plans:
        if settings.wav_files_dir not in filepaths_by_dir:
            found = _collect_audio_files(settings.wav_files_dir)
            if found is None:
                return EXIT_RUNTIME_ERROR
            filepaths_by_dir[settings.wav_files_dir] = found

    # One shared report directory, cleared once. Each mode writes distinct
    # filenames into it, so running several in one invocation does not make
    # them overwrite each other -- but a *previous* run's output still goes.
    for report_dir in dict.fromkeys(s.report_dir for _, s in plans):
        _clean_report_dir(report_dir)

    exit_code = EXIT_OK
    for config, settings in plans:
        filepaths = filepaths_by_dir[settings.wav_files_dir]
        if settings.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled.")

        # Re-seeded per mode so each starts from the same draw, which keeps a
        # mode's results identical whether it runs alone or after another.
        if settings.seed is not None:
            random.seed(settings.seed)
            np.random.seed(settings.seed)
            logger.info(f"Host-side RNGs seeded with {settings.seed}")

        logger.info(
            f"{'=' * 60}\nRunning mode '{config.mode}' from {config.source} "
            f"on {len(filepaths)} file(s)\n{'=' * 60}"
        )
        _log_attack_parameters(benchmark, config, only_configured=True)
        if _MODE_RUNNERS[config.mode](benchmark, filepaths, config,
                                      settings) != EXIT_OK:
            exit_code = EXIT_RUNTIME_ERROR

    return exit_code


def _describe_attack_selection(config):
    """One-line summary of what a config's attack selection resolves to."""
    if config.mode == "no_attacks":
        return "none (this mode applies no attacks)"
    specs = config.selected_attack_specs()
    if specs is None:
        # Only benchmark mode expands an empty selection to everything;
        # detection reliability adds attacks to a baseline it always measures.
        if config.mode == "detection_reliability":
            return "none (no-attack baseline only)"
        return "all discovered attacks"
    return f"{len(specs)}: {', '.join(specs[:6])}" + (
        ", ..." if len(specs) > 6 else ""
    )


def resolved_attack_parameters(benchmark, config):
    """Effective parameters per report row, worked out before anything runs.

    The plugin's preset for that version with the config's override
    applied on top -- which is exactly what ``apply()`` will see. Keyed by
    the name the row carries in the report, so a number in a table can be
    traced back to the version that produced it without inferring it from
    the results.

    An empty selection means different things per mode, and this has to
    agree with the run loop: everything in ``benchmark`` mode, the
    no-attack baseline alone in ``detection_reliability``, and nothing at
    all in ``no_attacks``.
    """
    if "attacks" not in MODE_KEYS[config.mode]:
        return {}

    specs = config.selected_attack_specs()
    if specs is None:
        specs = sorted(benchmark.attacks) if config.mode == "benchmark" else []

    rows = {}
    for class_name, display, overrides, load_version in expand_attacks(
        specs, benchmark.attacks,
        parameters=config.parameters_for,
        extra_versions=config.synthetic_versions,
    ):
        entry = benchmark.attacks.get(class_name) or {}
        raw = entry.get("_raw_config") or {}
        if "default" in raw and isinstance(raw["default"], dict):
            base = raw.get(load_version or "default") or {}
        else:
            base = entry.get("config") or {}
        # A plugin's config.json may carry "_"-prefixed notes of its own;
        # they are documentation, not parameters.
        merged = {
            key: value for key, value in {**base, **overrides}.items()
            if not key.startswith("_")
        }
        if merged:
            rows[display] = merged
    return rows


def _log_attack_parameters(benchmark, config, only_configured):
    """Print which version runs with which values.

    Args:
        only_configured: log just the rows the config file changed. A full
            attack set is one row per attack, which is noise when nothing
            was overridden; --validate-only passes False to show everything
            it selected.
    """
    try:
        rows = resolved_attack_parameters(benchmark, config)
    except Exception as exc:
        logger.debug(f"Could not resolve attack parameters: {exc}")
        return

    touched = set()
    for (attack, _version) in config.parameter_overrides:
        touched.add(attack)
    touched.update(config.synthetic_versions)
    if only_configured:
        rows = {
            name: params for name, params in rows.items()
            if name.split(" (")[0].split("_")[0] in touched
            or name.split(" (")[0] in touched
        }
    if not rows:
        return

    logger.info(f"{config.source}: attack parameters in effect --")
    for name, params in rows.items():
        values = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        logger.info(f"    {name}: {values}")


def _collect_audio_files(wav_files_dir):
    """List the audio files in a directory, or None after logging why not."""
    try:
        names = os.listdir(wav_files_dir)
    except FileNotFoundError:
        logger.error(f"Audio directory not found: {wav_files_dir}")
        return None
    except OSError as exc:
        logger.error(f"Error accessing audio directory {wav_files_dir}: {exc}")
        return None

    filepaths = [
        os.path.join(wav_files_dir, name) for name in names
        if name.lower().endswith(AUDIO_SUFFIXES)
    ]
    if not filepaths:
        logger.error(f"No .wav or .mp3 files found in directory: {wav_files_dir}")
        return None
    logger.info(f"Found {len(filepaths)} audio file(s) in {wav_files_dir}.")
    return filepaths


# A model container that has just started may load its checkpoint lazily,
# on the first request. WavMark takes over ten seconds to answer when cold,
# so a single short probe reports a service that is starting as one that is
# down, and aborts a run that would have worked seconds later.
_SERVICE_PROBE_TIMEOUT_S = 10
_SERVICE_PROBE_BUDGET_S = 60


def _unreachable_model_services(benchmark, configs, wait=True):
    """Check every selected model's Docker service before anything runs.

    A model whose container is down otherwise fails midway through the
    first file, after the attack set has already been applied.

    Any HTTP response counts as reachable, including a 404: the root path
    is not part of the plugin contract, so only the connection matters.

    Args:
        wait: keep retrying a silent service until the probe budget runs
            out, which is what makes running straight after
            ``docker compose up -d`` work. Pass False to probe once.
    """
    import time

    import requests as _requests

    messages = []
    for model_name in dict.fromkeys(m for c in configs for m in c.models):
        model_instance = benchmark.models[model_name]["class"]()
        base_url = getattr(model_instance, "base_url", None)
        if not base_url:
            continue

        deadline = time.monotonic() + (_SERVICE_PROBE_BUDGET_S if wait else 0)
        announced = False
        while True:
            try:
                _requests.get(base_url, timeout=_SERVICE_PROBE_TIMEOUT_S)
                break
            except (_requests.ConnectionError, _requests.Timeout):
                if time.monotonic() >= deadline:
                    service = model_name.lower().replace("model", "")
                    messages.append(
                        f"Model '{model_name}' service did not answer at "
                        f"{base_url} within "
                        f"{_SERVICE_PROBE_BUDGET_S if wait else _SERVICE_PROBE_TIMEOUT_S}s. "
                        f"Start it with: docker compose up -d {service}"
                    )
                    break
                if not announced:
                    logger.info(
                        f"Waiting for the {model_name} service at {base_url} "
                        f"to answer (a cold container loads its checkpoint on "
                        f"the first request)..."
                    )
                    announced = True
    return messages


def _warn_about_metric_services(configs):
    """Warn when a configured metric's backing service cannot answer.

    Not an error: the run still produces every other metric. But an N/A
    column from an unreachable service is indistinguishable from one the
    metric genuinely could not score, so it has to be said up front.
    """
    wanted = set()
    for config in configs:
        wanted.update(config.resolver.all_signal_metrics())

    if wanted & set(NISQA_METRICS):
        status = nisqa_status()
        if not status["available"]:
            logger.warning(
                f"[W006] NISQA metrics are enabled but the service is "
                f"unavailable ({status['reason']}), so every NISQA column "
                f"will read N/A. Start it with: docker compose up -d nisqa"
            )

    if "visqol" in wanted:
        try:
            import visqol  # noqa: F401
        except ImportError:
            logger.warning(
                "[W007] ViSQOL is enabled but the optional 'visqol' package "
                "is not installed, so the ViSQOL column will read N/A. "
                "Install it with: pip install 'deepmarkpy[metrics]'"
            )


_DEEPMARK_ASSETS = {"deepmark.cls", "deepmark-logo.png", "deepmark-logo.pdf", "deepmark-logo.jpg"}


def _clean_report_dir(report_dir):
    """Remove generated files from report dir, preserving deepmark assets."""
    if not os.path.exists(report_dir):
        return
    doomed = [i for i in sorted(os.listdir(report_dir)) if i not in _DEEPMARK_ASSETS]
    if doomed:
        logger.info(
            f"Clearing {len(doomed)} item(s) from {report_dir}/ before this run: "
            + ", ".join(doomed[:10])
            + (" ..." if len(doomed) > 10 else "")
        )
    for item in os.listdir(report_dir):
        if item in _DEEPMARK_ASSETS:
            continue
        path = os.path.join(report_dir, item)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def _copy_deepmark_assets(src_dir, dst_dir):
    """Copy deepmark assets from src to dst directory."""
    os.makedirs(dst_dir, exist_ok=True)
    for name in _DEEPMARK_ASSETS:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))


def _log_efficiency(config, stats):
    """Print the timings this run measured, under the efficiency tag.

    Its own tag so a run's timing output can be read, or filtered out, on
    its own -- the numbers describe the machine rather than the
    watermarking method, and are the one part of a run that does not
    reproduce.
    """
    from deepmarkpy.utils import efficiency
    from deepmarkpy.utils.latex_helpers import metric_label

    metrics = config.resolver.metrics_for_group(None, bucket="efficiency")
    if not metrics or not stats:
        return

    efficiency.log("processing time measured this run (does not reproduce "
                   "across machines):")
    for attack, row in sorted(stats.items()):
        parts = []
        for metric in metrics:
            value = row.get(f"{metric}_mean")
            if value is not None:
                parts.append(f"{metric_label(metric)}={float(value):.4f}")
        if parts:
            efficiency.log("    %s: %s", attack, ", ".join(parts))


def _container_rows(benchmark, config, model_names):
    """Memory held by the services this run used, when the config asks.

    Gated on ``efficiency.metrics.container_footprint.enabled``, which is
    off by default: reading it shells out to the docker CLI, and a run
    that does not care should not pay for that or depend on docker being
    there at all. Covers the models, the dockerized attacks and the
    metric services -- everything the run actually touched.
    """
    if not config.resolver.is_enabled(None, "container_footprint"):
        return []

    from deepmarkpy.utils import efficiency

    entries = []
    for name in model_names:
        entry = benchmark.models.get(name)
        if not entry:
            continue
        try:
            url = getattr(entry["class"](), "base_url", None)
        except Exception:  # noqa: BLE001 - a broken model is not this job
            url = None
        if url:
            entries.append(("Model", name, url))

    # No selection means "every attack" in benchmark mode only. no_attacks
    # runs none, and detection_reliability without attacks measures the
    # baseline alone, so their footprint must not include attack services.
    specs = config.selected_attack_specs()
    if specs is None:
        specs = list(benchmark.attacks) if config.mode == "benchmark" else []
    for spec in specs:
        name = spec.split(":")[0]
        entry = benchmark.attacks.get(name)
        if not entry:
            continue
        try:
            instance = entry["class"]()
        except Exception:  # noqa: BLE001
            continue
        # Dockerized attacks address a service; native ones have no url.
        url = getattr(instance, "endpoint", None) or getattr(
            instance, "base_url", None)
        if url and not any(e[1] == name for e in entries):
            entries.append(("Attack", name, url))

    # Enabled *anywhere*, not just in metrics.defaults: a config that
    # switches NISQA on per attack group still calls the service, and the
    # defaults-only check missed exactly that shape.
    if any(m in config.resolver.all_signal_metrics() for m in NISQA_METRICS):
        port = os.environ.get("NISQA_PORT", "10030")
        entries.append(("Metric", "NISQA", f"http://localhost:{port}"))

    rows = efficiency.container_snapshot(entries)
    for kind, name, container, used, limit in rows:
        efficiency.log("%s %s holds %.0f MiB in %s", kind.lower(), name,
                       used, container)
    return rows


def _duration_partitions(config, filepaths):
    """Duration bins for this config, or None when grouping is off."""
    if not config.has_duration_groups:
        return None
    from deepmarkpy.utils.utils import partition_files_by_duration

    partitions = partition_files_by_duration(
        filepaths, config.duration_boundaries,
    )
    # "Overall" is the combined view of the bins, so it only says anything
    # when there is more than one. With every file in the same bin it
    # repeats that bin verbatim, under a second heading -- the report then
    # carries two identical sections and invites the reader to look for a
    # difference between them.
    if config.duration_include_overall and len(partitions) > 1:
        partitions.append(("Overall", list(filepaths)))
    return partitions


# ---------------------------------------------------------------------------
# Mode: no_attacks
# ---------------------------------------------------------------------------

def run_no_attacks_mode(benchmark, filepaths, config, settings):
    """Run embed+detect without attacks for one or more models."""
    report_dir = settings.report_dir
    os.makedirs(report_dir, exist_ok=True)

    from deepmarkpy.utils.no_attacks_report_generator import generate_no_attacks_report

    model_names = config.models
    # Audio goes in a dedicated subfolder; use a per-model subfolder when
    # several models run so their watermarked files don't collide.
    multi_model = len(model_names) > 1
    dur_partitions = _duration_partitions(config, filepaths)

    all_results = {}
    for model_name in model_names:
        logger.info(f"Running no-attacks baseline for: {model_name}")
        audio_dir = None
        if settings.save_audio:
            audio_dir = (
                os.path.join(report_dir, model_name, "audio", config.mode)
                if multi_model
                else os.path.join(report_dir, "audio", config.mode)
            )
        try:
            results = benchmark.run_no_attacks(
                filepaths=filepaths,
                wm_model=model_name,
                sampling_rate=None,
                verbose=settings.verbose,
                calculate_quality_metrics=config.calculate_quality_metrics,
                save_audio=settings.save_audio,
                output_dir=audio_dir,
                metric_resolver=config.resolver,
            )
            all_results[model_name] = results
        except (MemoryError, ConnectionError, OSError) as e:
            logger.error(f"Model {model_name} failed: {type(e).__name__}: {e}. Skipping.")
            continue

        # Save each model's results to a separate JSON file
        model_results_path = os.path.join(report_dir, f"no_attacks_{model_name}.json")
        with open(model_results_path, "w") as fp:
            json.dump(to_json_safe(results), fp, indent=4)
        logger.info(f"Results for {model_name} saved to {model_results_path}")

        # Regenerate report after each successful model so a partial
        # report is available even if later models fail.
        try:
            latex_path = generate_no_attacks_report(
                all_results, report_dir=report_dir,
                duration_partitions=dur_partitions,
                resolver=config.resolver,
                containers=_container_rows(benchmark, config, model_names),
            )
            logger.info(f"No-attacks report updated: {latex_path}")
        except Exception as e:
            logger.error(f"Failed to generate no-attacks report: {e}")

    if not all_results:
        logger.error("No models completed successfully.")
        return EXIT_RUNTIME_ERROR
    return EXIT_OK


# ---------------------------------------------------------------------------
# Mode: detection_reliability
# ---------------------------------------------------------------------------

def run_detection_reliability_mode(benchmark, filepaths, config, settings):
    """Run detection-reliability for a single zero-bit or confidence-based model.

    Computes FP / FN both without attacks and (when attacks are
    requested) with each attack applied, then writes a dedicated PDF
    report.
    """
    from deepmarkpy.utils.detection_reliability import run_detection_reliability
    from deepmarkpy.utils.detection_reliability_report_generator import (
        generate_detection_reliability_report,
    )

    report_dir = settings.report_dir
    os.makedirs(report_dir, exist_ok=True)

    model_name = config.models[0]
    # An empty selection means "no attacks" here: the no-attack baseline is
    # always measured and attacks are added to it, so there is nothing to
    # expand to "all".
    attack_types = config.selected_attack_specs() or []

    audio_dir = (os.path.join(report_dir, "audio", config.mode)
                 if settings.save_audio else None)

    logger.info(
        f"Running detection-reliability for {model_name} on "
        f"{len(filepaths)} files; attacks={attack_types or 'none'}"
    )
    try:
        result = run_detection_reliability(
            benchmark,
            filepaths,
            wm_model=model_name,
            attack_types=attack_types,
            sampling_rate=None,
            verbose=settings.verbose,
            calculate_quality_metrics=config.calculate_quality_metrics,
            save_audio=settings.save_audio,
            output_dir=audio_dir,
            metric_resolver=config.resolver,
            attack_parameters=config.parameters_for,
            extra_attack_versions=config.synthetic_versions,
        )
    except ValueError as e:
        logger.error(f"Detection reliability run failed: {e}")
        return EXIT_RUNTIME_ERROR

    # Persist raw result so later runs can inspect/regenerate the PDF.
    result_path = os.path.join(report_dir, "detection_reliability.json")
    with open(result_path, "w") as fp:
        json.dump(to_json_safe(dict(result)), fp, indent=4)
    logger.info(f"Detection reliability data saved to {result_path}")

    try:
        tex_path = generate_detection_reliability_report(
            dict(result), report_dir=report_dir,
            resolver=config.resolver,
            duration_partitions=_duration_partitions(config, filepaths),
            containers=_container_rows(benchmark, config, config.models),
        )
        logger.info(f"Detection reliability report generated: {tex_path}")
    except Exception as e:
        logger.error(f"Failed to generate detection reliability report: {e}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Mode: benchmark
# ---------------------------------------------------------------------------

def run_benchmark_mode(benchmark, filepaths, config, settings):
    """Run the attack benchmark for one model, or several with a comparison."""
    if len(config.models) > 1:
        return run_multiple_models(
            benchmark, filepaths, config.models, config, settings)

    run_single_model(
        benchmark, filepaths, config.models[0], config, settings,
        output_dir=settings.report_dir,
    )
    return EXIT_OK


def write_run_metadata(report_dir, args, benchmark, model_names, extra=None):
    """Write run_metadata.json beside the results so a run can be situated later.

    Deliberately a sibling file rather than a wrapper around the existing
    artifacts: the report generators read benchmark_stats.json's top-level
    keys as attack names, and embedding a timestamp inside a result file
    would make byte-comparing two runs impossible.
    """
    metadata = {
        "deepmarkpy_version": __version__,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "command": sys.argv,
        "seed": getattr(args, "seed", None),
        "models": list(model_names),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "plugins": {
            "attacks_discovered": sorted(benchmark.attacks),
            "models_discovered": sorted(benchmark.models),
            "failed_imports": benchmark.plugin_manager.failed,
        },
        # A metric that could not run leaves N/A cells that look identical to
        # a metric that ran and had nothing to say.
        "metrics": {"nisqa": nisqa_status()},
    }
    if extra:
        metadata.update(extra)

    path = os.path.join(report_dir, "run_metadata.json")
    os.makedirs(report_dir, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(to_json_safe(metadata), fp, indent=4)
    logger.info(f"Run metadata saved to {path}")
    return path


def _git_revision():
    """Short git revision when running from a checkout, else None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def run_single_model(benchmark, filepaths, model_name, config, settings,
                     output_dir=None):
    """Run benchmark for a single model, save results and generate reports.

    Args:
        benchmark: Benchmark instance
        filepaths: List of audio file paths
        model_name: Name of the watermarking model
        config: the validated ``ModeConfig`` driving this run
        settings: resolved operational settings
        output_dir: Optional directory for outputs (default: the report dir)

    Returns:
        Tuple of (results, flattened_stats, stats): raw per-file results,
        the attack->primary-statistic mapping the comparative report ranks,
        and the full per-attack stats.
    """
    report_dir = output_dir or settings.report_dir
    os.makedirs(report_dir, exist_ok=True)

    run_kwargs = dict(
        wm_model=model_name,
        attack_types=config.selected_attack_specs(),
        sampling_rate=None,
        verbose=settings.verbose,
        save_audio=settings.save_audio,
        calculate_quality_metrics=config.calculate_quality_metrics,
        crop_before_attack=config.crop_before_attack,
        metric_resolver=config.resolver,
        # Routed per expanded attack entry rather than as one flat mapping,
        # so an override on one version cannot reach another.
        attack_parameters=config.parameters_for,
        extra_attack_versions=config.synthetic_versions,
    )
    # Keep audio files in a dedicated subfolder so they don't clutter
    # the report directory next to .tex/.pdf/.json outputs.
    if settings.save_audio:
        run_kwargs["output_dir"] = os.path.join(report_dir, "audio", config.mode)

    results_path = os.path.join(report_dir, "benchmark_results.json")

    # Rewrite the results file after every file rather than once at the end.
    # A long run used to keep everything in memory until the last file
    # finished, so interrupting it discarded all the work done so far.
    completed = {}

    def _persist(filepath, file_results):
        completed[filepath] = file_results
        with open(results_path, "w") as fp:
            json.dump(to_json_safe(completed), fp, indent=4)
        logger.info(
            f"Saved results for {len(completed)}/{len(filepaths)} files "
            f"to {results_path}"
        )

    results = benchmark.run(
        filepaths=filepaths, on_file_complete=_persist, **run_kwargs
    )

    with open(results_path, "w") as fp:
        json.dump(to_json_safe(results), fp, indent=4)
    logger.info(f"Results saved to {results_path}")

    primary = config.comparison_primary_statistic
    model_config = benchmark.models.get(model_name, {}).get("config") or {}
    is_zero_bit = bool(model_config.get("is_zero_bit", False))
    stats_path = os.path.join(report_dir, "benchmark_stats.json")
    partitions = _duration_partitions(config, filepaths)

    if partitions:
        all_group_stats = {}
        for group_label, group_files in partitions:
            group_results = {fp: results[fp] for fp in group_files if fp in results}
            if not group_results:
                continue
            all_group_stats[group_label] = {
                "stats": benchmark.compute_mean_accuracy(
                    group_results, resolver=config.resolver, is_zero_bit=is_zero_bit,
                ),
                "n_files": len(group_results),
            }

        with open(stats_path, "w") as fp:
            json.dump(to_json_safe(all_group_stats), fp, indent=4)
        logger.info(f"Duration-grouped statistics saved to {stats_path}")

        # The comparative report needs one flat table over every file.
        # "Overall" is exactly that when the config asked for it; otherwise
        # it is computed here. Falling back to the first bin, as this once
        # did, silently dropped every file in the later bins.
        if "Overall" in all_group_stats:
            stats = all_group_stats["Overall"]["stats"]
        else:
            stats = benchmark.compute_mean_accuracy(
                results, resolver=config.resolver, is_zero_bit=is_zero_bit,
            )
    else:
        stats = benchmark.compute_mean_accuracy(
            results, resolver=config.resolver, is_zero_bit=is_zero_bit,
        )
        with open(stats_path, "w") as fp:
            json.dump(to_json_safe(stats), fp, indent=4)
        logger.info(f"Statistics saved to {stats_path}")

    flattened_stats = {
        attack: metrics.get(f"accuracy_{primary}")
        for attack, metrics in stats.items()
    }

    write_run_metadata(
        report_dir, settings, benchmark, [model_name],
        extra={
            # The rate actually used, resolved from the model's config —
            # previously this was only logged to stderr and then lost.
            "sampling_rate": (benchmark.models.get(model_name, {}).get("config") or {}).get(
                "sampling_rate"
            ),
            "watermark_size": (benchmark.models.get(model_name, {}).get("config") or {}).get(
                "watermark_size"
            ),
            "attacks_run": sorted(stats),
            "n_files": len(filepaths),
            "config_file": config.source,
            # Which version ran with which values, so a table can be read
            # back months later without re-deriving it from the config.
            "attack_parameters_resolved": resolved_attack_parameters(
                benchmark, config,
            ),
        },
    )

    containers = _container_rows(benchmark, config, [model_name])

    _log_efficiency(config, stats)

    try:
        latex_path, chart_path = generate_benchmark_report(
            stats_file=stats_path,
            model_name=model_name,
            report_dir=report_dir,
            resolver=config.resolver,
            crop_before_attack=config.crop_before_attack,
            is_zero_bit=model_config.get("is_zero_bit", False),
            containers=containers,
        )
        logger.info(f"Benchmark report generated: {latex_path}")
    except Exception as e:
        logger.error(f"Failed to generate benchmark report: {e}")

    if config.resolver.any_signal_metric_enabled():
        try:
            from deepmarkpy.utils.detailed_report_generator import DetailedReportGenerator
            detailed_generator = DetailedReportGenerator(
                report_dir=report_dir, resolver=config.resolver,
            )
            latex_path = detailed_generator.generate_full_report(
                results, model_name=model_name,
                is_zero_bit=model_config.get("is_zero_bit", False),
                crop_before_attack=config.crop_before_attack,
                duration_partitions=partitions,
                containers=containers,
            )
            logger.info(f"Detailed report saved to: {latex_path}")
        except Exception as e:
            logger.error(f"Failed to generate detailed report: {e}")

    return results, flattened_stats, stats


def run_multiple_models(benchmark, filepaths, model_names, config, settings):
    """Run benchmark for multiple models and generate comparative report."""
    report_base = settings.report_dir

    all_results = {}
    all_stats = {}
    all_meta = {}

    failed_models = []
    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running benchmark for model: {model_name}")
        logger.info(f"{'='*60}")

        model_dir = os.path.join(report_base, model_name)
        _copy_deepmark_assets(report_base, model_dir)
        try:
            results, flattened_stats, model_stats = run_single_model(
                benchmark, filepaths, model_name, config, settings,
                output_dir=model_dir,
            )
        except (MemoryError, ConnectionError, OSError) as e:
            # Only infrastructure failures (OOM, Docker service crash,
            # network issue) are tolerated so that one model does not
            # block the whole run. Code-level exceptions (SyntaxError,
            # ImportError, NameError, AttributeError, ...) fall through
            # and abort the benchmark — they signal bugs that need to
            # be fixed rather than silently skipped.
            logger.error(
                f"Model {model_name} failed: {type(e).__name__}: {e}. "
                f"Skipping to next model."
            )
            failed_models.append(model_name)
            continue

        all_results[model_name] = results
        all_stats[model_name] = model_stats
        # Metadata the comparative report needs to keep the columns honest:
        # a zero-bit model's score is a detection rate (floor 0), a multi-bit
        # model's is bit agreement (floor ~50), and payload width differs.
        model_config = benchmark.models[model_name].get("config") or {}
        all_meta[model_name] = {
            "is_zero_bit": bool(model_config.get("is_zero_bit", False)),
            "watermark_size": model_config.get("watermark_size"),
            "sampling_rate": model_config.get("sampling_rate"),
            "n_files": max(
                (m.get("accuracy_n", 0) for m in model_stats.values()), default=0
            ),
        }

    if failed_models:
        logger.warning(
            f"Skipped {len(failed_models)} model(s) due to errors: "
            f"{', '.join(failed_models)}"
        )
    if not all_results:
        logger.error("No models completed successfully. Skipping comparative report.")
        return EXIT_RUNTIME_ERROR
    if len(all_results) < 2:
        # Comparative report needs at least two models to compare.
        only = next(iter(all_results))
        logger.info(
            f"Only {only} completed successfully; skipping comparative "
            f"report (single-model outputs are already in "
            f"{os.path.join(report_base, only)}/)."
        )
        return EXIT_OK

    # Generate comparative report
    try:
        from deepmarkpy.utils.comparative_report_generator import ComparativeReportGenerator
        logger.info("Generating comparative report...")
        comp_dir = os.path.join(report_base, "comparison")
        _copy_deepmark_assets(report_base, comp_dir)
        comp_generator = ComparativeReportGenerator(
            report_dir=comp_dir, resolver=config.resolver,
            primary_statistic=config.comparison_primary_statistic,
        )
        comp_generator.generate_full_report(
            all_stats,
            crop_before_attack=config.crop_before_attack,
            model_meta=all_meta,
        )
        logger.info(f"Comparative report saved to: {comp_dir}")
    except Exception as e:
        logger.error(f"Failed to generate comparative report: {e}")
    return EXIT_OK


_MODE_RUNNERS = {
    "benchmark": run_benchmark_mode,
    "no_attacks": run_no_attacks_mode,
    "detection_reliability": run_detection_reliability_mode,
}


if __name__ == "__main__":
    sys.exit(main())
