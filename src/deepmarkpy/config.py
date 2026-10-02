"""Benchmark configuration: schema, validation, and loading.

One file per mode. Each file is self-contained, declares its own
``"mode"``, and is only allowed to carry keys that mode actually uses --
so a ``no_attacks`` config cannot mention attacks, and the reader is
never asked to work out which sections to ignore. Pass several with
``--config a.json b.json`` to run several modes in one invocation.

The config file owns every measurement decision (mode, models, attacks,
attack parameters, metrics, statistics, duration groups, crop). The CLI
owns operational ones (where the audio is, where reports go, seed,
verbosity, audio dumping, plugin directory) and wins for the handful
mirrored under ``general``.

Validation collects **every** problem before the run starts and reports
them together, each with a stable code, the exact JSON path, the
offending value, and a suggestion when the value looks like a typo.
Errors block the run; warnings and info are printed and the run
proceeds.
"""

import difflib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from deepmarkpy.utils.attack_groups import (
    ATTACK_GROUPS,
    CONFIG_GROUP_KEYS,
    get_attacks_for_groups,
)
from deepmarkpy.utils.metric_resolver import (
    ALL_STATISTICS,
    CANONICAL_METRIC_ORDER,
    EFFICIENCY_METRICS,
    MANDATORY_METRICS,
    MetricResolver,
    NISQA_METRICS,
    STATISTICS_EXEMPT_METRICS,
)

logger = logging.getLogger(__name__)

VALID_MODES = ("benchmark", "no_attacks", "detection_reliability")

# Keys every mode's file accepts.
_COMMON_KEYS = frozenset({
    "mode",
    "general",
    "models",
    "calculate_quality_metrics",
    "statistics",
    "metrics",
    # Its own section in every mode: whether to time the run is a separate
    # decision from which quality metrics to compute.
    "efficiency",
    "duration_groups",
})

# Keys accepted per mode. Anything else is rejected by name, with a note
# saying which mode does accept it when one does -- that turns "unknown
# key" into "wrong file" and is the single most common mistake this split
# is meant to prevent.
MODE_KEYS = {
    "benchmark": _COMMON_KEYS | {
        "attacks", "attack_parameters", "comparison", "crop_before_attack",
    },
    "no_attacks": _COMMON_KEYS,
    "detection_reliability": _COMMON_KEYS | {"attacks", "attack_parameters"},
}

# ``metrics.per_group`` only means something where attacks run.
MODES_WITHOUT_GROUPS = frozenset({"no_attacks"})

# Bit error rate is the complement of bit agreement. Detection reliability
# scores a binary per-file outcome, so there are no bits to disagree.
MODE_FORBIDDEN_METRICS = {
    "detection_reliability": frozenset({"ber"}),
}

_GENERAL_KEYS = frozenset({
    "wav_files_dir", "report_dir", "seed", "verbose", "save_audio",
    "plugins_dir",
})

_METRIC_ENTRY_KEYS = frozenset({"enabled", "statistics"})

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "config_templates")

_UNSET = object()


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigIssue:
    """One validation finding, addressed to the person editing the file."""

    code: str
    path: str
    message: str
    value: Any = _UNSET
    suggestion: Optional[str] = None
    severity: str = "error"
    source: Optional[str] = None

    def render(self) -> str:
        location = f"{self.source}: " if self.source else ""
        line = f"[{self.code}] {location}{self.path}\n    {self.message}"
        if self.value is not _UNSET:
            line += f"\n    got: {json.dumps(self.value, default=str)}"
        if self.suggestion:
            line += f"\n    did you mean '{self.suggestion}'?"
        return line


class ConfigError(Exception):
    """Raised when one or more config files are invalid.

    Carries every blocking issue found, so the user fixes them in one
    pass instead of rerunning after each.
    """

    def __init__(self, issues: Sequence[ConfigIssue]):
        self.issues = list(issues)
        count = len(self.issues)
        noun = "problem" if count == 1 else "problems"
        body = "\n\n".join(issue.render() for issue in self.issues)
        super().__init__(
            f"Found {count} configuration {noun}:\n\n{body}\n\n"
            f"Run 'deepmark-benchmark --init <mode>' to print a fresh, "
            f"fully-commented config file for a mode."
        )


def _suggest(value: Any, candidates: Sequence[str]) -> Optional[str]:
    """Closest valid spelling of ``value``, or None when nothing is close.

    Tries case, then prefixes, before edit distance. An abbreviation is a
    common way to get a long key wrong -- "desync" for
    "desynchronization" -- and it scores far below difflib's threshold
    because the strings differ mostly in length.
    """
    if not isinstance(value, str) or not value:
        return None
    candidates = list(candidates)
    lowered = value.lower()

    for candidate in candidates:
        if candidate.lower() == lowered:
            return candidate

    prefixed = [
        candidate for candidate in candidates
        if candidate.lower().startswith(lowered)
        or lowered.startswith(candidate.lower())
    ]
    if prefixed:
        return min(prefixed, key=len)

    matches = difflib.get_close_matches(value, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


def _json_syntax_help(text: str, exc: json.JSONDecodeError) -> str:
    """Show the offending line and name the likely cause.

    Python's own message ("Expecting property name enclosed in double
    quotes") describes the parser's state, not the mistake. A comma left
    before a closing brace is by far the most common way to break one of
    these files, and it is worth saying so in those words.
    """
    lines = text.splitlines()
    if not text.strip():
        return "\n    The file is empty."

    parts = []
    if 1 <= exc.lineno <= len(lines):
        offending = lines[exc.lineno - 1]
        if exc.lineno >= 2:
            parts.append(f"    {exc.lineno - 1:>4} | {lines[exc.lineno - 2]}")
        parts.append(f"    {exc.lineno:>4} | {offending}")
        parts.append("    " + " " * 4 + " | " + " " * max(exc.colno - 1, 0) + "^")
    else:
        offending = ""

    # What precedes the failure point is what usually identifies the cause.
    before = text[:max(exc.pos, 0)].rstrip()
    prev = before[-1:] if before else ""
    hint = None
    if exc.msg.startswith("Expecting property name") and prev == ",":
        hint = ("A comma before the closing brace. JSON allows no trailing "
                "comma - remove it.")
    elif exc.msg.startswith("Expecting value") and prev == ",":
        hint = ("A comma before the closing bracket. JSON allows no trailing "
                "comma - remove it.")
    elif exc.msg.startswith("Expecting ',' delimiter"):
        hint = ("A comma is missing between two entries, or a { or [ above "
                "was never closed.")
    elif exc.msg.startswith("Extra data"):
        hint = ("Something follows the end of the object - usually one "
                "closing brace too many.")
    elif "'" in offending and exc.msg.startswith("Expecting property name"):
        hint = "JSON needs double quotes; single quotes are not valid."
    elif offending.lstrip().startswith(("//", "#", "/*")):
        hint = ("JSON has no comments. This project uses keys prefixed with "
                "\"_\" instead, which the parser ignores.")
    elif exc.msg.startswith("Expecting property name"):
        hint = "A key here must be a name in double quotes."
    if hint:
        parts.append(f"    {hint}")

    # Unbalanced brackets are invisible at the failure point, which is
    # usually far from the line that actually forgot to close.
    for opener, closer, label in (("{", "}", "Braces"), ("[", "]", "Brackets")):
        n_open, n_close = text.count(opener), text.count(closer)
        if n_open != n_close:
            parts.append(
                f"    {label} do not balance across the file: "
                f"{n_open} '{opener}' and {n_close} '{closer}'."
            )
    return "\n" + "\n".join(parts) if parts else ""


def _type_name(value: Any) -> str:
    return {
        dict: "an object", list: "an array", str: "a string",
        bool: "a boolean", int: "a number", float: "a number",
        type(None): "null",
    }.get(type(value), type(value).__name__)


def _real_keys(mapping: Dict[str, Any]) -> List[str]:
    """Keys excluding documentation keys.

    JSON has no comments, so the shipped templates document themselves
    with ``_``-prefixed keys. Those are ignored everywhere.
    """
    return [key for key in mapping if not key.startswith("_")]


# ---------------------------------------------------------------------------
# Parsed configuration
# ---------------------------------------------------------------------------

@dataclass
class ModeConfig:
    """A validated config file for one benchmark mode."""

    mode: str
    source: str
    general: Dict[str, Any] = field(default_factory=dict)
    models: List[str] = field(default_factory=list)
    attack_groups: List[str] = field(default_factory=list)
    attack_list: List[str] = field(default_factory=list)
    # {(attack, version): params} -- version is the resolved preset name,
    # or None for an attack that declares no presets.
    parameter_overrides: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    # {attack: {version: params}} -- versions this config defines, which the
    # plugin's own config.json does not declare.
    synthetic_versions: Dict[str, Dict[str, Dict[str, Any]]] = field(
        default_factory=dict)
    calculate_quality_metrics: bool = False
    statistics: Optional[List[str]] = None
    resolver: MetricResolver = field(default_factory=MetricResolver)
    comparison_primary_statistic: str = "mean"
    crop_before_attack: Optional[float] = None
    duration_boundaries: List[float] = field(default_factory=list)
    duration_include_overall: bool = False
    warnings: List[ConfigIssue] = field(default_factory=list)

    # -- attacks -------------------------------------------------------

    def selected_attack_specs(self) -> Optional[List[str]]:
        """Attack specs this config asks for, or None meaning "every attack".

        Groups are expanded to the attacks they *declare*, not to the ones
        that happen to have imported. A group whose plugin failed to load
        must fail loudly rather than measure a smaller set, which is what
        ``Benchmark.run`` raises on.
        """
        if not self.attack_groups and not self.attack_list:
            return None
        specs = list(self.attack_list)
        specs += [
            attack for attack in get_attacks_for_groups(self.attack_groups)
            if attack not in specs
        ]
        return specs

    def parameters_for(self, attack_name, version) -> Dict[str, Any]:
        """Parameters to apply to one expanded attack entry.

        A version this config defines carries its full parameter set; an
        override on an existing version carries only what it changes.
        Looked up by resolved version, so ``GaussianNoiseAttack`` and
        ``GaussianNoiseAttack:default`` are the same target and cannot
        leak onto ``:aggressive``.
        """
        defined = self.synthetic_versions.get(attack_name, {})
        if version in defined:
            return dict(defined[version])

        found = self.parameter_overrides.get((attack_name, version))
        if found is None and version in (None, "default"):
            # The two spellings are one target, but which key each side
            # uses depends on facts they do not share: the validator keys
            # a bare name by how many versions the plugin declares, the
            # run loop asks by how ``attacks.list`` spelled the attack.
            # Without this the override is dropped without a word.
            alias = "default" if version is None else None
            found = self.parameter_overrides.get((attack_name, alias))
        return dict(found or {})

    # -- duration groups -----------------------------------------------

    @property
    def has_duration_groups(self) -> bool:
        return bool(self.duration_boundaries)

    def duration_labels(self) -> List[str]:
        """Human-readable bin labels, e.g. ``["< 5s", "5-10s", "> 30s"]``."""
        bounds = self.duration_boundaries
        if not bounds:
            return []
        labels = [f"< {bounds[0]}s"]
        labels += [f"{bounds[i]}–{bounds[i + 1]}s"
                   for i in range(len(bounds) - 1)]
        labels.append(f"> {bounds[-1]}s")
        return labels


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_configs(
    paths: Sequence[str],
    attacks_registry: Optional[Dict[str, Any]] = None,
    models_registry: Optional[Dict[str, Any]] = None,
    quiet: bool = False,
) -> List[ModeConfig]:
    """Load, validate, and return one ``ModeConfig`` per path.

    Args:
        paths: config file paths, in the order their modes should run.
        attacks_registry: ``Benchmark.attacks``. When given, attack names,
            versions and parameter keys are checked against the discovered
            plugins. When None those checks are skipped, so the validator
            is usable without loading plugins -- which is how the CLI
            catches a broken file before the noisy plugin import runs.
        models_registry: ``Benchmark.models``, used the same way.
        quiet: skip logging the warnings. Set on the pre-flight pass so
            they are not printed twice.

    Raises:
        ConfigError: with every blocking issue found across every file.
    """
    return _validate_all(
        [(path, _UNSET) for path in paths],
        attacks_registry, models_registry, quiet,
    )


def load_config_data(
    data: Dict[str, Any],
    source: str = "<memory>",
    attacks_registry: Optional[Dict[str, Any]] = None,
    models_registry: Optional[Dict[str, Any]] = None,
    quiet: bool = False,
) -> ModeConfig:
    """Validate an already-built config mapping, without touching disk.

    The same checks, codes and messages ``load_configs`` applies to a
    file. For an application that assembles the configuration itself --
    from a form, a database, its own settings file -- this is how it
    finds out whether what it built is runnable, and what to tell its
    user when it is not, before writing anything.

    Args:
        data: the config mapping, shaped exactly like the JSON file.
        source: what to name this configuration in error messages.
        attacks_registry: ``Benchmark.attacks``; see ``load_configs``.
        models_registry: ``Benchmark.models``; see ``load_configs``.
        quiet: skip logging the warnings.

    Returns:
        The validated ``ModeConfig``, ready to hand to a mode runner or to
        serialise to a file with ``json.dump``.

    Raises:
        ConfigError: with every blocking issue found.
    """
    return _validate_all(
        [(source, data)], attacks_registry, models_registry, quiet,
    )[0]


def _validate_all(sources, attacks_registry, models_registry, quiet):
    """Validate several configurations, reporting all of their issues at once."""
    issues: List[ConfigIssue] = []
    configs: List[ModeConfig] = []

    for source, data in sources:
        validator = _Validator(
            source, attacks_registry, models_registry, data=data,
        )
        config = validator.run()
        issues.extend(validator.errors)
        if config is not None:
            configs.append(config)

    issues.extend(_check_mode_uniqueness(configs))

    if issues:
        raise ConfigError(issues)

    if not quiet:
        for config in configs:
            for warning in config.warnings:
                logger.warning(warning.render()) if warning.severity == "warning" \
                    else logger.info(warning.render())

    return configs


def _check_mode_uniqueness(configs: Sequence[ModeConfig]) -> List[ConfigIssue]:
    """Reject two config files declaring the same mode in one invocation.

    Each mode writes fixed output filenames, so the second run would
    overwrite the first's report with no warning.
    """
    seen: Dict[str, str] = {}
    issues = []
    for config in configs:
        if config.mode in seen:
            issues.append(ConfigIssue(
                code="E006",
                path="mode",
                source=config.source,
                value=config.mode,
                message=(
                    f"mode '{config.mode}' is already declared by "
                    f"'{seen[config.mode]}'. Each --config file passed to one "
                    f"invocation must declare a different mode, because both "
                    f"would write the same report filenames."
                ),
            ))
        else:
            seen[config.mode] = config.source
    return issues


def init_template(mode: str) -> str:
    """Return the fully-commented template config file for ``mode``."""
    if mode not in VALID_MODES:
        suggestion = _suggest(mode, VALID_MODES)
        hint = f" Did you mean '{suggestion}'?" if suggestion else ""
        raise ConfigError([ConfigIssue(
            code="E005",
            path="--init",
            value=mode,
            message=(
                f"unknown mode. Valid modes: {', '.join(VALID_MODES)}.{hint}"
            ),
            suggestion=suggestion,
        )])

    with open(os.path.join(_TEMPLATE_DIR, f"{mode}.json"), encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class _Validator:
    """Validates one config file, accumulating every issue it finds.

    Each ``_check_*`` method guards its own preconditions and returns a
    usable value (or a safe default) so later checks still run. That is
    what lets one pass report every problem at once.
    """

    def __init__(self, path, attacks_registry=None, models_registry=None,
                 data=_UNSET):
        """
        Args:
            path: the config file to read, or the name to report issues
                under when ``data`` is given.
            data: an already-parsed config mapping. Given, the file is
                never opened -- an embedding application can validate the
                configuration it built from its own UI before writing it
                anywhere, and show the same error codes the CLI prints.
        """
        self.path = path
        self.attacks_registry = attacks_registry
        self.models_registry = models_registry
        self.errors: List[ConfigIssue] = []
        self.warnings: List[ConfigIssue] = []
        self.raw: Dict[str, Any] = {}
        self.data = data
        self.mode: Optional[str] = None

    # -- issue helpers -------------------------------------------------

    def error(self, code, path, message, value=_UNSET, suggestion=None):
        self.errors.append(ConfigIssue(
            code=code, path=path, message=message, value=value,
            suggestion=suggestion, severity="error", source=self.path,
        ))

    def warn(self, code, path, message, value=_UNSET, severity="warning"):
        self.warnings.append(ConfigIssue(
            code=code, path=path, message=message, value=value,
            severity=severity, source=self.path,
        ))

    def _typed(self, container, key, path, expected, default):
        """Return ``container[key]`` when it has the expected type.

        Records a type error and returns ``default`` otherwise, so the
        caller can keep validating instead of aborting the whole file.
        """
        if key not in container:
            return default
        value = container[key]
        if not isinstance(value, expected) or (
            expected is not bool and isinstance(value, bool)
        ):
            self.error(
                "E009", path,
                f"must be {_TYPE_LABELS[expected]}, but is {_type_name(value)}.",
                value=value,
            )
            return default
        return value

    # -- entry point ---------------------------------------------------

    def run(self) -> Optional[ModeConfig]:
        if not self._read():
            return None
        if not self._check_not_old_format():
            return None

        self.mode = self._check_mode()
        if self.mode is None:
            return None

        self._check_top_level_keys()

        general = self._check_general()
        models = self._check_models()
        # Before _check_attacks, which needs to know the versions this file
        # defines in order to accept them in attacks.list.
        parameter_overrides, synthetic_versions = self._check_attack_parameters()
        groups, attack_list = self._check_attacks(synthetic_versions)
        calculate = self._typed(
            self.raw, "calculate_quality_metrics",
            "calculate_quality_metrics", bool, False,
        )
        statistics = self._check_statistics()
        defaults, per_group = self._check_metrics(statistics, calculate)
        efficiency = self._check_efficiency(statistics)
        crop = self._check_crop()
        boundaries, include_overall = self._check_duration_groups()

        resolver = MetricResolver(
            defaults=defaults,
            per_group=per_group,
            statistics=statistics,
            calculate_quality_metrics=calculate,
            efficiency=efficiency,
        )

        comparison = self._check_comparison(resolver)
        self._check_group_coverage(per_group, groups, attack_list)
        self._check_parameter_coverage(
            parameter_overrides, synthetic_versions, groups, attack_list,
        )
        self._check_nisqa_cost(resolver)

        if self.errors:
            return None

        return ModeConfig(
            mode=self.mode,
            source=self.path,
            general=general,
            models=models,
            attack_groups=groups,
            attack_list=attack_list,
            parameter_overrides=parameter_overrides,
            synthetic_versions=synthetic_versions,
            calculate_quality_metrics=calculate,
            statistics=statistics,
            resolver=resolver,
            comparison_primary_statistic=comparison,
            crop_before_attack=crop,
            duration_boundaries=boundaries,
            duration_include_overall=include_overall,
            warnings=self.warnings,
        )

    # -- file ----------------------------------------------------------

    def _read(self) -> bool:
        if self.data is not _UNSET:
            return self._accept(self.data)

        if not os.path.exists(self.path):
            self.error(
                "E001", "--config",
                "config file does not exist. Create one with "
                "'deepmark-benchmark --init <mode> > <path>'.",
                value=self.path,
            )
            return False
        try:
            with open(self.path, encoding="utf-8") as fh:
                text = fh.read()
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            self.error(
                "E002", f"line {exc.lineno}, column {exc.colno}",
                f"file is not valid JSON: {exc.msg}."
                + _json_syntax_help(text, exc),
            )
            return False
        except OSError as exc:
            self.error("E001", "--config", f"cannot read config file: {exc}",
                       value=self.path)
            return False

        return self._accept(raw)

    def _accept(self, raw) -> bool:
        """Take a parsed config mapping, whatever it was parsed from."""
        if not isinstance(raw, dict):
            self.error("E003", "(root)",
                       f"the top level must be a JSON object, but is "
                       f"{_type_name(raw)}.")
            return False

        self.raw = raw
        return True

    def _check_not_old_format(self) -> bool:
        """Reject the pre-2.1 single-file format explicitly.

        Its ``modes`` block and positional ``"mean:T std:F"`` statistic
        strings would otherwise be read as unknown keys and wrong types,
        producing a pile of errors that never says what actually happened.
        """
        old_signals = []
        if "modes" in self.raw:
            old_signals.append("a 'modes' block")
        if isinstance(self.raw.get("attacks"), dict) and \
                "source" in self.raw["attacks"]:
            old_signals.append("'attacks.source'")
        metrics = self.raw.get("metrics")
        if isinstance(metrics, dict):
            positional = [
                key for key, entry in metrics.items()
                if isinstance(entry, dict) and isinstance(
                    entry.get("statistics"), str)
            ]
            if positional:
                old_signals.append(
                    f"positional statistic strings "
                    f"(metrics.{positional[0]}.statistics)"
                )

        if not old_signals:
            return True

        self.error(
            "E038", "(root)",
            f"this is a pre-2.1 config file ({', '.join(old_signals)}). The "
            f"format changed: one file per mode, each declaring \"mode\", with "
            f"metrics under 'metrics.defaults'/'metrics.per_group' and "
            f"statistics written as JSON arrays. Run "
            f"'deepmark-benchmark --init benchmark' for a fresh file.",
        )
        return False

    # -- mode ----------------------------------------------------------

    def _check_mode(self) -> Optional[str]:
        if "mode" not in self.raw:
            self.error(
                "E004", "mode",
                f"missing required key. Every config file must declare which "
                f"mode it configures, one of: {', '.join(VALID_MODES)}.",
            )
            return None

        mode = self.raw["mode"]
        if mode not in VALID_MODES:
            self.error(
                "E005", "mode",
                f"unknown mode. Valid modes: {', '.join(VALID_MODES)}.",
                value=mode, suggestion=_suggest(mode, VALID_MODES),
            )
            return None
        return mode

    def _check_top_level_keys(self):
        allowed = MODE_KEYS[self.mode]
        for key in _real_keys(self.raw):
            if key in allowed:
                continue
            other_modes = sorted(
                m for m, keys in MODE_KEYS.items() if key in keys
            )
            if other_modes:
                self.error(
                    "E008", key,
                    f"'{key}' is not used by mode '{self.mode}', so setting it "
                    f"here would have no effect. It belongs to: "
                    f"{', '.join(other_modes)}.",
                )
            else:
                self.error(
                    "E007", key,
                    f"unknown key. Keys accepted in mode '{self.mode}': "
                    f"{', '.join(sorted(allowed))}.",
                    suggestion=_suggest(key, sorted(allowed)),
                )

    # -- general -------------------------------------------------------

    def _check_general(self) -> Dict[str, Any]:
        general = self._typed(self.raw, "general", "general", dict, {})
        clean = {}
        for key in _real_keys(general):
            if key not in _GENERAL_KEYS:
                self.error(
                    "E039", f"general.{key}",
                    f"unknown key. Accepted: {', '.join(sorted(_GENERAL_KEYS))}. "
                    f"Measurement settings (models, attacks, metrics, "
                    f"statistics) are top-level keys, not 'general' ones.",
                    suggestion=_suggest(key, sorted(_GENERAL_KEYS)),
                )
                continue
            clean[key] = general[key]

        for key in ("wav_files_dir", "report_dir", "plugins_dir"):
            if clean.get(key) is not None and not isinstance(clean[key], str):
                self.error("E009", f"general.{key}",
                           f"must be a string or null, but is "
                           f"{_type_name(clean[key])}.", value=clean[key])
        for key in ("verbose", "save_audio"):
            if key in clean and not isinstance(clean[key], bool):
                self.error("E009", f"general.{key}",
                           f"must be true or false, but is "
                           f"{_type_name(clean[key])}.", value=clean[key])

        seed = clean.get("seed")
        if seed is not None and (isinstance(seed, bool)
                                 or not isinstance(seed, int)):
            self.error(
                "E037", "general.seed",
                "must be an integer or null. null keeps a fresh watermark per "
                "file and fresh attack noise per run.",
                value=seed,
            )
        return clean

    # -- models --------------------------------------------------------

    def _check_models(self) -> List[str]:
        if "models" not in self.raw:
            self.error(
                "E010", "models",
                "missing required key: list at least one watermarking model "
                "class name, e.g. [\"AudioSealModel\"].",
            )
            return []

        models = self._typed(self.raw, "models", "models", list, [])
        if not models:
            self.error("E010", "models",
                       "must list at least one model class name.", value=models)
            return []

        known = sorted(self.models_registry) if self.models_registry else None
        clean = []
        for index, name in enumerate(models):
            if not isinstance(name, str):
                self.error("E009", f"models[{index}]",
                           f"must be a model class name (a string), but is "
                           f"{_type_name(name)}.", value=name)
                continue
            if known is not None and name not in known:
                self.error(
                    "E011", f"models[{index}]",
                    f"unknown model. Discovered models: {', '.join(known)}.",
                    value=name, suggestion=_suggest(name, known),
                )
                continue
            if name in clean:
                self.error("E016", f"models[{index}]",
                           "listed more than once.", value=name)
                continue
            clean.append(name)

        if self.mode == "detection_reliability" and len(clean) > 1:
            self.error(
                "E012", "models",
                f"mode 'detection_reliability' measures one model at a time "
                f"(false-positive and false-negative rates are per model), but "
                f"{len(clean)} are listed. Use one file per model.",
                value=clean,
            )
        return clean

    # -- attacks -------------------------------------------------------

    def _check_attacks(self, synthetic_versions=None):
        if "attacks" not in MODE_KEYS[self.mode]:
            return [], []
        synthetic_versions = synthetic_versions or {}

        attacks = self._typed(self.raw, "attacks", "attacks", dict, {})
        for key in _real_keys(attacks):
            if key not in ("groups", "list"):
                self.error(
                    "E007", f"attacks.{key}",
                    "unknown key. 'attacks' accepts 'groups' and 'list'; "
                    "leaving both empty runs every discovered attack.",
                    suggestion=_suggest(key, ["groups", "list"]),
                )

        groups = self._check_attack_groups(
            self._typed(attacks, "groups", "attacks.groups", list, []))
        attack_list = self._check_attack_list(
            self._typed(attacks, "list", "attacks.list", list, []),
            synthetic_versions)
        return groups, attack_list

    def _check_attack_groups(self, groups) -> List[str]:
        selectable = sorted(ATTACK_GROUPS)
        clean = []
        for index, name in enumerate(groups):
            path = f"attacks.groups[{index}]"
            if not isinstance(name, str):
                self.error("E009", path,
                           f"must be a group name (a string), but is "
                           f"{_type_name(name)}.", value=name)
                continue
            if name not in ATTACK_GROUPS:
                # Subgroups configure metrics but do not select attacks.
                extra = ""
                if name in CONFIG_GROUP_KEYS:
                    extra = (
                        f" '{name}' is a report subsection usable under "
                        f"metrics.per_group, not a selectable attack group."
                    )
                self.error(
                    "E013", path,
                    f"unknown attack group. Selectable groups: "
                    f"{', '.join(selectable)}.{extra}",
                    value=name, suggestion=_suggest(name, selectable),
                )
                continue
            if name in clean:
                self.error("E016", path, "listed more than once.", value=name)
                continue
            clean.append(name)
        return clean

    def _check_attack_list(self, attack_list, synthetic_versions) -> List[str]:
        known = sorted(self.attacks_registry) if self.attacks_registry else None
        clean = []
        for index, spec in enumerate(attack_list):
            path = f"attacks.list[{index}]"
            if not isinstance(spec, str):
                self.error("E009", path,
                           f"must be an attack name, optionally "
                           f"'AttackName:version', but is {_type_name(spec)}.",
                           value=spec)
                continue
            name, _, version = spec.partition(":")
            if known is not None and name not in known:
                self.error(
                    "E014", path,
                    f"unknown attack. Discovered attacks: {', '.join(known)}.",
                    value=spec, suggestion=_suggest(name, known),
                )
                continue
            if version and self.attacks_registry is not None:
                self._check_attack_version(
                    path, name, version, spec, synthetic_versions)
            if spec in clean:
                self.error("E016", path, "listed more than once.", value=spec)
                continue
            clean.append(spec)
        return clean

    def _versions_of(self, name):
        """Version names an attack declares in its config.json.

        A single-version config has no named presets, so it reports the
        one implicit ``default``.
        """
        raw_config = self.attacks_registry[name].get("_raw_config") or {}
        if "default" in raw_config and isinstance(raw_config["default"], dict):
            return _real_keys(raw_config)
        return ["default"]

    def _check_attack_version(self, path, name, version, spec, synthetic):
        """Reject a version that neither the plugin nor the config defines."""
        declared = self._versions_of(name)
        added = sorted(synthetic.get(name, {}))
        if version in declared or version in added:
            return
        available = sorted(set(declared) | set(added))
        hint = ""
        if added:
            hint = (
                f" Versions defined in attack_parameters: {', '.join(added)}."
            )
        self.error(
            "E015", path,
            f"attack '{name}' has no version '{version}'. Available "
            f"versions: {', '.join(available)}.{hint} To define a new one, "
            f"give ALL of this attack's parameters under "
            f"'attack_parameters.{name}:{version}'.",
            value=spec, suggestion=_suggest(version, available),
        )

    def _check_attack_parameters(self):
        """Validate ``attack_parameters`` and resolve what each key targets.

        A key is ``AttackName`` (the default version) or
        ``AttackName:version``. A version the plugin does not declare is
        *defined* by the entry when every parameter is supplied, and
        skipped with a warning when only some are -- a partly-specified
        version would silently inherit the rest from the default preset,
        which is not a version so much as a mislabelled default.

        Returns:
            ``(overrides, synthetic)`` where ``overrides`` maps
            ``(attack, version)`` to the parameters to apply, and
            ``synthetic`` maps an attack to the versions this config
            defines for it.
        """
        if "attack_parameters" not in MODE_KEYS[self.mode]:
            return {}, {}

        block = self._typed(self.raw, "attack_parameters",
                            "attack_parameters", dict, {})
        known = sorted(self.attacks_registry) if self.attacks_registry else None

        overrides: Dict[Any, Dict[str, Any]] = {}
        synthetic: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for spec in _real_keys(block):
            path = f"attack_parameters.{spec}"
            attack_name, _, version = spec.partition(":")
            version = version or None

            if known is not None and attack_name not in known:
                self.error(
                    "E017", path,
                    f"unknown attack. Keys of 'attack_parameters' are attack "
                    f"class names, optionally 'AttackName:version'. "
                    f"Discovered attacks: {', '.join(known)}.",
                    value=attack_name, suggestion=_suggest(attack_name, known),
                )
                continue

            params = self._typed(block, spec, path, dict, None)
            if params is None:
                continue

            defaults = (
                (self.attacks_registry[attack_name].get("config") or {})
                if self.attacks_registry else None
            )
            resolved = self._check_parameter_values(path, attack_name,
                                                    params, defaults)
            if not resolved:
                continue

            if self.attacks_registry is None:
                overrides[(attack_name, version)] = resolved
                continue

            declared = self._versions_of(attack_name)
            if version is None or version in declared:
                overrides[self._version_key(attack_name, version, declared)] = resolved
                continue

            # A version the plugin does not declare. Only real parameters
            # count: a config.json may carry ``_``-prefixed documentation
            # keys, and demanding those be "set" made the version
            # impossible to define at all.
            real = _real_keys(defaults or {})
            missing = [key for key in real if key not in resolved]
            if missing:
                self.warn(
                    "W010", path,
                    f"'{attack_name}' has no version '{version}', and only "
                    f"{len(resolved)} of its {len(real)} parameters "
                    f"are set here, so this entry is skipped. To define the "
                    f"version, set the missing one(s) too: "
                    f"{', '.join(sorted(missing))}.",
                )
                continue

            synthetic.setdefault(attack_name, {})[version] = resolved

        return overrides, synthetic

    @staticmethod
    def _version_key(attack_name, version, declared):
        """Normalise a target so the run loop can look it up by resolved version.

        A bare attack name means its default version; for a single-version
        attack there is no version to name, so the key carries None.
        """
        if version is not None:
            return (attack_name, version)
        return (attack_name, "default" if len(declared) > 1 else None)

    def _check_parameter_values(self, path, attack_name, params, defaults):
        """Check every parameter name and type against the plugin's defaults."""
        resolved = {}
        for key in _real_keys(params):
            key_path = f"{path}.{key}"
            if defaults is not None and key not in defaults:
                self.error(
                    "E018", key_path,
                    f"'{attack_name}' has no parameter '{key}'. Its "
                    f"parameters: {', '.join(sorted(defaults)) or '(none)'}.",
                    suggestion=_suggest(key, sorted(defaults)),
                )
                continue
            if defaults is not None and not _compatible_type(
                params[key], defaults[key]
            ):
                self.error(
                    "E019", key_path,
                    f"must be {_type_name(defaults[key])} to match the "
                    f"plugin default ({json.dumps(defaults[key], default=str)}), "
                    f"but is {_type_name(params[key])}.",
                    value=params[key],
                )
                continue
            resolved[key] = params[key]
        return resolved

    # -- statistics ----------------------------------------------------

    def _check_statistics(self) -> Optional[List[str]]:
        """The top-level list, or None when omitted (all eight then apply)."""
        if "statistics" not in self.raw:
            return None
        return self._check_statistic_list(
            self.raw["statistics"], "statistics", allow_empty=False)

    def _check_statistic_list(self, value, path, allow_empty):
        if not isinstance(value, list):
            self.error("E009", path,
                       f"must be an array of statistic names, e.g. "
                       f"[\"mean\", \"median\", \"worst_case\"], but is "
                       f"{_type_name(value)}.", value=value)
            return None

        if not value and not allow_empty:
            self.error(
                "E022", path,
                f"is empty. A metric with no statistic has nothing to report; "
                f"to drop the metric set its 'enabled' to false instead. "
                f"Available statistics: {', '.join(ALL_STATISTICS)}.",
                value=value,
            )
            return None

        clean = []
        for index, name in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(name, str):
                self.error("E009", item_path,
                           f"must be a statistic name (a string), but is "
                           f"{_type_name(name)}.", value=name)
                continue
            if name not in ALL_STATISTICS:
                self.error(
                    "E021", item_path,
                    f"unknown statistic. Available: "
                    f"{', '.join(ALL_STATISTICS)}.",
                    value=name, suggestion=_suggest(name, ALL_STATISTICS),
                )
                continue
            if name in clean:
                self.error("E023", item_path,
                           "listed more than once; each statistic is one "
                           "report column.", value=name)
                continue
            clean.append(name)
        return clean or None

    # -- metrics -------------------------------------------------------

    def _check_metrics(self, statistics, calculate):
        if "metrics" not in self.raw:
            if calculate:
                self.warn(
                    "W004", "metrics",
                    "'calculate_quality_metrics' is true but no 'metrics' "
                    "block is present, so the built-in per-group defaults "
                    "apply (the same matrix 'deepmark-benchmark --init' "
                    "ships).", severity="info",
                )
            builtin = MetricResolver.from_attack_groups()
            return builtin.defaults, builtin.per_group

        metrics = self._typed(self.raw, "metrics", "metrics", dict, {})
        for key in _real_keys(metrics):
            if key not in ("defaults", "per_group"):
                self.error(
                    "E007", f"metrics.{key}",
                    "unknown key. 'metrics' accepts 'defaults' (applies to "
                    "every attack group) and 'per_group' (overrides for one "
                    "group). Metric names go one level deeper, inside those.",
                    suggestion=_suggest(key, ["defaults", "per_group"]),
                )

        if "defaults" not in metrics:
            # A metric named nowhere is off, so a metrics block that only
            # narrows per group silently switches every other one off --
            # a very quiet way to get an almost-empty report.
            self.warn(
                "W012", "metrics.defaults",
                "missing, so every metric not named under 'per_group' is "
                "off and most tables will be empty. Add a 'defaults' block, "
                "or delete 'metrics' entirely to use the built-in matrix.",
            )

        defaults = self._check_metric_map(
            self._typed(metrics, "defaults", "metrics.defaults", dict, {}),
            "metrics.defaults",
        )

        if not calculate and _real_keys(
            self._typed(metrics, "defaults", "metrics.defaults", dict, {})
        ):
            self.warn(
                "W002", "calculate_quality_metrics",
                "is false (or absent), so the 'metrics' enable flags are "
                "ignored: only accuracy and the always-on trio (pesq, visqol, "
                "stoi) are computed. Set it to true to use the metrics block. "
                "Per-metric statistics still apply.",
            )

        per_group = self._check_per_group(metrics)
        return defaults, per_group

    def _check_per_group(self, metrics):
        if "per_group" not in metrics:
            return {}

        if self.mode in MODES_WITHOUT_GROUPS:
            self.error(
                "E028", "metrics.per_group",
                f"mode '{self.mode}' applies no attacks, so there are no "
                f"attack groups to configure. Put the metrics for this mode "
                f"under 'metrics.defaults'.",
            )
            return {}

        block = self._typed(metrics, "per_group", "metrics.per_group", dict, {})
        clean = {}
        for group_key in _real_keys(block):
            path = f"metrics.per_group.{group_key}"
            if group_key not in CONFIG_GROUP_KEYS:
                self.error(
                    "E027", path,
                    f"unknown attack group. Configurable groups: "
                    f"{', '.join(CONFIG_GROUP_KEYS)}.",
                    value=group_key,
                    suggestion=_suggest(group_key, CONFIG_GROUP_KEYS),
                )
                continue
            entry = self._typed(block, group_key, path, dict, None)
            if entry is None:
                continue
            resolved = self._check_metric_map(entry, path)
            if resolved:
                clean[group_key] = resolved
        return clean

    def _check_metric_map(self, mapping, path):
        """Validate a ``{metric: {enabled, statistics}}`` mapping."""
        forbidden = MODE_FORBIDDEN_METRICS.get(self.mode, frozenset())
        clean = {}

        for metric in _real_keys(mapping):
            metric_path = f"{path}.{metric}"
            if metric not in CANONICAL_METRIC_ORDER:
                # The efficiency names are listed as a suggestion source
                # but not as available here, so a typo for one is still
                # recognised and answered by E043 on the next attempt.
                available = [m for m in CANONICAL_METRIC_ORDER
                             if m not in EFFICIENCY_METRICS]
                self.error(
                    "E024", metric_path,
                    f"unknown metric. Available: {', '.join(available)}.",
                    value=metric,
                    suggestion=_suggest(metric, CANONICAL_METRIC_ORDER),
                )
                continue
            if metric in EFFICIENCY_METRICS:
                # It is in CANONICAL_METRIC_ORDER, so it would be accepted
                # here and then read from the efficiency section anyway --
                # the 'enabled' flag silently doing nothing. Say so.
                self.error(
                    "E043", metric_path,
                    f"'{metric}' is an efficiency metric and is configured "
                    f"in the 'efficiency' section, not in 'metrics'. Move "
                    f"it to efficiency.metrics.{metric}; setting it here "
                    f"has no effect on whether it is measured.",
                    value=metric,
                )
                continue
            if metric in forbidden:
                self.error(
                    "E025", metric_path,
                    f"metric '{metric}' does not apply to mode '{self.mode}'. "
                    f"Detection reliability scores each file as a binary "
                    f"detected/not-detected outcome, so there are no payload "
                    f"bits for a bit error rate.",
                )
                continue

            entry = self._typed(mapping, metric, metric_path, dict, None)
            if entry is None:
                continue

            resolved = {}
            for key in _real_keys(entry):
                if key not in _METRIC_ENTRY_KEYS:
                    self.error(
                        "E007", f"{metric_path}.{key}",
                        f"unknown key. A metric accepts "
                        f"{' and '.join(sorted(_METRIC_ENTRY_KEYS))}.",
                        suggestion=_suggest(key, sorted(_METRIC_ENTRY_KEYS)),
                    )

            if "enabled" in entry:
                enabled = entry["enabled"]
                if not isinstance(enabled, bool):
                    self.error("E009", f"{metric_path}.enabled",
                               f"must be true or false, but is "
                               f"{_type_name(enabled)}.", value=enabled)
                elif metric in MANDATORY_METRICS and not enabled:
                    self.error(
                        "E026", f"{metric_path}.enabled",
                        f"'{metric}' cannot be disabled -- it is the "
                        f"measurement the benchmark exists to make. Remove "
                        f"this key.",
                        value=enabled,
                    )
                else:
                    resolved["enabled"] = enabled

            if "statistics" in entry:
                if metric in STATISTICS_EXEMPT_METRICS:
                    self.error(
                        "E029", f"{metric_path}.statistics",
                        f"'{metric}' takes no statistics: it reports a count "
                        f"of exactly-recovered files and the matching rate, "
                        f"not a distribution. Remove this key.",
                        value=entry["statistics"],
                    )
                else:
                    stats = self._check_statistic_list(
                        entry["statistics"], f"{metric_path}.statistics",
                        allow_empty=False,
                    )
                    if stats is not None:
                        resolved["statistics"] = stats

            if resolved:
                clean[metric] = resolved
        return clean

    # -- crop, duration, comparison ------------------------------------

    def _check_efficiency(self, statistics):
        """Validate the ``efficiency`` section.

        Its own section rather than a bucket inside ``metrics``: these
        measure the machine, not the watermark, and a run decides whether
        to take the measurement at all separately from which quality
        metrics it wants. Memory and any later efficiency measure will be
        listed here beside latency.
        """
        if "efficiency" not in self.raw:
            return {}

        block = self._typed(self.raw, "efficiency", "efficiency", dict, {})
        for key in _real_keys(block):
            if key not in ("enabled", "metrics"):
                self.error(
                    "E040", f"efficiency.{key}",
                    "unknown key. The efficiency section takes 'enabled' "
                    "and 'metrics'.",
                    value=key, suggestion=_suggest(key, ["enabled", "metrics"]),
                )

        enabled = self._typed(block, "enabled", "efficiency.enabled", bool,
                              False)

        metrics = self._typed(block, "metrics", "efficiency.metrics", dict, {})
        clean = {}
        for name in _real_keys(metrics):
            path = f"efficiency.metrics.{name}"
            if name not in EFFICIENCY_METRICS:
                self.error(
                    "E041", path,
                    f"unknown efficiency metric. Available: "
                    f"{', '.join(EFFICIENCY_METRICS)}.",
                    value=name,
                    suggestion=_suggest(name, list(EFFICIENCY_METRICS)),
                )
                continue
            entry = self._typed(metrics, name, path, dict, {})
            for key in _real_keys(entry):
                if key not in _METRIC_ENTRY_KEYS:
                    self.error(
                        "E042", f"{path}.{key}",
                        "unknown key. A metric entry takes 'enabled' and "
                        "'statistics'.",
                        value=key,
                        suggestion=_suggest(key, sorted(_METRIC_ENTRY_KEYS)),
                    )
            cleaned = {}
            if "enabled" in entry:
                cleaned["enabled"] = self._typed(
                    entry, "enabled", f"{path}.enabled", bool, True)
            if "statistics" in entry:
                cleaned["statistics"] = self._check_statistic_list(
                    entry["statistics"], f"{path}.statistics",
                    allow_empty=False)
            clean[name] = cleaned

        if not enabled and clean:
            # Info, not a warning: the shipped templates carry this section
            # switched off with its metrics listed as documentation, so a
            # warning here would fire on every default run.
            self.warn(
                "W013", "efficiency.enabled",
                "is false, so no timing is measured and the entries under "
                "'efficiency.metrics' are ignored. Set it to true to record "
                "them.", severity="info",
            )

        return {"enabled": enabled, "metrics": clean}

    def _check_crop(self) -> Optional[float]:
        if "crop_before_attack" not in MODE_KEYS[self.mode]:
            return None
        crop = self.raw.get("crop_before_attack")
        if crop is None:
            return None
        if isinstance(crop, bool) or not isinstance(crop, (int, float)):
            self.error(
                "E030", "crop_before_attack",
                f"must be a percentage (a number) or null, but is "
                f"{_type_name(crop)}.", value=crop,
            )
            return None
        if not 0 < crop < 100:
            self.error(
                "E031", "crop_before_attack",
                "must be greater than 0 and less than 100: it is the "
                "percentage cropped from the start of the watermarked audio. "
                "Use null to disable cropping.",
                value=crop,
            )
            return None
        return float(crop)

    def _check_duration_groups(self):
        block = self._typed(self.raw, "duration_groups", "duration_groups",
                            dict, {})
        for key in _real_keys(block):
            if key not in ("boundaries", "include_overall"):
                self.error(
                    "E007", f"duration_groups.{key}",
                    "unknown key. 'duration_groups' accepts 'boundaries' and "
                    "'include_overall'.",
                    suggestion=_suggest(key, ["boundaries", "include_overall"]),
                )

        boundaries = self._typed(block, "boundaries",
                                 "duration_groups.boundaries", list, [])
        clean = []
        for index, value in enumerate(boundaries):
            path = f"duration_groups.boundaries[{index}]"
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                self.error("E032", path,
                           f"must be a duration in seconds (a number), but is "
                           f"{_type_name(value)}.", value=value)
                continue
            if value <= 0:
                self.error("E032", path,
                           "must be greater than 0 seconds.", value=value)
                continue
            if value in clean:
                self.error("E034", path,
                           "duplicate boundary; each value opens one duration "
                           "bin.", value=value)
                continue
            clean.append(float(value))

        if clean != sorted(clean):
            self.error(
                "E033", "duration_groups.boundaries",
                f"must be in ascending order, so the bins read left to right. "
                f"Sorted, this is {sorted(clean)}.",
                value=clean,
            )
            clean = sorted(clean)

        include_overall = self._typed(
            block, "include_overall", "duration_groups.include_overall",
            bool, False,
        )
        if include_overall and not clean:
            self.warn(
                "W005", "duration_groups.include_overall",
                "is true but no boundaries are set, so there is nothing to "
                "split and no separate 'Overall' section is added.",
            )
        return clean, include_overall

    def _check_comparison(self, resolver) -> str:
        if "comparison" not in MODE_KEYS[self.mode]:
            return "mean"

        block = self._typed(self.raw, "comparison", "comparison", dict, {})
        for key in _real_keys(block):
            if key != "primary_statistic":
                self.error(
                    "E007", f"comparison.{key}",
                    "unknown key. 'comparison' accepts 'primary_statistic'.",
                    suggestion=_suggest(key, ["primary_statistic"]),
                )

        configured = resolver.statistics_for(None, "accuracy")
        # Unset means "whichever statistic accuracy leads with", not a hard
        # "mean" -- a config that drops the mean has not made a mistake.
        if "primary_statistic" not in block:
            return configured[0] if configured else "mean"

        primary = block["primary_statistic"]
        if not isinstance(primary, str) or primary not in ALL_STATISTICS:
            self.error(
                "E035" if isinstance(primary, str) else "E009",
                "comparison.primary_statistic",
                f"must be one of: {', '.join(ALL_STATISTICS)}. It selects the "
                f"statistic shown in the colour-ranked multi-model table; the "
                f"others each get their own table below it.",
                value=primary,
                suggestion=_suggest(primary, ALL_STATISTICS),
            )
            return configured[0] if configured else "mean"

        if primary not in configured:
            self.error(
                "E036", "comparison.primary_statistic",
                f"'{primary}' is not among the statistics configured for "
                f"accuracy ({', '.join(configured)}), so it is never computed "
                f"and the main comparison table would be empty. Add it to "
                f"accuracy's statistics, or pick one of those.",
                value=primary,
            )
            return "mean"
        return primary

    # -- cross-cutting warnings ----------------------------------------

    def _check_group_coverage(self, per_group, groups, attack_list):
        """Note per_group entries for groups this run will not reach.

        Not an error: keeping a full matrix in the file and narrowing the
        attack selection per run is the expected workflow.
        """
        if not per_group:
            return
        selected = self._groups_in_run(groups, attack_list)
        if selected is None:
            return
        for group_key in per_group:
            if group_key not in selected:
                self.warn(
                    "W001", f"metrics.per_group.{group_key}",
                    f"no attack in this run belongs to '{group_key}', so this "
                    f"section is unused. Nothing to fix if that is intended.",
                    severity="info",
                )

    def _check_parameter_coverage(self, overrides, synthetic, groups,
                                  attack_list):
        """Note attack_parameters entries this run will not apply.

        Narrowing the attack selection leaves parameter entries behind,
        and an entry that silently does nothing is the same trap as an
        unused ``per_group`` section -- so it gets the same treatment.
        """
        if not overrides and not synthetic:
            return
        if not groups and not attack_list:
            # Everything runs, so every entry is reachable.
            return

        selected = {spec.partition(":")[0] for spec in attack_list}
        for group_key in groups:
            if group_key in ATTACK_GROUPS:
                selected.update(ATTACK_GROUPS[group_key]["attacks"])

        configured = {attack for attack, _version in overrides}
        configured.update(synthetic)
        for attack in sorted(configured - selected):
            self.warn(
                "W011", f"attack_parameters.{attack}",
                f"this run does not select '{attack}', so its parameters are "
                f"unused. Add it to attacks.list, or remove the entry.",
                severity="info",
            )

        # A version defined here but never selected is also worth saying,
        # since defining one is deliberate work.
        for attack, versions in synthetic.items():
            if attack not in selected:
                continue
            named = {
                spec.partition(":")[2] for spec in attack_list
                if spec.partition(":")[0] == attack
            }
            bare = any(spec == attack for spec in attack_list) or any(
                attack in ATTACK_GROUPS.get(g, {}).get("attacks", [])
                for g in groups
            )
            if bare:
                continue
            for version in versions:
                if version not in named:
                    self.warn(
                        "W011", f"attack_parameters.{attack}:{version}",
                        f"version '{version}' is defined but not selected: "
                        f"attacks.list names other versions of '{attack}' and "
                        f"not this one. Add '{attack}:{version}' to run it.",
                        severity="info",
                    )

    def _groups_in_run(self, groups, attack_list):
        """Group keys reachable by this run's attack selection, or None for all."""
        from deepmarkpy.utils.attack_groups import (
            OTHER_GROUP_KEY, get_group_for_attack, get_subgroup_for_attack,
            subgroups_of,
        )

        if not groups and not attack_list:
            return None

        reachable = set()
        for group_key in groups:
            reachable.add(group_key)
            reachable.update(subgroups_of(group_key))
        for spec in attack_list:
            name = spec.partition(":")[0]
            reachable.add(get_group_for_attack(name) or OTHER_GROUP_KEY)
            subgroup = get_subgroup_for_attack(name)
            if subgroup:
                reachable.add(subgroup)
        return reachable

    def _check_nisqa_cost(self, resolver):
        """Say plainly that enabling fewer NISQA dimensions costs the same.

        All five come back from one forward pass, so trimming the list
        shortens tables and saves nothing.
        """
        enabled = {
            metric for group_key in list(resolver.per_group) + [None]
            for metric in resolver.metrics_for_group(group_key)
            if metric in NISQA_METRICS
        }
        if enabled and len(enabled) < len(NISQA_METRICS):
            missing = [m for m in NISQA_METRICS if m not in enabled]
            self.warn(
                "W003", "metrics",
                f"{len(enabled)} of {len(NISQA_METRICS)} NISQA dimensions are "
                f"enabled ({', '.join(sorted(enabled))}). All five come from a "
                f"single NISQA request, so leaving out "
                f"{', '.join(missing)} makes the tables narrower but costs the "
                f"same to run.", severity="info",
            )


_TYPE_LABELS = {
    dict: "a JSON object", list: "an array", str: "a string",
    bool: "true or false", int: "a number", float: "a number",
}


def _compatible_type(value, default) -> bool:
    """Whether a config value matches the shape of a plugin's default.

    Ints are accepted where a float is expected (JSON writes ``2`` for
    ``2.0``); bools are kept distinct from numbers, which Python's
    ``isinstance`` otherwise conflates.
    """
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, list):
        return isinstance(value, list)
    if isinstance(default, str):
        return isinstance(value, str)
    if default is None:
        return True
    return isinstance(value, type(default))
