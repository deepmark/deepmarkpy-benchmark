import argparse
import json
import os
import shutil

import logging

from benchmark import Benchmark
from utils.report_generator import generate_benchmark_report
from utils.attack_groups import ATTACK_GROUPS, get_attacks_for_groups

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


import numpy as np


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

def main():
    benchmark = Benchmark()

    models, attacks, valid_args = benchmark.get_available_args()

    parser = argparse.ArgumentParser(description="Run DeepMark Benchmark CLI")

    # Add model and attack selection
    parser.add_argument(
        "--wav_files_dir",
        type=str,
        help="Path to the directory containing .wav files.",
        required=True,
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--wm_model",
        type=str,
        choices=models,
        help="Single watermarking model to benchmark.",
    )
    model_group.add_argument(
        "--wm_models",
        type=str,
        nargs="+",
        choices=models,
        metavar="MODEL",
        help="Multiple watermarking models to benchmark and compare.",
    )
    parser.add_argument(
        "--no_attacks",
        action="store_true",
        default=False,
        help=(
            "Skip all attacks. Only embed and detect the watermark to "
            "measure baseline model fidelity (accuracy and confidence)."
        ),
    )
    parser.add_argument(
        "--attack_types",
        type=str,
        nargs="*",
        choices=attacks,
        default=None,
        metavar="ATTACK",
        help="List of attacks to apply. Allowed values: " + ", ".join(attacks),
    )
    parser.add_argument(
        "--attack_groups",
        type=str,
        nargs="+",
        choices=list(ATTACK_GROUPS.keys()),
        default=None,
        metavar="GROUP",
        help="Attack group(s) to run. Available: " + ", ".join(ATTACK_GROUPS.keys()),
    )

    # Add verbose flag
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--calculate_quality_metrics",
        action="store_true",
        default=False,
        help=(
            "Calculate audio quality metrics (PESQ, PSNR, SI-SDR, MCD, ViSQOL) "
            "and speech intelligibility metrics (STOI, SII, NCM); "
            "generates a detailed report."
        ),
    )

    parser.add_argument(
        "--crop_before_attack",
        type=float,
        default=None,
        help=(
            "Optional: crop this percentage from the beginning of the watermarked "
            "audio before each attack is applied (uses CropBeginningAttack). "
            "Disabled when not set."
        ),
    )

    parser.add_argument(
        "--save_audio",
        action="store_true",
        default=False,
        help=(
            "Save watermarked and attacked audio files to disk for manual inspection. "
            "Files are written to <report_dir>/audio/ (per-model subfolder in multi-model mode)."
        ),
    )

    # Dynamically add configuration parameters from the available plugins
    for arg, default_value in valid_args.items():
        if isinstance(default_value, bool):
            # Use BooleanOptionalAction to support both --flag and --no-flag
            parser.add_argument(
                f"--{arg}",
                action=argparse.BooleanOptionalAction,
                default=default_value,
                help=f"Enable/disable {arg} (default: {default_value})",
            )
        else:
            parser.add_argument(
                f"--{arg}",
                type=type(default_value),
                default=default_value,
                help=f"Set {arg} (default: {default_value})",
            )

    args = parser.parse_args()

    # Resolve attack groups into individual attack types
    if args.attack_groups:
        attacks_from_groups = get_attacks_for_groups(args.attack_groups)
        # Filter to only attacks that are actually available
        available = set(attacks)
        attacks_from_groups = [a for a in attacks_from_groups if a in available]
        if args.attack_types:
            # Combine with explicitly listed attacks
            combined = list(dict.fromkeys(args.attack_types + attacks_from_groups))
            args.attack_types = combined
        else:
            args.attack_types = attacks_from_groups
        logger.info(f"Attack groups {args.attack_groups} resolved to {len(args.attack_types)} attacks")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    try:
        all_files = os.listdir(args.wav_files_dir)
        filepaths = [
            os.path.join(args.wav_files_dir, f)
            for f in all_files
            if f.lower().endswith(".wav") or f.lower().endswith(".mp3")
        ]
        if not filepaths:
            logger.error(f"No .wav files found in directory: {args.wav_files_dir}")
            return
        logger.info(f"Found {len(filepaths)} .wav files to process.")
    except FileNotFoundError:
        logger.error(f"Audio directory not found: {args.wav_files_dir}")
        return
    except Exception as e:
        logger.error(f"Error accessing audio directory {args.wav_files_dir}: {e}")
        return

    if args.no_attacks:
        # --- No-attacks mode: embed + detect only ---
        model_names = args.wm_models if args.wm_models else [args.wm_model]
        _clean_report_dir("report")
        run_no_attacks_mode(benchmark, filepaths, model_names, args)
    elif args.wm_models and len(args.wm_models) > 1:
        # --- Multi-model mode ---
        run_multiple_models(benchmark, filepaths, args.wm_models, args)
    else:
        # --- Single-model mode (--wm_model or --wm_models with one entry) ---
        model_name = args.wm_models[0] if args.wm_models else args.wm_model
        _clean_report_dir("report")
        run_single_model(benchmark, filepaths, model_name, args)


_DEEPMARK_ASSETS = {"deepmark.cls", "deepmark-logo.png", "deepmark-logo.pdf", "deepmark-logo.jpg"}


def _clean_report_dir(report_dir):
    """Remove generated files from report dir, preserving deepmark assets."""
    if not os.path.exists(report_dir):
        return
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


def run_no_attacks_mode(benchmark, filepaths, model_names, args):
    """Run embed+detect without attacks for one or more models."""
    report_dir = "report"
    os.makedirs(report_dir, exist_ok=True)

    from utils.no_attacks_report_generator import generate_no_attacks_report

    all_results = {}
    for model_name in model_names:
        logger.info(f"Running no-attacks baseline for: {model_name}")
        try:
            results = benchmark.run_no_attacks(
                filepaths=filepaths,
                wm_model=model_name,
                sampling_rate=None,
                verbose=args.verbose,
                calculate_quality_metrics=args.calculate_quality_metrics,
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
            )
            logger.info(f"No-attacks report updated: {latex_path}")
        except Exception as e:
            logger.error(f"Failed to generate no-attacks report: {e}")

    if not all_results:
        logger.error("No models completed successfully.")


def run_single_model(benchmark, filepaths, model_name, args, output_dir=None):
    """Run benchmark for a single model, save results and generate reports.

    Args:
        benchmark: Benchmark instance
        filepaths: List of audio file paths
        model_name: Name of the watermarking model
        args: Parsed CLI arguments
        output_dir: Optional directory for outputs (default: report/)

    Returns:
        Tuple of (results, flattened_stats) for use in comparative reports
    """
    args_dict = vars(args).copy()
    args_dict["wm_model"] = model_name

    report_dir = output_dir or "report"
    os.makedirs(report_dir, exist_ok=True)

    # Keep audio files in a dedicated subfolder so they don't clutter
    # the report directory next to .tex/.pdf/.json outputs.
    if args.save_audio:
        args_dict["output_dir"] = os.path.join(report_dir, "audio")

    results = benchmark.run(filepaths=filepaths, **args_dict)

    results_path = os.path.join(report_dir, "benchmark_results.json")
    with open(results_path, "w") as fp:
        json.dump(to_json_safe(results), fp, indent=4)
    logger.info(f"Results saved to {results_path}")

    stats = benchmark.compute_mean_accuracy(results)
    flattened_stats = {attack: metrics["accuracy_mean"] for attack, metrics in stats.items()}
    # Persist the full per-attack stats (accuracy_mean + always-on
    # metric means) so the basic report can show PESQ/ViSQOL/STOI
    # averages without needing the detailed-report flag. The flat
    # accuracy-only mapping stays in memory for the comparative report.
    stats_path = os.path.join(report_dir, "benchmark_stats.json")
    with open(stats_path, "w") as fp:
        json.dump(to_json_safe(stats), fp, indent=4)
    logger.info(f"Statistics saved to {stats_path}")

    try:
        latex_path, chart_path = generate_benchmark_report(
            stats_file=stats_path,
            model_name=model_name,
            report_dir=report_dir,
        )
        logger.info(f"Benchmark report generated: {latex_path}")
    except Exception as e:
        logger.error(f"Failed to generate benchmark report: {e}")

    if args.calculate_quality_metrics:
        try:
            from utils.detailed_report_generator import DetailedReportGenerator
            detailed_generator = DetailedReportGenerator(report_dir=report_dir)
            model_config = benchmark.models.get(model_name, {}).get("config") or {}
            is_zero_bit = model_config.get("is_zero_bit", False)
            latex_path = detailed_generator.generate_full_report(
                results, model_name=model_name, is_zero_bit=is_zero_bit,
            )
            logger.info(f"Detailed report saved to: {latex_path}")
        except Exception as e:
            logger.error(f"Failed to generate detailed report: {e}")

    return results, flattened_stats


def run_multiple_models(benchmark, filepaths, model_names, args):
    """Run benchmark for multiple models and generate comparative report."""
    report_base = "report"
    _clean_report_dir(report_base)
    logger.info(f"Cleaned previous results from {report_base}/")

    all_results = {}
    all_stats = {}

    failed_models = []
    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running benchmark for model: {model_name}")
        logger.info(f"{'='*60}")

        model_dir = os.path.join(report_base, model_name)
        _copy_deepmark_assets(report_base, model_dir)
        try:
            results, flattened_stats = run_single_model(
                benchmark, filepaths, model_name, args, output_dir=model_dir,
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
        all_stats[model_name] = flattened_stats

    if failed_models:
        logger.warning(
            f"Skipped {len(failed_models)} model(s) due to errors: "
            f"{', '.join(failed_models)}"
        )
    if not all_results:
        logger.error("No models completed successfully. Skipping comparative report.")
        return
    if len(all_results) < 2:
        # Comparative report needs at least two models to compare.
        only = next(iter(all_results))
        logger.info(
            f"Only {only} completed successfully; skipping comparative "
            f"report (single-model outputs are already in report/{only}/)."
        )
        return

    # Generate comparative report
    try:
        from utils.comparative_report_generator import ComparativeReportGenerator
        logger.info("Generating comparative report...")
        comp_dir = os.path.join(report_base, "comparison")
        _copy_deepmark_assets(report_base, comp_dir)
        comp_generator = ComparativeReportGenerator(report_dir=comp_dir)
        comp_generator.generate_full_report(
            all_results, all_stats,
            calculate_quality_metrics=args.calculate_quality_metrics,
        )
        logger.info(f"Comparative report saved to: {comp_dir}")
    except Exception as e:
        logger.error(f"Failed to generate comparative report: {e}")


if __name__ == "__main__":
    main()
