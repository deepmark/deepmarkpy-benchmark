import argparse
import json
import os

import logging

from benchmark import Benchmark
from report_generator import generate_benchmark_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--wm_model",
        type=str,
        choices=models,
        required=True,
        help="Watermarking model to use.",
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

    # Add verbose flag
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Enable verbose logging",
    )

    # Dynamically add configuration parameters from the available plugins
    for arg, default_value in valid_args.items():
        if isinstance(default_value, bool):
            parser.add_argument(
                f"--{arg}",
                action="store_true",
                help=f"Enable {arg} (default: {default_value})",
            )
        else:
            parser.add_argument(
                f"--{arg}",
                type=type(default_value),
                default=default_value,
                help=f"Set {arg} (default: {default_value})",
            )

    args = parser.parse_args()

    if args.verbose:
        logger.getLogger().setLevel(logger.DEBUG)  # Set root logger level
        logger.debug("Verbose logging enabled.")

    args_dict = vars(args)

    try:
        all_files = os.listdir(args.wav_files_dir)
        filepaths = [
            os.path.join(args.wav_files_dir, f)
            for f in all_files
            if f.lower().endswith(".wav")
        ]
        if not filepaths:
            logger.error(f"No .wav files found in directory: {args.wav_files_dir}")
            return  # Exit if no files found
        logger.info(f"Found {len(filepaths)} .wav files to process.")
    except FileNotFoundError:
        logger.error(f"Audio directory not found: {args.wav_files_dir}")
        return
    except Exception as e:
        logger.error(f"Error accessing audio directory {args.wav_files_dir}: {e}")
        return

    results = benchmark.run(filepaths=filepaths, **args_dict)

    with open("benchmark_results.json", "w") as fp:
        json.dump(results, fp, indent=4)

    logger.info("Benchmark completed. Results saved to benchmark_results.json")

    with open("benchmark_stats.json", "w") as fp:
        json.dump(benchmark.compute_mean_accuracy(results), fp, indent=4)

    logger.info("Benchmark statistics saved to benchmark_stats.json")

    try:
        logger.info("Generating benchmark report...")
        latex_path, chart_path = generate_benchmark_report(
            stats_file="benchmark_stats.json",
            model_name=args.wm_model,
            report_dir="report"
        )
        logger.info(f"Benchmark report generated: {latex_path}")
        logger.info(f"Chart saved: {chart_path}")
    except Exception as e:
        logger.error(f"Failed to generate benchmark report: {e}")
        logger.info("Benchmark data is still available in JSON files")



if __name__ == "__main__":
    main()
