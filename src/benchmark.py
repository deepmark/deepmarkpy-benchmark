import inspect
import logging
import os

import numpy as np
import soundfile as sf

from plugin_manager import PluginManager
from utils.utils import load_audio, snr
from utils.metrics import pesq_wrapper, psnr, stoi_wrapper, si_sdr


logger = logging.getLogger(__name__)


class Benchmark:
    """
    A class to perform various attacks on watermarking models and benchmark their performance.
    """

    def __init__(self):
        """
        Initialize Benchmark class with PluginManager.
        """
        self.plugin_manager = PluginManager()
        # Now these are dicts of the form { "class_name": {"class": ActualClass, "config": {...}} }
        self.attacks = self.plugin_manager.get_attacks()
        self.models = self.plugin_manager.get_models()

    def get_available_args(self):
        valid_args = {}
        attacks = []
        models = self.models.keys()
        attacks = self.attacks.keys()
        for attack in attacks:
            config = self.attacks[attack]["config"]
            if config is not None:
                for key, value in config.items():
                    valid_args[key] = value
        return list(models), list(attacks), valid_args

    def show_available_plugins(self):
        """
        Print out all discovered models and attacks, including any __init__ parameters
        and key-value pairs from config.json (defaults).
        """
        logger.info("===== Available Models =====")
        for model_name, model_entry in self.models.items():
            model_cls = model_entry["class"]
            config = model_entry.get("config") or {}

            signature = inspect.signature(model_cls.__init__)
            params = [p for p in signature.parameters.values() if p.name != "self"]

            init_params = {
                p.name: (None if p.default is inspect.Parameter.empty else p.default)
                for p in params
            }

            logger.info(f"\nModel: {model_name}")
            logger.info(f"  - Constructor parameters: {init_params}")

            logger.info("  - Arguments defaults:")
            if config:
                for key, val in config.items():
                    logger.info(f"    {key}: {val}")
            else:
                logger.info("    (none found)")

        logger.info("\n===== Available Attacks =====")
        for attack_name, attack_entry in self.attacks.items():
            attack_cls = attack_entry["class"]
            config = attack_entry.get("config") or {}

            signature = inspect.signature(attack_cls.__init__)
            params = [p for p in signature.parameters.values() if p.name != "self"]
            init_params = {
                p.name: (None if p.default is inspect.Parameter.empty else p.default)
                for p in params
            }

            logger.info(f"\nAttack: {attack_name}")
            logger.info(f"  - Constructor parameters: {init_params}")

            logger.info("  - Argument defaults:")
            if config:
                for key, val in config.items():
                    logger.info(f"    {key}: {val}")
            else:
                logger.info("    (none found)")

    def run(
        self,
        filepaths,
        wm_model,
        watermark_data=None,
        attack_types=None,
        sampling_rate=None,
        verbose=False,
        save_audio= False,
        output_dir="audio_processed",
        **kwargs,
    ):
        """
        Benchmark the watermarking models against selected attacks.

        Args:
            filepaths (str or list): Path(s) to the audio file(s) to benchmark.
            model_name (str): The model to benchmark (e.g., 'AudioSeal', 'WavMark', 'SilentCipher').
            watermark_data (np.ndarray, optional): The binary watermark data to embed. Defaults to random message.
            attack_types (list, optional): A list of attack types to perform. Defaults to all available attacks.
            sampling_rate (int, optional): Target sampling rate for loading audio. Defaults to None.
            verbose (bool, optional): Print verbose info. Defaults to True.
            save_audio (bool, optional): Whether to save processed audio files. Defaults to True.
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

        # If user doesn't specify attacks, use them all
        attack_types = attack_types or list(self.attacks.keys())
        results = {}

        if wm_model not in self.models:
            raise ValueError(
                f"Model '{wm_model}' not found. Available: {list(self.models.keys())}"
            )

        model_cls = self.models[wm_model]["class"]
        model_instance = model_cls()

        if sampling_rate is None:
            sampling_rate = self.models[wm_model]["config"]["sampling_rate"]
            logger.info(f"Using default sampling rate {sampling_rate} for model {wm_model}")

        attack_kwargs = {
            **kwargs,
            "model": model_instance,
            "watermark_data": watermark_data,
            "sampling_rate": sampling_rate,
            "models": self.models,
        }

        for filepath in filepaths:
            if verbose:
                logger.info(f"\nProcessing file: {filepath}")
            results[filepath] = {}

            # Get base filename without extension
            base_filename = os.path.splitext(os.path.basename(filepath))[0]

            # If no user-supplied watermark, pick a random message size
            if watermark_data is None:
                watermark_data = model_instance.generate_watermark()
                attack_kwargs["watermark_data"] = model_instance.generate_watermark()

            # Load audio
            audio, sampling_rate = load_audio(filepath, target_sr=sampling_rate)
            logger.info(f"Sampling rate is: {sampling_rate}")
            attack_kwargs["orig_audio"] = audio

            # Embed watermark
            watermarked_audio = model_instance.embed(
                audio=audio, watermark_data=watermark_data, sampling_rate=sampling_rate
            )

            # Save watermarked audio
            if save_audio:
                watermarked_filename = f"{base_filename}_watermarked.wav"
                watermarked_path = os.path.join(output_dir, watermarked_filename)
                sf.write(watermarked_path, watermarked_audio, sampling_rate)
                if verbose:
                    logger.info(f"  Saved watermarked audio: {watermarked_filename}")

            # Apply each attack and compute metrics
            for attack_name in attack_types:
                if attack_name not in self.attacks:
                    logger.warning(f"Attack '{attack_name}' not found. Skipping.")
                    continue

                if verbose:
                    logger.info(f"  Applying attack: {attack_name}")

                attack_instance = self.attacks[attack_name]["class"]()

                if (attack_name =="CrossModelAttack"):
                    
                    different_model_name = kwargs.get("different_model_name")
                    logger.info(f"Different model is chosen and it's {different_model_name}")
                    different_model_cls = self.models[different_model_name]["class"]
                    different_model_instance = different_model_cls()

                    attacked_audio, different_watermark = attack_instance.apply(
                        watermarked_audio, **attack_kwargs
                    )
                    attacked_audio_metrics, _ = attack_instance.apply(
                        audio, **attack_kwargs
                    )


                #in case of the collusion mod attack
                elif (attack_name == "ZeroBitCollusionAttack"):
                    attack_kwargs["original_audio_collusion"] = audio

                    attacked_audio = attack_instance.apply(
                        watermarked_audio, **attack_kwargs
                    )
                    attacked_audio_metrics = attack_instance.apply(
                        audio, **attack_kwargs
                    )

                else:
                    attacked_audio = attack_instance.apply(
                        watermarked_audio, **attack_kwargs
                    )
                    attacked_audio_metrics = attack_instance.apply(
                        audio, **attack_kwargs
                    )

                #attacked_audio_metrics=np.squeeze(attacked_audio_metrics)
                # Save attacked audio
                if save_audio:
                    logger.info(attacked_audio.shape)
                    logger.info(type(attacked_audio))
                    if attacked_audio.ndim == 1:
                        attacked_audio = np.expand_dims(attacked_audio, axis=1)
                    attacked_filename = f"{base_filename}_{attack_name}.wav"
                    attacked_path = os.path.join(output_dir, attacked_filename)
                    sf.write(attacked_path, attacked_audio, 16000)
                    if verbose:
                        logger.info(f"Saved attacked audio: {attacked_filename}")
                
                if isinstance(attacked_audio, np.ndarray):
                    attacked_audio = attacked_audio.squeeze()   # (N,1) -> (N,)
                    #attacked_audio = attacked_audio.tolist()
                detected_message = model_instance.detect(attacked_audio, sampling_rate)

                if (attack_name =="CrossModelAttack"):
                    different_detected_message = different_model_instance.detect(attacked_audio, sampling_rate)
                    if (different_model_name=="PerthModel"):
                        if isinstance(different_detected_message, np.ndarray):
                            different_accuracy = different_detected_message.tolist()
                        else:
                            different_accuracy = different_detected_message
                    else:
                        different_accuracy = self.compare_watermarks(different_watermark, different_detected_message)
                

                if abs(len(audio) - len(attacked_audio)) > 1:
                    snr_val = "N/A"
                else:
                    snr_val = snr(audio, attacked_audio)
                

                sr_scalar = int(sampling_rate) if isinstance(sampling_rate, (np.ndarray, list)) else sampling_rate
                stoi_val = stoi_wrapper(audio, attacked_audio_metrics, sr_scalar)
                pesq_val = pesq_wrapper(audio, attacked_audio_metrics, sr_scalar, 'wb')


                if (wm_model=="PerthModel"):
                    if isinstance(detected_message, np.ndarray):
                        accuracy = detected_message.tolist()
                    else:
                       accuracy = detected_message
                else:             
                    accuracy = self.compare_watermarks(watermark_data, detected_message)
                    
                results[filepath][attack_name] = {
                    "accuracy": accuracy,
                    "stoi": stoi_val,
                    "pesq": pesq_val
                    }

                if attack_name == "CrossModelAttack":
                    results[filepath][attack_name]["accuracy_second"] = different_accuracy

        return results

    def compute_mean_accuracy(self, results):

        attack_accuracies = {}

        for _, attack_dict in results.items():
            for attack_name, metrics in attack_dict.items():
                if attack_name not in attack_accuracies:
                    attack_accuracies[attack_name] = {
                        "accuracy": [],
                        "accuracy_second": []
                    }

                attack_accuracies[attack_name]["accuracy"].append(metrics["accuracy"])

                if "accuracy_second" in metrics:
                    attack_accuracies[attack_name]["accuracy_second"].append(
                        metrics["accuracy_second"]
                    )

        mean_accuracies = {}

        for attack_name, acc in attack_accuracies.items():
            mean_accuracies[attack_name] = {}

            mean_accuracies[attack_name]["accuracy_mean"] = float(
                np.mean([a for a in acc["accuracy"] if a is not None])
            )

            if acc["accuracy_second"]:
                mean_accuracies[attack_name]["accuracy_second_mean"] = float(
                    np.mean([a for a in acc["accuracy_second"] if a is not None])
                )

        return mean_accuracies


    def compare_watermarks(self, original, detected):
        """
        Compare the original and detected watermarks.

        Args:
            original (np.ndarray): The original binary watermark.
            detected (np.ndarray): The detected binary watermark.

        Returns:
            float: The accuracy of the detected watermark (percentage).
        """
        if (detected is None) or (np.any(detected == np.array(None))):
            logger.info("Detected watermark is none!")
            return 50.00
        matches = np.sum(original == detected)
        return (matches / len(original)) * 100