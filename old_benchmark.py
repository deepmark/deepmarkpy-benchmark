import inspect
import logging

import numpy as np
import os
import soundfile as sf

from plugin_manager import PluginManager
from utils.utils import load_audio, snr, pesq_wrapper, psnr, stoi_wrapper, si_sdr


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
            **kwargs: Additional parameters for specific attacks.

        Returns:
            dict: A dictionary containing benchmark results for each file and attack.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

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

            # If no user-supplied watermark, pick a random message size
            if watermark_data is None:
                watermark_data = model_instance.generate_watermark()
                attack_kwargs["watermark_data"] = model_instance.generate_watermark()

            # Load audio
            audio, sampling_rate = load_audio(filepath, target_sr=sampling_rate)
            attack_kwargs["orig_audio"] = audio

            # Embed watermark
            watermarked_audio = model_instance.embed(
                audio=audio, watermark_data=watermark_data, sampling_rate=sampling_rate
            )

            # Apply each attack and compute metrics
            for attack_name in attack_types:
                if attack_name not in self.attacks:
                    logger.warning(f"Attack '{attack_name}' not found. Skipping.")
                    continue

                if verbose:
                    logger.info(f"  Applying attack: {attack_name}")

                attack_instance = self.attacks[attack_name]["class"]()

                #in case of the collusion mod attack
                if (attack_name == "ZeroBitCollusionAttack"):
                    attack_kwargs["original_audio_collusion"] = audio

                attacked_audio = attack_instance.apply(
                    watermarked_audio, **attack_kwargs
                )
                
                detected_message = model_instance.detect(attacked_audio, sampling_rate)
                print("detected message is ", detected_message, type(detected_message))
                print(np.any(detected_message == None))
                print(detected_message, "this is the detected message")

                if abs(len(audio) - len(attacked_audio)) > 1:
                    snr_val = "N/A"
                else:
                    snr_val = snr(audio, attacked_audio)
                
                print("sampling rate is ", sampling_rate)
                sr_scalar = int(sampling_rate) if isinstance(sampling_rate, (np.ndarray, list)) else sampling_rate
                print("sr scalar is ", sr_scalar)
                psnr_val = psnr(watermarked_audio, attacked_audio)
                stoi_val = stoi_wrapper(audio, attacked_audio, sr_scalar)
                si_sdr_val = si_sdr(audio, attacked_audio)
                pesq_val = pesq_wrapper(audio, attacked_audio, sr_scalar, 'wb')


                if (wm_model=="PerthModel"):
                    #print("accuracy is ", detected_message)
                    if isinstance(detected_message, np.ndarray):
                        accuracy = detected_message.tolist()
                    else:
                       accuracy = detected_message
                
                else:             
                    accuracy = self.compare_watermarks(watermark_data, detected_message)
                    
                results[filepath][attack_name] = {
                    "accuracy": accuracy,
                    "snr": snr_val,
                    "psnr": psnr_val,
                    "stoi": stoi_val,
                    "si_sdr": si_sdr_val,
                    "pesq": pesq_val
                    }

        return results

    def compute_mean_accuracy(self, results):
        """
        Compute the mean accuracy for each attack across all files.

        Args:
            results (dict): Dictionary where each key is a filepath, and the value is another dictionary
                            containing attack results with accuracy and other metrics.

        Returns:
            dict: A dictionary with attacks as keys and their mean accuracy as values.
        """
        attack_accuracies = {}

        for _, attack_dict in results.items():
            for attack_name, metrics in attack_dict.items():
                if attack_name not in attack_accuracies:
                    attack_accuracies[attack_name] = []
                attack_accuracies[attack_name].append(metrics["accuracy"])

        mean_accuracies = {
            attack_name: np.mean([a for a in accuracies if a is not None])
            for attack_name, accuracies in attack_accuracies.items()
        }
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
            logger.info("detected is none!!!")
            return 50.0
        matches = np.sum(original == detected)
        return (matches / len(original)) * 100