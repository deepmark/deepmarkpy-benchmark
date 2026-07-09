import logging
import os
import sys
import tempfile
from typing import List
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from clearvoice import ClearVoice
from deepmarkpy.utils.utils import load_config, resample_audio
import soundfile as sf

logger = logging.getLogger(__name__)
app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int
    model_name: str


@app.post("/attack")
async def attack(request: AttackRequest):
    sampling_rate = request.sampling_rate
    audio = np.array(request.audio)
    
    # Create a temporary file with .wav extension (not .mp3)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Add noise before saving
        noise_strength = config.get("noise_strength", 0.0)
        if noise_strength > 0:
            noisy = audio + noise_strength * np.random.normal(0, 1, size=(len(audio)))
            audio = noisy

        # Resample to 16kHz for the model
        target_sr = 16000
        if sampling_rate != target_sr:
            audio = resample_audio(audio, sampling_rate, target_sr)

        # Save the audio array to the temporary WAV file

        sf.write(tmp_path, audio, target_sr)
        
        # Pass the temporary file path to ClearVoice
        logger.info(f"Processing with ClearVoice model: {request.model_name}")
        myClearVoice = ClearVoice(task='speech_enhancement', model_names=[request.model_name])
        audio_cv = myClearVoice(input_path=tmp_path, online_write=False)
        

        # ClearVoice returns a dict {filename: audio_array} when online_write=False
        if isinstance(audio_cv, dict):
            audio_cv = list(audio_cv.values())[0]
        audio_cv = np.array(audio_cv).squeeze()
        if sampling_rate != target_sr:
            audio_cv = resample_audio(audio_cv, target_sr, sampling_rate)
        return {"audio": audio_cv.tolist()}
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        return {"error": str(e), "audio": None}
    
    finally:
        # Delete the temporary file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.info(f"Deleted temporary file: {tmp_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {tmp_path}: {str(e)}")
