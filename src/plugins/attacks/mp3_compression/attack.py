import subprocess
import time
import os
import numpy as np
import soundfile as sf
import tempfile

from core.base_attack import BaseAttack

class Mp3CompressionAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform mp3 compression attack on an audio signal. 
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the mp3 compression:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - quality_mp3: MP3 quality (0=best, 9=worst) 
        Returns:
            np.ndarray: The processed mp3 signal.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """
        sampling_rate = kwargs.get("sampling_rate", None)
        quality = kwargs.get(
            "quality_mp3", self.config.get("quality_mp3")
        )

        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("FFmpeg not found. Please install FFmpeg or skip MP3 tests.")
        
        temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
        temp_mp3_fd, mp3_path = tempfile.mkstemp(suffix='.mp3')

        try:
        # Close file descriptors immediately to avoid conflicts
            os.close(temp_wav_fd)
            os.close(temp_mp3_fd)
            
            # Apply PCM conversion and save
            audio = self.pcm_bit_depth_conversion(audio, sampling_rate, 16)
            sf.write(temp_wav_path, audio, sampling_rate)
            
            # Convert to MP3 using ffmpeg
            _ = subprocess.run(['ffmpeg', '-i', temp_wav_path, '-q:a', str(quality), mp3_path, '-y'], 
                                capture_output=True, check=True)
            
            # Small delay to ensure FFmpeg fully releases files
            time.sleep(0.1)
            
            # Load the MP3 file
            audio_data, _ = sf.read(mp3_path)
            
            # Another small delay before cleanup
            time.sleep(0.1)
        
            return audio_data
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
        except Exception as e:
            raise e
        finally:
            # Clean up temporary files with retry logic
            self.safe_delete(temp_wav_path)
            self.safe_delete(mp3_path)


    def safe_delete(self, filepath: str, max_retries: int = 5) -> None:
        """Safely delete a file with retries for Windows file locking issues"""
        for attempt in range(max_retries):
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Wait 100ms before retry
                else:
                    print(f"Warning: Could not delete {filepath} after {max_retries} attempts")


    def pcm_bit_depth_conversion(self, audio: np.ndarray, sr: int, pcm: int = 16) -> np.ndarray:
        """
        Simulate MP3 compression with PCM bit depth conversion
        Args:
            audio: Input audio (float32, range -1 to 1)
            sr: Sample rate
            pcm: PCM bit depth (8, 16, 24)
            quality: MP3 quality (0=best, 9=worst)
        """
        # Convert to specified PCM bit depth and back (simulates quantization)
        if pcm == 8:
            # 8-bit signed: -128 to 127
            audio_int = np.clip(audio * 127.0, -128, 127).astype(np.int8)
            audio = audio_int.astype(np.float32) / 127.0
        elif pcm == 16:   
            # 16-bit signed: -32768 to 32767
            audio_int = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
            audio = audio_int.astype(np.float32) / 32767.0
        elif pcm == 24:
            # 24-bit signed: -8388608 to 8388607
            audio_int = np.clip(audio * 8388607.0, -8388608, 8388607).astype(np.int32)
            audio = audio_int.astype(np.float32) / 8388607.0
        else:
            raise ValueError(f"Unsupported PCM bit depth: {pcm}")
        return audio