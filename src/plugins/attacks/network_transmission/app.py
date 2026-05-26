"""
Opus Network Emulation Attack - FastAPI Server

Simulates realistic VoIP/WebRTC network conditions using:
- WebRTC Audio Processing Module for noise suppression and VAD
- tc netem for network impairments (delay, jitter, packet loss)
- Opus encoding/decoding with real packet loss concealment (PLC)

Requires container to run with --cap-add=NET_ADMIN for tc commands.
"""

import logging
import os
import random
import socket
import subprocess
import tempfile
import time
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from webrtc_audio_processing import AudioProcessingModule as AP

# Opus codec only supports these internal rates. opusenc silently remaps
# anything else (e.g. 22050 -> 24000, 44100 -> 48000), so we resample
# explicitly to keep the pipeline honest.
OPUS_SUPPORTED_RATES = (8000, 12000, 16000, 24000, 48000)
# WebRTC APM is locked to these rates (NOT 12k/24k/44.1k). We pick the
# nearest supported rate to the input signal so high-fidelity inputs
# (44.1k, 48k) don't lose the upper spectrum to forced 16 kHz downsampling.
WEBRTC_APM_SUPPORTED_RATES = (8000, 16000, 32000, 48000)


def _nearest_rate(rate: int, supported: tuple) -> int:
    """Return the supported rate closest to ``rate`` (ties favor higher)."""
    return min(supported, key=lambda r: (abs(r - rate), -r))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int
    bitrate: int = 16
    framesize: float = 20
    delay_ms: int = 50
    jitter_ms: int = 20
    packet_loss: int = 5
    

class NetworkEmulator:
    """Manages tc netem rules for network emulation."""

    def __init__(self):
        self.interface = "lo"
        self.current_handle = None

    def setup(self, delay_ms: int, jitter_ms: int, packet_loss: int):
        """Configure tc netem with specified parameters."""
        self.cleanup()

        try:
            cmd = [
                "tc", "qdisc", "add", "dev", self.interface, "root",
                "netem",
                "delay", f"{delay_ms}ms", f"{jitter_ms}ms", "distribution", "normal",
                "loss", f"{packet_loss}%"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"tc setup failed (may need NET_ADMIN): {result.stderr}")
                return False

            logger.info(f"Network emulation configured: delay={delay_ms}ms, jitter={jitter_ms}ms, loss={packet_loss}%")
            self.current_handle = True
            return True

        except Exception as e:
            logger.warning(f"Failed to setup network emulation: {e}")
            return False

    def cleanup(self):
        """Remove tc netem rules."""
        try:
            subprocess.run(
                ["tc", "qdisc", "del", "dev", self.interface, "root"],
                capture_output=True,
                text=True
            )
        except Exception:
            pass
        self.current_handle = None



class OpusPacketSimulator:
    """
    Simulates Opus packet transmission over an impaired network.

    Feature:
    - UDP packet transmission with tc netem
    """

    def __init__(self, port_base: int = 30000):
        self.port_base = port_base
        self.packet_size = 1500

    def simulate_transmission(
        self,
        opus_data: bytes,
    ) -> list:
        """
        Simulate sending Opus packets through impaired network.

        Returns:
            List of received packets (None for lost packets)
        """
        port = self.port_base + (os.getpid() % 1000)

        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        receiver.bind(('127.0.0.1', port))
        receiver.settimeout(0.5)

        received_chunks = []

        try:
            # Split data into chunks
            chunk_size = 960  # ~20ms at 48kHz
            chunks = [opus_data[i:i+chunk_size] for i in range(0, len(opus_data), chunk_size)]

            send_times = {}

            # Send all chunks
            for i, chunk in enumerate(chunks):
                header = i.to_bytes(4, 'big')
                packet = header + chunk
                send_times[i] = time.time()
                sender.sendto(packet, ('127.0.0.1', port))
                time.sleep(0.001)

            # Receive packets
            received = {}
            receive_times = {}
            start_time = time.time()
            max_wait = len(chunks) * 0.1 + 2.0

            while time.time() - start_time < max_wait:
                try:
                    data, _ = receiver.recvfrom(2048)
                    recv_time = time.time()
                    if len(data) > 4:
                        seq = int.from_bytes(data[:4], 'big')
                        received[seq] = data[4:]
                        receive_times[seq] = recv_time
                except socket.timeout:
                    if len(received) >= len(chunks) * 0.8:
                        break
                    continue

            # Calculate arrival delays
            for i in range(len(chunks)):
                if i in received:
                    received_chunks.append(received[i])
                else:
                    received_chunks.append(None)

        finally:
            sender.close()
            receiver.close()

        return received_chunks


def encode_with_opus(
    input_wav: str,
    output_opus: str,
    bitrate: int,
    framesize: int,
) -> None:
    """
    Encode audio with optional bandwidth collapse simulation.

    Bandwidth collapse occurs when network congestion forces
    the codec to reduce bitrate mid-stream, causing quality degradation.

    Args:
        input_wav: Input WAV file path
        output_opus: Output Opus file path
        bitrate: Target bitrate in kbps
        framesize: Frame size in ms
    """


    framesize_str = str(int(framesize)) if framesize >= 5 else "2.5"
    cmd = [
        "opusenc",
        "--bitrate", str(int(bitrate)),
        "--framesize", framesize_str,
        "--quiet",
        input_wav,
        output_opus
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"opusenc failed: {result.stderr}")
    return


def process_with_opus_and_network(
    audio: np.ndarray,
    sampling_rate: int,
    bitrate: int,
    framesize: float,
    delay_ms: int,
    jitter_ms: int,
    packet_loss: int,
) -> np.ndarray:
    """
    Process audio through Opus codec with full network emulation.

    Simulates:
    1. Opus encoding 
    2. Network delay and jitter via tc netem
    5. Opus decoding with PLC for lost packets
    """

    network = NetworkEmulator()
    simulator = OpusPacketSimulator()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
        input_wav = f_in.name
    with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f_opus:
        opus_file = f_opus.name
    with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f_opus_out:
        opus_file_out = f_opus_out.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        output_wav = f_out.name

    try:
        sf.write(input_wav, audio, sampling_rate)

        encode_with_opus(input_wav, opus_file, bitrate, framesize)

        # Setup network emulation
        network_enabled = network.setup(delay_ms, jitter_ms, packet_loss)

        if network_enabled:
            with open(opus_file, 'rb') as f:
                opus_data = f.read()

            # Simulate network transmission with all impairments
            received_chunks = simulator.simulate_transmission(opus_data)

            reconstructed = b''.join(c for c in received_chunks if c is not None)
            loss_ratio = sum(1 for c in received_chunks if c is None) / max(len(received_chunks), 1)

            if loss_ratio > 0.5:
                logger.info(f"High packet loss ({loss_ratio*100:.1f}%), using opusdec PLC")
                opus_file_out = opus_file
                effective_loss = int(loss_ratio * 100)
            else:
                with open(opus_file_out, 'wb') as f:
                    logger.info(f"Successfully reconstructed {len(reconstructed)} bytes after network emulation")
                    f.write(reconstructed) if reconstructed else f.write(opus_data)
                effective_loss = 0

            network.cleanup()
        else:
            logger.info("Network emulation unavailable, using opusdec packet-loss simulation")
            opus_file_out = opus_file
            effective_loss = packet_loss

        # Decode with PLC (--rate ensures output matches input sample rate)
        decode_cmd = [
            "opusdec",
            "--rate", str(sampling_rate),
            "--packet-loss", str(effective_loss),
            "--quiet",
            opus_file_out,
            output_wav
        ]
        result = subprocess.run(decode_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            decode_cmd = ["opusdec", "--rate", str(sampling_rate), "--quiet", opus_file, output_wav]
            result = subprocess.run(decode_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"opusdec failed: {result.stderr}")

        decoded_audio, sr = sf.read(output_wav)

        if len(decoded_audio) > len(audio):
            decoded_audio = decoded_audio[:len(audio)]
        elif len(decoded_audio) < len(audio):
            decoded_audio = np.pad(decoded_audio, (0, len(audio) - len(decoded_audio)))

        logger.info(f"Opus network attack complete: bitrate={bitrate}k, delay={delay_ms}ms, "
                   f"jitter={jitter_ms}ms, loss={packet_loss}, attacked audio sampling rate={sr}Hz ")

        return decoded_audio.astype(np.float32)

    finally:
        for f in [input_wav, opus_file, opus_file_out, output_wav]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        network.cleanup()


def audio_preprocessing(audio: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Apply WebRTC Audio Processing.

    WebRTC APM only accepts 8/16/32/48 kHz with 10ms frames. We resample
    to the *nearest* supported rate (so 44.1k stays at 48k, 22050 goes
    to 16k, etc.) so high-fidelity inputs don't lose their upper spectrum
    to a forced 16 kHz downsample. Output is resampled back to the
    caller's SR.
    """
    apm_rate = _nearest_rate(sampling_rate, WEBRTC_APM_SUPPORTED_RATES)
    needs_resample = sampling_rate != apm_rate
    if needs_resample:
        apm_audio = librosa.resample(
            audio, orig_sr=sampling_rate, target_sr=apm_rate
        )
    else:
        apm_audio = audio

    ap = AP(enable_vad=True, enable_ns=True)
    ap.set_stream_format(apm_rate, 1)
    ap.set_ns_level(1)
    ap.set_vad_level(1)

    # Clip first so out-of-range samples don't wrap on int16 cast.
    apm_audio = np.clip(apm_audio, -1.0, 1.0)
    audio_int16 = (apm_audio * 32767).astype(np.int16)

    frame_size = apm_rate // 100  # 10ms frame at the chosen APM rate
    processed_frames = []

    n_full = (len(audio_int16) // frame_size) * frame_size
    for i in range(0, n_full, frame_size):
        frame = audio_int16[i:i + frame_size]
        out_bytes = ap.process_stream(frame.tobytes())
        processed_frames.append(np.frombuffer(out_bytes, dtype=np.int16))

    # Preserve any trailing partial frame so length is conserved through
    # the rest of the pipeline (raw, since APM can't process it).
    tail = audio_int16[n_full:]
    if len(tail) > 0:
        processed_frames.append(tail)

    out = np.concatenate(processed_frames).astype(np.float32) / 32767.0
    if needs_resample:
        out = librosa.resample(
            out, orig_sr=apm_rate, target_sr=sampling_rate
        )
        # librosa.resample length can drift by ±1; lock to caller's length
        # so the downstream pad/truncate doesn't accumulate further drift.
        if len(out) > len(audio):
            out = out[:len(audio)]
        elif len(out) < len(audio):
            out = np.pad(out, (0, len(audio) - len(out)))
    return out


@app.post("/attack")
async def attack(request: AttackRequest):
    """Process audio through Opus codec with full network emulation."""
    try:
        audio = np.array(request.audio, dtype=np.float32)
        original_sr = request.sampling_rate
        original_len = len(audio)

        audio = audio_preprocessing(audio, original_sr)

        # Opus only supports a fixed set of internal rates. Resample to
        # the nearest supported rate for the codec stage, then back to
        # the caller's SR so detect() sees the audio at the same rate it
        # provided. This keeps the rest of the benchmark pipeline
        # consistent regardless of model.
        codec_sr = _nearest_rate(original_sr, OPUS_SUPPORTED_RATES)
        if codec_sr != original_sr:
            codec_audio = librosa.resample(
                audio, orig_sr=original_sr, target_sr=codec_sr
            )
            logger.info(
                f"Resampling {original_sr} Hz -> {codec_sr} Hz for Opus codec"
            )
        else:
            codec_audio = audio

        codec_out = process_with_opus_and_network(
            audio=codec_audio,
            sampling_rate=codec_sr,
            bitrate=request.bitrate,
            framesize=request.framesize,
            delay_ms=request.delay_ms,
            jitter_ms=request.jitter_ms,
            packet_loss=request.packet_loss,
        )

        if codec_sr != original_sr:
            result = librosa.resample(
                codec_out, orig_sr=codec_sr, target_sr=original_sr
            )
        else:
            result = codec_out

        if len(result) > original_len:
            result = result[:original_len]
        elif len(result) < original_len:
            result = np.pad(result, (0, original_len - len(result)))

        return {"audio": result.astype(np.float32).tolist()}

    except Exception as e:
        logger.error(f"Attack failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "opus_network"}
