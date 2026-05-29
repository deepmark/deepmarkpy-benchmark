"""
RTP VoIP Network Transmission Attack - FastAPI Server

Realistic VoIP call simulation with proper RTP framing, real-time
pacing, jitter buffer, Opus FEC/PLC, and optional audio processing.

Pipeline:
  Sender: Audio → AEC → Denoise → AGC → Opus Encode → RTP packets
  Network: tc netem (delay, jitter, loss, reorder, duplication, corruption)
  Receiver: RTP → Jitter Buffer → Opus Decode (FEC+PLC) → Denoise → AGC

Requires container to run with --cap-add=NET_ADMIN for tc commands.
"""

import logging
import os
import random
import socket
import struct
import subprocess
import threading
import time
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

OPUS_SUPPORTED_RATES = (8000, 12000, 16000, 24000, 48000)


def _nearest_opus_rate(rate: int) -> int:
    return min(OPUS_SUPPORTED_RATES, key=lambda r: (abs(r - rate), -r))


# ---------------------------------------------------------------------------
# RTP header (RFC 3550)
# ---------------------------------------------------------------------------

def _build_rtp_header(seq: int, timestamp: int, ssrc: int) -> bytes:
    """Build a minimal 12-byte RTP header (V=2, PT=111 dynamic Opus)."""
    V, P, X, CC = 2, 0, 0, 0
    M = 0
    PT = 111
    first_byte = (V << 6) | (P << 5) | (X << 4) | CC
    second_byte = (M << 7) | PT
    return struct.pack(
        "!BBHII", first_byte, second_byte, seq & 0xFFFF,
        timestamp & 0xFFFFFFFF, ssrc,
    )


def _parse_rtp_header(data: bytes):
    """Extract seq and payload from an RTP packet."""
    if len(data) < 12:
        return None, None
    _, _, seq, _, _ = struct.unpack("!BBHII", data[:12])
    return seq, data[12:]


# ---------------------------------------------------------------------------
# Sender-side audio processing
# ---------------------------------------------------------------------------

def _measure_lufs(audio: np.ndarray, fs: int) -> float:
    """Simplified loudness measurement (RMS-based approximation)."""
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    return 20 * np.log10(rms + 1e-10)


def _agc(audio: np.ndarray, target_lufs: float, fs: int) -> np.ndarray:
    """Simple AGC: normalize to target loudness level."""
    current = _measure_lufs(audio, fs)
    gain_db = target_lufs - current
    gain_db = np.clip(gain_db, -20, 20)
    audio = audio * (10 ** (gain_db / 20))
    return np.clip(audio, -1.0, 1.0)


def _aec_nlms(mic_signal: np.ndarray, filter_len: int = 512, mu: float = 0.01) -> np.ndarray:
    """NLMS echo cancellation (uses same signal as reference — simulation)."""
    N = len(mic_signal)
    if N < filter_len:
        return mic_signal
    ref_signal = np.roll(mic_signal, filter_len // 2)
    w = np.zeros(filter_len)
    e = np.zeros(N)
    for n in range(filter_len, N):
        x_vec = ref_signal[n - filter_len:n][::-1]
        y_hat = np.dot(w, x_vec)
        e[n] = mic_signal[n] - y_hat
        norm = np.dot(x_vec, x_vec) + 1e-8
        w += (2 * mu * e[n] / norm) * x_vec
    e[:filter_len] = mic_signal[:filter_len]
    return e.astype(np.float32)


def _denoise_noisereduce(audio: np.ndarray, fs: int) -> np.ndarray:
    """Apply spectral-gating noise reduction."""
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=audio, sr=fs, prop_decrease=0.6)
    except ImportError:
        logger.warning("noisereduce not installed, skipping denoise")
        return audio


def _sender_process(
    audio: np.ndarray, fs: int, aec_enabled: bool,
    denoise_method: str, agc_target: float,
) -> np.ndarray:
    """Sender-side processing: AEC → Denoise → AGC."""
    out = audio.copy()
    if aec_enabled:
        out = _aec_nlms(out)
    if denoise_method == "noisereduce":
        out = _denoise_noisereduce(out, fs)
    if agc_target is not None:
        out = _agc(out, agc_target, fs)
    return out


# ---------------------------------------------------------------------------
# Receiver-side audio processing
# ---------------------------------------------------------------------------

def _receiver_process(
    audio: np.ndarray, fs: int, denoise_method: str, agc_target: float,
) -> np.ndarray:
    """Receiver-side processing: Denoise → AGC."""
    out = audio.copy()
    if denoise_method == "noisereduce":
        out = _denoise_noisereduce(out, fs)
    if agc_target is not None:
        out = _agc(out, agc_target, fs)
    return out


# ---------------------------------------------------------------------------
# Network emulation (tc netem)
# ---------------------------------------------------------------------------

class NetworkEmulator:
    """Manages tc netem rules on loopback interface."""

    def __init__(self):
        self.interface = "lo"

    def setup(
        self, delay_ms: int, jitter_ms: int, packet_loss: int,
        duplication: int = 0, reorder: int = 0, corruption: int = 0,
    ) -> bool:
        self.cleanup()
        try:
            cmd = [
                "tc", "qdisc", "add", "dev", self.interface, "root", "netem",
                "delay", f"{delay_ms}ms", f"{jitter_ms}ms", "distribution", "normal",
                "loss", f"{packet_loss}%",
            ]
            if duplication > 0:
                cmd += ["duplicate", f"{duplication}%"]
            if reorder > 0:
                cmd += ["reorder", f"{reorder}%"]
            if corruption > 0:
                cmd += ["corrupt", f"{corruption}%"]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"tc setup failed: {result.stderr}")
                return False

            logger.info(
                f"netem configured: delay={delay_ms}ms±{jitter_ms}ms, "
                f"loss={packet_loss}%, dup={duplication}%, reorder={reorder}%, "
                f"corrupt={corruption}%"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to setup netem: {e}")
            return False

    def cleanup(self):
        try:
            subprocess.run(
                ["tc", "qdisc", "del", "dev", self.interface, "root"],
                capture_output=True, text=True,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# RTP sender/receiver threads
# ---------------------------------------------------------------------------

def _rtp_sender_thread(
    packets: List[bytes], sock: socket.socket, dest: tuple,
    frame_duration: float, stop_event: threading.Event,
):
    """Send RTP packets with drift-free real-time pacing.

    Uses absolute target times (start + n*frame_duration) instead of cumulative
    sleeps so per-call sleep error doesn't accumulate over the whole stream.
    """
    start = time.monotonic()
    for seq, pkt in enumerate(packets):
        if stop_event.is_set():
            break
        sock.sendto(pkt, dest)
        target = start + (seq + 1) * frame_duration
        remaining = target - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


def _rtp_receiver_thread(
    sock: socket.socket, received: list,
    stop_event: threading.Event, timeout: float = 0.5,
):
    """Receive RTP packets and record arrival timestamps.

    The jitter buffer needs arrival times to enforce per-frame playout
    deadlines (late packets are discarded just like a real VoIP client).
    """
    sock.settimeout(timeout)
    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(4096)
            received.append((time.monotonic(), data))
        except socket.timeout:
            continue
        except OSError:
            break


# ---------------------------------------------------------------------------
# Jitter buffer
# ---------------------------------------------------------------------------

def _jitter_buffer_playout(
    received_with_arrival: List[tuple], total_frames: int,
    frame_duration_s: float, playout_delay_s: float,
) -> List[Optional[bytes]]:
    """Deadline-based jitter buffer playout.

    Each frame has a playout deadline = first_arrival + playout_delay + seq*frame_duration.
    Packets that arrive after their deadline are dropped, just like in a real
    VoIP client where late packets can't be played back.

    Args:
        received_with_arrival: list of (arrival_time, raw_packet) tuples
        total_frames: number of expected frames
        frame_duration_s: frame size in seconds (e.g. 0.020 for 20ms)
        playout_delay_s: jitter buffer depth in seconds (e.g. 0.060 for 60ms)
    """
    by_seq = {}
    for arrival_time, raw in received_with_arrival:
        seq, payload = _parse_rtp_header(raw)
        if seq is None or not payload:
            continue
        # Keep earliest arrival for duplicates (closest to "real" delivery time)
        if seq not in by_seq or arrival_time < by_seq[seq][0]:
            by_seq[seq] = (arrival_time, payload)

    if not by_seq:
        return [None] * total_frames

    # First arrival anchors the playout clock (first packet defines t=0 for playout)
    first_arrival = min(t for t, _ in by_seq.values())

    output: List[Optional[bytes]] = []
    late_count = 0
    for expected_seq in range(total_frames):
        deadline = first_arrival + playout_delay_s + expected_seq * frame_duration_s
        entry = by_seq.get(expected_seq % 65536)
        if entry is None:
            output.append(None)  # never arrived
        else:
            arrival, payload = entry
            if arrival <= deadline:
                output.append(payload)
            else:
                output.append(None)  # arrived too late, drop
                late_count += 1

    if late_count:
        logger.info(f"Jitter buffer dropped {late_count} late packets")
    return output


# ---------------------------------------------------------------------------
# Opus encode/decode with opuslib
# ---------------------------------------------------------------------------

def _opus_encode_frames(
    audio: np.ndarray, fs: int, frame_size: int,
    bitrate: int, fec_enabled: bool, expected_loss: int,
) -> List[bytes]:
    """Encode audio into Opus frames using opuslib."""
    import opuslib
    import opuslib.api.encoder as _enc_api
    import opuslib.api.ctl as _ctl

    encoder = opuslib.Encoder(fs, 1, opuslib.APPLICATION_VOIP)
    encoder.bitrate = bitrate
    if fec_enabled:
        # opuslib's property setters are buggy for inband_fec / packet_loss_perc
        # in this version (missing positional arg in ctl_set). Use the low-level
        # encoder_ctl API directly which works correctly.
        _enc_api.encoder_ctl(encoder.encoder_state, _ctl.set_inband_fec, 1)
        _enc_api.encoder_ctl(
            encoder.encoder_state, _ctl.set_packet_loss_perc, int(expected_loss)
        )

    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    frames = []
    for i in range(0, len(audio_int16) - frame_size + 1, frame_size):
        frame_data = audio_int16[i:i + frame_size].tobytes()
        encoded = encoder.encode(frame_data, frame_size)
        frames.append(encoded)
    return frames


def _opus_decode_frames(
    opus_frames: List[Optional[bytes]], fs: int, frame_size: int,
    fec_enabled: bool,
) -> np.ndarray:
    """Decode Opus frames with FEC overwriting PLC where possible.

    Pipeline (matches real VoIP behavior):
      1. Lost frame → emit PLC immediately so playout is uninterrupted.
      2. Next frame arrives → if FEC is enabled, run a side decoder once
         with decode_fec=True to recover the lost frame from the FEC payload
         carried in the next packet, and overwrite the placeholder PLC output.
      3. Then decode the next frame normally on the main decoder.

    Two decoders are used to avoid corrupting the main decoder's internal
    state with the FEC look-back call (the main decoder must stay aligned
    with the actual playout sequence).
    """
    import opuslib

    main_decoder = opuslib.Decoder(fs, 1)
    fec_decoder = opuslib.Decoder(fs, 1)

    n = len(opus_frames)
    pcm_chunks: List[Optional[np.ndarray]] = [None] * n

    for i, frame in enumerate(opus_frames):
        if frame is not None:
            # Try FEC recovery for the previous frame if it was lost
            if i > 0 and opus_frames[i - 1] is None and fec_enabled:
                try:
                    fec_pcm = fec_decoder.decode(frame, frame_size, decode_fec=True)
                    pcm_chunks[i - 1] = np.frombuffer(fec_pcm, dtype=np.int16)
                except opuslib.OpusError as e:
                    logger.debug(f"FEC recovery failed at seq {i-1}: {e}")
            # Normal decode of current frame on the main decoder
            pcm = main_decoder.decode(frame, frame_size, decode_fec=False)
            pcm_chunks[i] = np.frombuffer(pcm, dtype=np.int16)
            # Keep fec_decoder state aligned with main_decoder
            try:
                fec_decoder.decode(frame, frame_size, decode_fec=False)
            except opuslib.OpusError:
                pass
        else:
            # Emit PLC placeholder; may be overwritten by FEC on next iteration
            pcm = main_decoder.decode(b"", frame_size, decode_fec=False)
            pcm_chunks[i] = np.frombuffer(pcm, dtype=np.int16)
            try:
                fec_decoder.decode(b"", frame_size, decode_fec=False)
            except opuslib.OpusError:
                pass

    if not any(c is not None for c in pcm_chunks):
        return np.zeros(0, dtype=np.float32)

    # Replace any None with silence (shouldn't happen — every iteration writes)
    silence = np.zeros(frame_size, dtype=np.int16)
    safe_chunks = [c if c is not None else silence for c in pcm_chunks]
    return np.concatenate(safe_chunks).astype(np.float32) / 32767.0


# ---------------------------------------------------------------------------
# Full VoIP pipeline
# ---------------------------------------------------------------------------

def voip_pipeline(
    audio: np.ndarray,
    fs: int,
    bitrate_bps: int = 24000,
    frame_duration_ms: int = 20,
    delay_ms: int = 50,
    jitter_ms: int = 10,
    packet_loss: int = 5,
    duplication: int = 0,
    reorder: int = 0,
    corruption: int = 0,
    fec_enabled: bool = True,
    expected_loss: int = 5,
    aec_enabled: bool = False,
    denoise_method: str = "none",
    agc_target: Optional[float] = -18,
    playout_delay_ms: int = 60,
) -> np.ndarray:
    """
    Full VoIP pipeline:
    1. Sender processing (AEC, denoise, AGC)
    2. Opus encode
    3. RTP packetization + real-time send through tc netem
    4. Jitter buffer playout
    5. Opus decode with FEC/PLC
    6. Receiver processing (denoise, AGC)
    """
    original_len = len(audio)

    # Resample to nearest Opus-supported rate
    codec_sr = _nearest_opus_rate(fs)
    if codec_sr != fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=codec_sr)
        logger.info(f"Resampled {fs} Hz → {codec_sr} Hz for Opus")

    frame_size = int(codec_sr * frame_duration_ms / 1000)

    # --- Sender processing ---
    processed = _sender_process(audio, codec_sr, aec_enabled, denoise_method, agc_target)

    # --- Opus encode ---
    opus_frames = _opus_encode_frames(
        processed, codec_sr, frame_size, bitrate_bps, fec_enabled, expected_loss
    )
    total_frames = len(opus_frames)
    if total_frames == 0:
        logger.warning("No frames encoded — audio too short")
        return np.zeros(original_len, dtype=np.float32)

    # --- Build RTP packets ---
    ssrc = random.randint(1, 0xFFFFFFFF)
    rtp_packets = []
    for seq, opus_frame in enumerate(opus_frames):
        ts = seq * frame_size
        hdr = _build_rtp_header(seq, ts, ssrc)
        rtp_packets.append(hdr + opus_frame)

    # --- Network emulation ---
    network = NetworkEmulator()
    network_enabled = network.setup(
        delay_ms, jitter_ms, packet_loss, duplication, reorder, corruption
    )

    # --- RTP transmission with threading ---
    port = 30000 + (os.getpid() % 10000)
    sender_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver_sock.bind(("127.0.0.1", port))

    received_packets = []
    stop_event = threading.Event()

    recv_thread = threading.Thread(
        target=_rtp_receiver_thread,
        args=(receiver_sock, received_packets, stop_event),
    )
    recv_thread.start()

    send_thread = threading.Thread(
        target=_rtp_sender_thread,
        args=(rtp_packets, sender_sock, ("127.0.0.1", port),
              frame_duration_ms / 1000.0, stop_event),
    )
    send_thread.start()
    send_thread.join()

    # Wait for stragglers
    time.sleep(max(delay_ms * 3 / 1000.0, 0.5))
    stop_event.set()
    recv_thread.join(timeout=2.0)

    sender_sock.close()
    receiver_sock.close()
    network.cleanup()

    loss_count = total_frames - len(received_packets)
    logger.info(
        f"RTP transmission: sent={total_frames}, received={len(received_packets)}, "
        f"lost={loss_count} ({100*loss_count/max(total_frames,1):.1f}%)"
    )

    # --- Jitter buffer (deadline-based playout) ---
    playout_frames = _jitter_buffer_playout(
        received_packets, total_frames,
        frame_duration_s=frame_duration_ms / 1000.0,
        playout_delay_s=playout_delay_ms / 1000.0,
    )

    # --- Opus decode with FEC/PLC ---
    decoded = _opus_decode_frames(playout_frames, codec_sr, frame_size, fec_enabled)

    # --- Receiver processing ---
    if denoise_method != "none" or agc_target is not None:
        decoded = _receiver_process(decoded, codec_sr, denoise_method, agc_target)

    # Resample back to original rate
    if codec_sr != fs:
        decoded = librosa.resample(decoded, orig_sr=codec_sr, target_sr=fs)

    # Match original length
    if len(decoded) > original_len:
        decoded = decoded[:original_len]
    elif len(decoded) < original_len:
        decoded = np.pad(decoded, (0, original_len - len(decoded)))

    return decoded.astype(np.float32)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int
    bitrate_bps_netem2: int = 24000
    frame_duration_ms_netem2: int = 20
    delay_ms_netem2: int = 50
    jitter_ms_netem2: int = 10
    packet_loss_netem2: int = 5
    duplication_netem2: int = 0
    reorder_netem2: int = 0
    corruption_netem2: int = 0
    fec_enabled_netem2: bool = True
    expected_loss_netem2: int = 5
    aec_enabled_netem2: bool = False
    denoise_method_netem2: str = "none"
    agc_target_lufs_netem2: float = -18
    playout_delay_ms_netem2: int = 60


@app.post("/attack")
async def attack(request: AttackRequest):
    """Process audio through realistic VoIP pipeline."""
    try:
        audio = np.array(request.audio, dtype=np.float32)
        result = voip_pipeline(
            audio=audio,
            fs=request.sampling_rate,
            bitrate_bps=request.bitrate_bps_netem2,
            frame_duration_ms=request.frame_duration_ms_netem2,
            delay_ms=request.delay_ms_netem2,
            jitter_ms=request.jitter_ms_netem2,
            packet_loss=request.packet_loss_netem2,
            duplication=request.duplication_netem2,
            reorder=request.reorder_netem2,
            corruption=request.corruption_netem2,
            fec_enabled=request.fec_enabled_netem2,
            expected_loss=request.expected_loss_netem2,
            aec_enabled=request.aec_enabled_netem2,
            denoise_method=request.denoise_method_netem2,
            agc_target=request.agc_target_lufs_netem2,
            playout_delay_ms=request.playout_delay_ms_netem2,
        )
        return {"audio": result.tolist()}
    except Exception as e:
        logger.error(f"Attack failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "network_transmission_2"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "10021"))
    uvicorn.run(app, host=host, port=port)
