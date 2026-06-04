"""
RTP VoIP Network Transmission Attack v3 - FastAPI Server

Combines:
  - v1's library-backed audio processing (WebRTC APM for NS+VAD,
    pyloudnorm for AGC) instead of v2's custom NLMS / spectral-gating /
    RMS implementations.
  - v2's realistic VoIP pipeline (RTP framing, drift-free pacing,
    deadline-based jitter buffer, FEC overwrites PLC, opuslib for
    in-band FEC).

The pipeline is locked to 16 kHz throughout. Resampling happens once on
entry (input_sr -> 16k) and once on exit (16k -> input_sr). 16 kHz is
WebRTC's wideband VoIP rate, supported by both APM and Opus, and matches
what real wideband VoIP clients (Skype, WhatsApp, Discord) actually use.

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
from webrtc_audio_processing import AudioProcessingModule as AP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Pipeline runs entirely at 16 kHz. WebRTC's wideband VoIP rate, an Opus
# native rate, and APM's most common processing rate -- everything plays
# nice without any nearest-rate dance.
PIPELINE_SR = 16000


# ---------------------------------------------------------------------------
# RTP header (RFC 3550)
# ---------------------------------------------------------------------------

def _build_rtp_header(seq: int, timestamp: int, ssrc: int) -> bytes:
    V, P, X, CC = 2, 0, 0, 0
    M = 0
    PT = 111  # dynamic payload type for Opus, per WebRTC convention
    first_byte = (V << 6) | (P << 5) | (X << 4) | CC
    second_byte = (M << 7) | PT
    return struct.pack(
        "!BBHII", first_byte, second_byte, seq & 0xFFFF,
        timestamp & 0xFFFFFFFF, ssrc,
    )


def _parse_rtp_header(data: bytes):
    if len(data) < 12:
        return None, None
    _, _, seq, _, _ = struct.unpack("!BBHII", data[:12])
    return seq, data[12:]


# ---------------------------------------------------------------------------
# Audio processing -- WebRTC APM for NS+VAD, pyloudnorm for AGC
# ---------------------------------------------------------------------------

def _webrtc_apm_process(
    audio: np.ndarray, fs: int, ns_enabled: bool, vad_enabled: bool,
) -> np.ndarray:
    """Run audio through WebRTC APM at the pipeline rate.

    APM accepts 8/16/32/48 kHz and 10ms frames. Caller must provide audio
    already at PIPELINE_SR (16 kHz), so the internal frame size is fixed
    at 160 samples.
    """
    if not (ns_enabled or vad_enabled):
        return audio  # APM has nothing to do

    ap = AP(enable_vad=vad_enabled, enable_ns=ns_enabled)
    ap.set_stream_format(fs, 1)
    if ns_enabled:
        ap.set_ns_level(1)
    if vad_enabled:
        ap.set_vad_level(1)

    # APM works on int16 PCM. Clip first so out-of-range samples don't
    # wrap on cast.
    apm_audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (apm_audio * 32767).astype(np.int16)

    frame_size = fs // 100  # 10ms => 160 samples at 16 kHz
    processed_frames = []
    n_full = (len(audio_int16) // frame_size) * frame_size
    for i in range(0, n_full, frame_size):
        frame = audio_int16[i:i + frame_size]
        out_bytes = ap.process_stream(frame.tobytes())
        processed_frames.append(np.frombuffer(out_bytes, dtype=np.int16))

    # Preserve trailing partial frame raw so length is conserved.
    tail = audio_int16[n_full:]
    if len(tail) > 0:
        processed_frames.append(tail)

    return np.concatenate(processed_frames).astype(np.float32) / 32767.0


def _agc_pyloudnorm(
    audio: np.ndarray, fs: int, target_lufs: float,
) -> np.ndarray:
    """LUFS-based AGC using pyloudnorm's BS.1770-compliant meter.

    Replaces v2's RMS-only approximation with the broadcast-standard
    integrated loudness measurement used in WebRTC and EBU R128.
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        logger.warning("pyloudnorm not installed, skipping AGC")
        return audio

    # BS.1770 needs at least ~0.4s for the gating window. Below that,
    # the meter returns -inf; fall back to a no-op.
    if len(audio) < int(fs * 0.4):
        return audio

    meter = pyln.Meter(fs)  # uses 400ms gating window per BS.1770
    try:
        current = meter.integrated_loudness(audio)
    except Exception as e:
        logger.warning(f"LUFS measurement failed: {e}")
        return audio

    if not np.isfinite(current):  # silent audio
        return audio

    return pyln.normalize.loudness(audio, current, target_lufs)


# ---------------------------------------------------------------------------
# Network emulation (tc netem)
# ---------------------------------------------------------------------------

class NetworkEmulator:
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
# RTP sender / receiver threads (drift-free pacing, arrival timestamps)
# ---------------------------------------------------------------------------

def _rtp_sender_thread(
    packets: List[bytes], sock: socket.socket, dest: tuple,
    frame_duration: float, stop_event: threading.Event,
):
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
# Jitter buffer (deadline-based)
# ---------------------------------------------------------------------------

def _jitter_buffer_playout(
    received_with_arrival: List[tuple], total_frames: int,
    frame_duration_s: float, playout_delay_s: float,
) -> List[Optional[bytes]]:
    by_seq = {}
    for arrival_time, raw in received_with_arrival:
        seq, payload = _parse_rtp_header(raw)
        if seq is None or not payload:
            continue
        if seq not in by_seq or arrival_time < by_seq[seq][0]:
            by_seq[seq] = (arrival_time, payload)

    if not by_seq:
        return [None] * total_frames

    first_arrival = min(t for t, _ in by_seq.values())
    output: List[Optional[bytes]] = []
    late_count = 0
    for expected_seq in range(total_frames):
        deadline = first_arrival + playout_delay_s + expected_seq * frame_duration_s
        entry = by_seq.get(expected_seq % 65536)
        if entry is None:
            output.append(None)
        else:
            arrival, payload = entry
            if arrival <= deadline:
                output.append(payload)
            else:
                output.append(None)
                late_count += 1
    if late_count:
        logger.info(f"Jitter buffer dropped {late_count} late packets")
    return output


# ---------------------------------------------------------------------------
# Opus encode / decode (FEC overwrites PLC; two decoders for state isolation)
# ---------------------------------------------------------------------------

def _opus_encode_frames(
    audio: np.ndarray, fs: int, frame_size: int,
    bitrate: int, fec_enabled: bool, expected_loss: int,
) -> List[bytes]:
    import opuslib
    import opuslib.api.encoder as _enc_api
    import opuslib.api.ctl as _ctl

    encoder = opuslib.Encoder(fs, 1, opuslib.APPLICATION_VOIP)
    encoder.bitrate = bitrate
    if fec_enabled:
        # opuslib's property setters for inband_fec / packet_loss_perc are
        # broken in the current version (missing positional arg in
        # ctl_set lambda). Use the low-level encoder_ctl API directly.
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
    """Decode Opus frames; FEC payload in packet N+1 overwrites PLC for N."""
    import opuslib

    main_decoder = opuslib.Decoder(fs, 1)
    fec_decoder = opuslib.Decoder(fs, 1)
    n = len(opus_frames)
    pcm_chunks: List[Optional[np.ndarray]] = [None] * n

    for i, frame in enumerate(opus_frames):
        if frame is not None:
            if i > 0 and opus_frames[i - 1] is None and fec_enabled:
                try:
                    fec_pcm = fec_decoder.decode(frame, frame_size, decode_fec=True)
                    pcm_chunks[i - 1] = np.frombuffer(fec_pcm, dtype=np.int16)
                except opuslib.OpusError as e:
                    logger.debug(f"FEC recovery failed at seq {i-1}: {e}")
            pcm = main_decoder.decode(frame, frame_size, decode_fec=False)
            pcm_chunks[i] = np.frombuffer(pcm, dtype=np.int16)
            try:
                fec_decoder.decode(frame, frame_size, decode_fec=False)
            except opuslib.OpusError:
                pass
        else:
            pcm = main_decoder.decode(b"", frame_size, decode_fec=False)
            pcm_chunks[i] = np.frombuffer(pcm, dtype=np.int16)
            try:
                fec_decoder.decode(b"", frame_size, decode_fec=False)
            except opuslib.OpusError:
                pass

    if not any(c is not None for c in pcm_chunks):
        return np.zeros(0, dtype=np.float32)
    silence = np.zeros(frame_size, dtype=np.int16)
    safe = [c if c is not None else silence for c in pcm_chunks]
    return np.concatenate(safe).astype(np.float32) / 32767.0


# ---------------------------------------------------------------------------
# Full VoIP pipeline (locked to 16 kHz internally)
# ---------------------------------------------------------------------------

def voip_pipeline(
    audio: np.ndarray,
    fs: int,
    bitrate_bps: int,
    frame_duration_ms: int,
    delay_ms: int,
    jitter_ms: int,
    packet_loss: int,
    duplication: int,
    reorder: int,
    corruption: int,
    fec_enabled: bool,
    expected_loss: int,
    ns_enabled: bool,
    vad_enabled: bool,
    agc_enabled: bool,
    agc_target_lufs: float,
    playout_delay_ms: int,
) -> np.ndarray:
    original_len = len(audio)
    original_sr = fs

    # Single resample on entry: input_sr -> 16k.
    if original_sr != PIPELINE_SR:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=PIPELINE_SR)
        logger.info(f"Resampled {original_sr} Hz -> {PIPELINE_SR} Hz")

    frame_size = PIPELINE_SR * frame_duration_ms // 1000  # 320 samples for 20ms

    # --- Sender APM (NS + VAD via WebRTC APM) ---
    audio = _webrtc_apm_process(audio, PIPELINE_SR, ns_enabled, vad_enabled)

    # --- Opus encode ---
    opus_frames = _opus_encode_frames(
        audio, PIPELINE_SR, frame_size, bitrate_bps, fec_enabled, expected_loss
    )
    total_frames = len(opus_frames)
    if total_frames == 0:
        logger.warning("No frames encoded -- audio too short")
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
    network.setup(delay_ms, jitter_ms, packet_loss, duplication, reorder, corruption)

    # --- RTP transmission with sender / receiver threads ---
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

    # --- Jitter buffer (deadline-based) ---
    playout_frames = _jitter_buffer_playout(
        received_packets, total_frames,
        frame_duration_s=frame_duration_ms / 1000.0,
        playout_delay_s=playout_delay_ms / 1000.0,
    )

    # --- Opus decode (FEC overwrites PLC) ---
    decoded = _opus_decode_frames(playout_frames, PIPELINE_SR, frame_size, fec_enabled)

    # --- Receiver-side AGC (pyloudnorm BS.1770) ---
    if agc_enabled:
        decoded = _agc_pyloudnorm(decoded, PIPELINE_SR, agc_target_lufs)

    # Single resample on exit: 16k -> input_sr.
    if original_sr != PIPELINE_SR:
        decoded = librosa.resample(decoded, orig_sr=PIPELINE_SR, target_sr=original_sr)

    # Lock to caller's length.
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
    bitrate_bps_netem3: int = 24000
    frame_duration_ms_netem3: int = 20
    delay_ms_netem3: int = 100
    jitter_ms_netem3: int = 30
    packet_loss_netem3: int = 10
    duplication_netem3: int = 0
    reorder_netem3: int = 3
    corruption_netem3: int = 0
    fec_enabled_netem3: bool = True
    expected_loss_netem3: int = 10
    ns_enabled_netem3: bool = True
    vad_enabled_netem3: bool = True
    agc_enabled_netem3: bool = True
    agc_target_lufs_netem3: float = -18
    playout_delay_ms_netem3: int = 60


@app.post("/attack")
async def attack(request: AttackRequest):
    try:
        audio = np.array(request.audio, dtype=np.float32)
        result = voip_pipeline(
            audio=audio,
            fs=request.sampling_rate,
            bitrate_bps=request.bitrate_bps_netem3,
            frame_duration_ms=request.frame_duration_ms_netem3,
            delay_ms=request.delay_ms_netem3,
            jitter_ms=request.jitter_ms_netem3,
            packet_loss=request.packet_loss_netem3,
            duplication=request.duplication_netem3,
            reorder=request.reorder_netem3,
            corruption=request.corruption_netem3,
            fec_enabled=request.fec_enabled_netem3,
            expected_loss=request.expected_loss_netem3,
            ns_enabled=request.ns_enabled_netem3,
            vad_enabled=request.vad_enabled_netem3,
            agc_enabled=request.agc_enabled_netem3,
            agc_target_lufs=request.agc_target_lufs_netem3,
            playout_delay_ms=request.playout_delay_ms_netem3,
        )
        return {"audio": result.tolist()}
    except Exception as e:
        logger.error(f"Attack failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "network_transmission_3"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "10022"))
    uvicorn.run(app, host=host, port=port)
