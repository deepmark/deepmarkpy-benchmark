"""
RTP VoIP Network Transmission Attack - FastAPI Server

Realistic VoIP simulation: WebRTC APM for NS+VAD, opuslib with in-band
FEC, RTP framing with drift-free pacing, deadline-based jitter buffer,
FEC overwrites PLC on decode, pyloudnorm AGC.

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
from pydantic import BaseModel, Field

from deepmarkpy.core.inference import MAX_AUDIO_SAMPLES
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
    audio_int16 = np.round(apm_audio * 32767).astype(np.int16)

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

    # Anchor the playout clock to when seq 0 *should* have arrived, derived
    # from the earliest (arrival - seq*frame_duration) across all received
    # packets. Using the raw earliest arrival would be wrong whenever seq 0
    # is lost or a reordered higher-seq packet arrives first -- then every
    # deadline would be shifted and borderline packets dropped spuriously.
    anchor = min(
        arrival - seq * frame_duration_s
        for seq, (arrival, _) in by_seq.items()
    )
    output: List[Optional[bytes]] = []
    late_count = 0
    for expected_seq in range(total_frames):
        deadline = anchor + playout_delay_s + expected_seq * frame_duration_s
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

    audio_int16 = np.round(np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
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
    """Decode Opus frames using the canonical single-decoder FEC/PLC pattern.

    A single stateful decoder is used (one decoder per stream, as in a real
    receiver). When a frame is lost and the *next* frame is available, Opus
    in-band FEC lets us recover the lost frame from the next packet:

        decode(next_pkt, decode_fec=True)   -> reconstructs the lost frame
        decode(next_pkt, decode_fec=False)  -> decodes the next frame

    Both calls run on the same decoder, in order, which keeps its internal
    state aligned with the playout sequence -- the two-decoder scheme used
    previously accumulated an extra decode per recovered loss and drifted.

    A lost frame whose successor is also missing (or when FEC is disabled)
    falls back to PLC via an empty-payload decode.
    """
    import opuslib

    decoder = opuslib.Decoder(fs, 1)
    n = len(opus_frames)
    pcm_chunks: List[np.ndarray] = []

    i = 0
    while i < n:
        frame = opus_frames[i]
        if frame is not None:
            # Normal decode of an available frame.
            pcm = decoder.decode(frame, frame_size, decode_fec=False)
            pcm_chunks.append(np.frombuffer(pcm, dtype=np.int16))
            i += 1
        elif (
            fec_enabled
            and i + 1 < n
            and opus_frames[i + 1] is not None
        ):
            # Frame i is lost but frame i+1 carries FEC for it. Recover i
            # from i+1, then decode i+1 normally -- both on the same decoder.
            nxt = opus_frames[i + 1]
            try:
                fec_pcm = decoder.decode(nxt, frame_size, decode_fec=True)
                pcm_chunks.append(np.frombuffer(fec_pcm, dtype=np.int16))
            except opuslib.OpusError as e:
                logger.debug(f"FEC recovery failed at seq {i}: {e}")
                plc = decoder.decode(b"", frame_size, decode_fec=False)
                pcm_chunks.append(np.frombuffer(plc, dtype=np.int16))
            nxt_pcm = decoder.decode(nxt, frame_size, decode_fec=False)
            pcm_chunks.append(np.frombuffer(nxt_pcm, dtype=np.int16))
            i += 2
        else:
            # Lost frame with no FEC available -> packet loss concealment.
            plc = decoder.decode(b"", frame_size, decode_fec=False)
            pcm_chunks.append(np.frombuffer(plc, dtype=np.int16))
            i += 1

    if not pcm_chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pcm_chunks).astype(np.float32) / 32767.0


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
        # Returns before netem is ever set up, so nothing was impaired.
        return np.zeros(original_len, dtype=np.float32), False

    # --- Build RTP packets ---
    ssrc = random.randint(1, 0xFFFFFFFF)
    rtp_packets = []
    for seq, opus_frame in enumerate(opus_frames):
        ts = seq * frame_size
        hdr = _build_rtp_header(seq, ts, ssrc)
        rtp_packets.append(hdr + opus_frame)

    # --- Network emulation + RTP transmission ---
    # Everything from netem setup to teardown is wrapped in try/finally so a
    # failure mid-transmission (jitter buffer, decode, etc.) can never leave a
    # tc netem qdisc installed on `lo` -- that would silently corrupt every
    # subsequent attack run and any other loopback traffic in the container.
    network = NetworkEmulator()
    port = 30000 + (os.getpid() % 10000)
    sender_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    received_packets = []
    try:
        # Whether the kernel impairment actually engaged. tc needs NET_ADMIN
        # and a netem-capable kernel; without both the RTP round-trip still
        # runs and returns audio that was never impaired, which the benchmark
        # would otherwise score as a successful attack.
        netem_active = network.setup(
            delay_ms, jitter_ms, packet_loss, duplication, reorder, corruption
        )

        receiver_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        receiver_sock.bind(("127.0.0.1", port))

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
    finally:
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

    return decoded.astype(np.float32), netem_active


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

class AttackRequest(BaseModel):
    audio: List[float] = Field(..., max_length=MAX_AUDIO_SAMPLES)
    sampling_rate: int
    bitrate_bps_netem: int = 24000
    frame_duration_ms_netem: int = 20
    delay_ms_netem: int = 100
    jitter_ms_netem: int = 30
    packet_loss_netem: int = 10
    duplication_netem: int = 0
    reorder_netem: int = 3
    corruption_netem: int = 0
    fec_enabled_netem: bool = True
    expected_loss_netem: int = 10
    ns_enabled_netem: bool = True
    vad_enabled_netem: bool = True
    agc_enabled_netem: bool = True
    agc_target_lufs_netem: float = -18
    playout_delay_ms_netem: int = 60


@app.post("/attack")
async def attack(request: AttackRequest):
    try:
        audio = np.array(request.audio, dtype=np.float32)
        result, netem_active = voip_pipeline(
            audio=audio,
            fs=request.sampling_rate,
            bitrate_bps=request.bitrate_bps_netem,
            frame_duration_ms=request.frame_duration_ms_netem,
            delay_ms=request.delay_ms_netem,
            jitter_ms=request.jitter_ms_netem,
            packet_loss=request.packet_loss_netem,
            duplication=request.duplication_netem,
            reorder=request.reorder_netem,
            corruption=request.corruption_netem,
            fec_enabled=request.fec_enabled_netem,
            expected_loss=request.expected_loss_netem,
            ns_enabled=request.ns_enabled_netem,
            vad_enabled=request.vad_enabled_netem,
            agc_enabled=request.agc_enabled_netem,
            agc_target_lufs=request.agc_target_lufs_netem,
            playout_delay_ms=request.playout_delay_ms_netem,
        )
        if not netem_active:
            logger.warning(
                "netem did not engage; the audio made the codec and "
                "jitter-buffer round trip but carries no kernel-level "
                "delay, loss or reordering"
            )
        return {"audio": result.tolist(), "netem_active": netem_active}
    except Exception as e:
        logger.error(f"Attack failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "network_transmission"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "10020"))
    uvicorn.run(app, host=host, port=port)
