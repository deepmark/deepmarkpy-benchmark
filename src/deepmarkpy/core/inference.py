"""Base classes for the per-plugin inference engines.

Every containerized plugin ships an ``inference.py`` whose engine class
derives from one of the bases below and follows one constructor
convention::

    EngineClass(config: dict, device: str | None = None)

``config`` is the plugin's ``config.json`` dict. ``device`` of ``None``
means the engine decides (typically cuda-when-available). Construction
loads weights, with one exception: ``speech_enhancement_2`` constructs its
ClearVoice model per ``apply`` call — an intentional, preserved behavior.

The bases type the method names and parameters, not the exact return
shapes, which vary by plugin and are part of each service's frozen
contract: most attacks return an ``np.ndarray`` but ``opus_codec`` returns
``(audio, output_sr)``; model ``detect`` returns an ndarray, a
``(watermark, confidence)`` tuple, a scalar, a bit list, or ``None``
depending on the model.

This module stays import-light (stdlib + numpy only) so consumers can
type against it without pulling any ML runtime.
"""

import abc

import numpy as np


# Upper bound on the audio arrays a service will accept. Every request model
# declared a bare List[float], and FastAPI buffers and parses the whole body
# before validation, so a single large POST could exhaust a container that has
# no memory limit. Ten minutes at 48 kHz is far beyond any benchmark clip --
# the corpora are seconds long -- while still bounding the cost of one request.
MAX_AUDIO_SAMPLES = 48_000 * 600

# The same ceiling expressed for the base64 wire form: 8 bytes per sample,
# then 4 base64 characters per 3 bytes. Requests carry the encoded string, so
# the cap has to be applied in the units the field actually holds.
MAX_AUDIO_B64_CHARS = ((MAX_AUDIO_SAMPLES * 8 + 2) // 3) * 4

# Watermark payloads are tens of bits; nothing legitimate approaches this.
MAX_WATERMARK_BITS = 4096


class BaseAttackEngine(abc.ABC):
    """An audio attack: transforms audio, returning the attacked signal."""

    @abc.abstractmethod
    def apply(self, audio, sampling_rate: int, **params):
        """Apply the attack to ``audio`` at ``sampling_rate``.

        ``params`` carries the attack's request-level parameters (names as
        in the plugin's ``config.json``). Returns the attacked audio as an
        ``np.ndarray`` — except ``opus_codec``, which returns
        ``(np.ndarray, int)`` carrying the decoder's output rate.
        """


class BaseModelEngine(abc.ABC):
    """A watermarking model: embeds and detects watermarks in audio."""

    @abc.abstractmethod
    def embed(self, audio, watermark_data, sampling_rate: int) -> np.ndarray:
        """Embed ``watermark_data`` into ``audio``; returns the watermarked signal."""

    @abc.abstractmethod
    def detect(self, audio, sampling_rate: int):
        """Detect the watermark in ``audio``.

        The return shape is model-specific: an ndarray of bits, a
        ``(watermark, confidence)`` tuple, a scalar score, a bit list, or
        ``None`` on decode failure.
        """
