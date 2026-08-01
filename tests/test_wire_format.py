"""Audio must survive the HTTP hop bit-for-bit.

Audio used to cross as a JSON array of decimal floats, which round-tripped
float64 exactly. The base64 form must be exactly as faithful: anything
narrower would change attacked audio and therefore the reported accuracy,
which would be a behavior change wearing a transport change's clothes.
"""

import ast
import base64
import glob
import json
import os

import numpy as np
import pytest

from deepmarkpy.core.wire import AUDIO_ENCODING, decode_audio, encode_audio

REPO = os.path.join(os.path.dirname(__file__), "..")
APPS = sorted(glob.glob(os.path.join(REPO, "src", "deepmarkpy", "plugins", "*", "*", "app.py")))
CLIENTS = sorted(
    glob.glob(os.path.join(REPO, "src", "deepmarkpy", "plugins", "attacks", "*", "attack.py"))
    + glob.glob(os.path.join(REPO, "src", "deepmarkpy", "plugins", "models", "*", "model.py"))
)


class TestCodecIsLossless:
    @pytest.mark.parametrize("n", [1, 2, 1000, 16000])
    def test_round_trip_is_bit_identical(self, n):
        audio = 0.5 * np.random.default_rng(0).standard_normal(n)
        assert np.array_equal(decode_audio(encode_audio(audio)), audio)

    def test_matches_what_the_json_path_produced(self):
        """The old wire form round-tripped float64 exactly; so must this one."""
        audio = 0.5 * np.random.default_rng(1).standard_normal(4096)
        via_json = np.asarray(json.loads(json.dumps(audio.tolist())), dtype=np.float64)
        via_wire = decode_audio(encode_audio(audio))
        assert np.array_equal(via_json, via_wire)

    @pytest.mark.parametrize("value", [0.0, -0.0, 1e-300, 1e300, np.pi])
    def test_extreme_values_survive(self, value):
        arr = np.array([value], dtype=np.float64)
        out = decode_audio(encode_audio(arr))
        assert out[0] == value or (np.isnan(value) and np.isnan(out[0]))

    def test_nan_and_inf_survive(self):
        arr = np.array([np.nan, np.inf, -np.inf, 0.0])
        out = decode_audio(encode_audio(arr))
        assert np.array_equal(out, arr, equal_nan=True)

    def test_float32_input_is_widened_not_truncated(self):
        arr = (0.5 * np.random.default_rng(2).standard_normal(512)).astype(np.float32)
        out = decode_audio(encode_audio(arr))
        assert np.array_equal(out, arr.astype(np.float64))

    def test_output_is_json_safe(self):
        """The encoded form has to survive json.dumps -- it rides in a body."""
        encoded = encode_audio(np.array([1.0, 2.0]))
        assert isinstance(encoded, str)
        assert json.loads(json.dumps({"audio": encoded}))["audio"] == encoded

    def test_a_list_still_decodes(self):
        """A service must still understand a client that predates this."""
        assert np.array_equal(decode_audio([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))

    def test_none_passes_through(self):
        assert decode_audio(None) is None


class TestItIsActuallySmaller:
    def test_beats_the_json_encoding(self):
        audio = 0.5 * np.random.default_rng(3).standard_normal(16000)
        json_bytes = len(json.dumps(audio.tolist()))
        wire_bytes = len(encode_audio(audio))
        assert wire_bytes < json_bytes / 1.5, (
            f"base64 {wire_bytes} vs json {json_bytes} -- not worth the change"
        )

    def test_encoding_is_named(self):
        assert AUDIO_ENCODING == "base64-float64"


class TestEveryHopUsesIt:
    """A single service left on the old format silently breaks that plugin."""

    @pytest.mark.parametrize("path", APPS, ids=lambda p: os.path.basename(os.path.dirname(p)))
    def test_service_decodes_and_encodes(self, path):
        src = open(path).read()
        assert "decode_audio" in src, f"{path}: reads audio without decoding it"
        tree = ast.parse(src)
        audio_fields = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name) and n.target.id == "audio"
        ]
        for f in audio_fields:
            assert ast.unparse(f.annotation) == "str", (
                f"{path}: audio field is {ast.unparse(f.annotation)}, not the encoded str"
            )

    @pytest.mark.parametrize("path", CLIENTS, ids=lambda p: os.path.basename(os.path.dirname(p)))
    def test_client_encodes_what_it_sends(self, path):
        src = open(path).read()
        if '"audio"' not in src:
            pytest.skip("does not send audio over HTTP")
        assert "encode_audio" in src, f"{path}: sends audio without encoding it"
        assert '"audio": ' not in src.replace('"audio": encode_audio', ""), (
            f"{path}: an audio payload bypasses encode_audio"
        )
