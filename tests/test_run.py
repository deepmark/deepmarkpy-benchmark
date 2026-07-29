"""Tests for deepmarkpy.cli — to_json_safe utility."""

import numpy as np
import pytest

from deepmarkpy.cli import build_parser, to_json_safe


class TestToJsonSafe:
    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_json_safe(arr)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_numpy_float32(self):
        val = np.float32(3.14)
        result = to_json_safe(val)
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.01

    def test_numpy_float64(self):
        val = np.float64(2.718)
        result = to_json_safe(val)
        assert isinstance(result, float)

    def test_numpy_int32(self):
        val = np.int32(42)
        result = to_json_safe(val)
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_int64(self):
        val = np.int64(99)
        result = to_json_safe(val)
        assert result == 99
        assert isinstance(result, int)

    def test_nested_dict(self):
        data = {
            "accuracy": np.float64(95.5),
            "watermark": np.array([1, 0, 1]),
            "name": "test",
        }
        result = to_json_safe(data)
        assert result["accuracy"] == 95.5
        assert result["watermark"] == [1, 0, 1]
        assert result["name"] == "test"

    def test_nested_list(self):
        data = [np.float32(1.0), np.int64(2), "hello"]
        result = to_json_safe(data)
        assert result == [1.0, 2, "hello"]

    def test_deeply_nested(self):
        data = {"results": {"file.wav": {"attack": {"acc": np.float64(99.9)}}}}
        result = to_json_safe(data)
        assert result["results"]["file.wav"]["attack"]["acc"] == 99.9

    def test_plain_python_types_unchanged(self):
        data = {"a": 1, "b": 2.0, "c": "hello", "d": True, "e": None}
        result = to_json_safe(data)
        assert result == {"a": 1, "b": 2.0, "c": "hello", "d": True, "e": "N/A"}

    def test_empty_structures(self):
        assert to_json_safe({}) == {}
        assert to_json_safe([]) == []


class TestAttackParamParser:
    def _parser(self):
        return build_parser(
            models=["DummyModel"],
            attacks=["GaussianNoiseAttack"],
            valid_args={"gaussian_noise": {"snr_db": 35}},
        )

    def test_namespaced_attack_flag_sets_canonical_dest(self):
        args = self._parser().parse_args(
            [
                "--wav_files_dir",
                "samples",
                "--wm_model",
                "DummyModel",
                "--gaussian_noise.snr_db",
                "20",
            ]
        )
        assert args.gaussian_noise__snr_db == 20

    def test_legacy_attack_flag_sets_same_dest(self):
        args = self._parser().parse_args(
            [
                "--wav_files_dir",
                "samples",
                "--wm_model",
                "DummyModel",
                "--snr_db_gaussian_noise",
                "20",
            ]
        )
        assert args.gaussian_noise__snr_db == 20
