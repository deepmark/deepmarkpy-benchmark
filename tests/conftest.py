"""Shared fixtures for the DeepMarkPy benchmark test suite."""

import sys
import os

import numpy as np
import pytest

# Add src/ to Python path so tests can import benchmark modules


@pytest.fixture
def sample_audio():
    """Generate a 1-second sine wave at 440 Hz, 16 kHz sample rate."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def short_audio():
    """Generate a short 0.1-second sine wave."""
    sr = 16000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_watermark():
    """Generate a 16-bit binary watermark."""
    np.random.seed(42)
    return np.random.randint(0, 2, size=16, dtype=np.int32)


@pytest.fixture
def sample_stats():
    """Sample benchmark stats dict with known accuracy values."""
    return {
        "GaussianNoiseAttack": 97.5,
        "SignInversionAttack": 29.0,
        "SmoothingAttack": 53.1,
        "LowpassFilterAttack": 100.0,
        "DiffusionAttack": 41.0,
        "CropBeginningAttack": 82.0,
        "ChorusAttack": 91.0,
    }


@pytest.fixture
def sample_results():
    """Sample benchmark results dict as returned by Benchmark.run()."""
    return {
        "file1.wav": {
            "attacks": {
                "GaussianNoiseAttack": {"accuracy": 95.0},
                "SignInversionAttack": {"accuracy": 30.0},
            },
        },
        "file2.wav": {
            "attacks": {
                "GaussianNoiseAttack": {"accuracy": 100.0},
                "SignInversionAttack": {"accuracy": 28.0},
            },
        },
    }


@pytest.fixture
def sample_results_with_confidence():
    """Results including confidence scores."""
    return {
        "file1.wav": {
            "attacks": {
                "Attack1": {"accuracy": 95.0, "confidence": 0.95},
            },
        },
        "file2.wav": {
            "attacks": {
                "Attack1": {"accuracy": 90.0, "confidence": 0.88},
            },
        },
    }


@pytest.fixture
def sample_results_with_cross_model():
    """Results including cross-model accuracy."""
    return {
        "file1.wav": {
            "attacks": {
                "CrossModelAttack": {
                    "accuracy": 50.0,
                    "accuracy_cross_model": 95.0,
                },
            },
        },
    }
