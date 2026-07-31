"""Engine base-class and naming contract, checked structurally.

Engine modules import their ML runtimes (torch, clearvoice, ...), so they
cannot be imported in the host test environment. These tests parse each
plugin's ``inference.py`` with ``ast`` instead and lock the contract:
exactly one engine class per module, named as registered below, deriving
from the correct base.
``deepmarkpy.core.inference`` itself must stay import-light so consumers
can type against it without any ML runtime.
"""

import ast
import os

import pytest

from deepmarkpy.core.inference import BaseAttackEngine, BaseModelEngine  # noqa: F401

PLUGINS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "deepmarkpy", "plugins"
)

# plugin dir -> (engine class name, base class name)
ENGINES = {
    "attacks/vae": ("VAEEngine", "BaseAttackEngine"),
    "attacks/diffusion": ("DiffusionEngine", "BaseAttackEngine"),
    "attacks/neural_vocoder": ("NeuralVocoderEngine", "BaseAttackEngine"),
    "attacks/speech_enhancement_1": ("SpeechEnhancement1Engine", "BaseAttackEngine"),
    "attacks/speech_enhancement_2": ("SpeechEnhancement2Engine", "BaseAttackEngine"),
    "attacks/speech_tokenization": ("SpeechTokenizationEngine", "BaseAttackEngine"),
    "attacks/opus_codec": ("OpusCodecEngine", "BaseAttackEngine"),
    "attacks/encodec": ("EncodecEngine", "BaseAttackEngine"),
    "attacks/descript_audio_codec": ("DescriptAudioCodecEngine", "BaseAttackEngine"),
    "models/audio_seal": ("AudioSealEngine", "BaseModelEngine"),
    "models/aware": ("AwareEngine", "BaseModelEngine"),
    "models/perth": ("PerthEngine", "BaseModelEngine"),
    "models/silent_cipher": ("SilentCipherEngine", "BaseModelEngine"),
    "models/timbrewm": ("TimbreWMEngine", "BaseModelEngine"),
    "models/wavmark": ("WavMarkEngine", "BaseModelEngine"),
}

REQUIRED_METHODS = {"BaseAttackEngine": {"apply"}, "BaseModelEngine": {"embed", "detect"}}


def _module_tree(plugin):
    path = os.path.join(PLUGINS_DIR, plugin, "inference.py")
    assert os.path.exists(path), f"missing inference.py for {plugin}"
    return ast.parse(open(path).read())


@pytest.mark.parametrize("plugin", sorted(ENGINES))
def test_engine_class_name_base_and_methods(plugin):
    name, base = ENGINES[plugin]
    tree = _module_tree(plugin)
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    engines = [c for c in classes
               if any(isinstance(b, ast.Name) and b.id in REQUIRED_METHODS for b in c.bases)]
    assert len(engines) == 1, f"{plugin}: expected exactly one engine class, found {len(engines)}"
    cls = engines[0]
    assert cls.name == name, f"{plugin}: engine class is {cls.name}, expected {name}"
    assert any(isinstance(b, ast.Name) and b.id == base for b in cls.bases), (
        f"{plugin}: {cls.name} does not derive from {base}"
    )
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    missing = REQUIRED_METHODS[base] - methods
    assert not missing, f"{plugin}: {cls.name} missing {sorted(missing)}"


def test_core_inference_is_import_light():
    path = os.path.join(PLUGINS_DIR, "..", "core", "inference.py")
    tree = ast.parse(open(path).read())
    allowed = {"abc", "numpy", "typing"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            continue
        assert names <= allowed, f"heavy import in core/inference.py: {names - allowed}"
