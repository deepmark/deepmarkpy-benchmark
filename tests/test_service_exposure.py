"""The inference services must not be reachable or exhaustible from the network.

Every service publishes an unauthenticated FastAPI app with /embed, /detect or
/attack, plus FastAPI's own /docs and /openapi.json. Nothing in the stack
authenticates, so the only thing standing between those endpoints and the rest
of the network is how docker-compose publishes the port.
"""

import ast
import glob
import os
import re

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
COMPOSE = os.path.join(REPO, "docker-compose.yml")
APPS = sorted(glob.glob(os.path.join(REPO, "src", "deepmarkpy", "plugins", "*", "*", "app.py")))


def _published_ports():
    """Every host port mapping in docker-compose.yml."""
    with open(COMPOSE) as fh:
        return re.findall(r'^\s+- "([^"]+)"\s*(?:#.*)?$', fh.read(), re.M)


def test_compose_publishes_something():
    assert len(_published_ports()) >= 16, "port mappings not found; regex is stale"


@pytest.mark.parametrize("mapping", _published_ports())
def test_every_service_is_loopback_only(mapping):
    """A bare "PORT:PORT" binds 0.0.0.0 — the LAN and any VPN interface."""
    assert mapping.startswith("127.0.0.1:"), (
        f'"{mapping}" publishes on all interfaces. These services have no '
        "authentication, so that exposes an embed/detect oracle and an "
        "unbounded compute sink to every device on the network."
    )


def test_app_files_were_found():
    assert len(APPS) >= 16, "app.py glob is stale"


@pytest.mark.parametrize("path", APPS, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_audio_arrays_are_length_capped(path):
    """FastAPI buffers and parses the whole body before validation runs."""
    src = open(path).read()
    tree = ast.parse(src)

    audio_fields = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "audio"
    ]
    assert audio_fields, f"{path}: no audio field found"
    for field in audio_fields:
        assert field.value is not None, (
            f"{os.path.basename(os.path.dirname(path))}: audio is declared with "
            "no length bound, so one request can size the allocation"
        )
        assert "MAX_AUDIO_SAMPLES" in ast.unparse(field.value), (
            f"{path}: audio bound is not the shared MAX_AUDIO_SAMPLES"
        )


class TestClearVoiceModelNameIsWhitelisted:
    """speech_enhancement_2 forwards model_name into a config path.

    ClearVoice interpolates it into a filename and only validates the name
    after opening the file, so an unchecked value turned the endpoint into a
    filesystem oracle: absolute paths and ../ both resolved, and the YAML key
    names of whatever it found came back in the error response.
    """

    def _engine_source(self):
        path = os.path.join(
            REPO, "src", "deepmarkpy", "plugins", "attacks",
            "speech_enhancement_2", "inference.py",
        )
        return open(path).read()

    def test_a_whitelist_exists(self):
        assert "SPEECH_ENHANCEMENT_MODELS" in self._engine_source()

    def test_apply_rejects_before_reaching_clearvoice(self):
        src = self._engine_source()
        tree = ast.parse(src)
        apply_fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "apply"
        )
        body = ast.unparse(apply_fn)
        guard = body.index("SPEECH_ENHANCEMENT_MODELS")
        # The constructor call, not the word in the docstring.
        construct = body.index("ClearVoice(")
        assert guard < construct, (
            "the whitelist check must run before ClearVoice is constructed"
        )

    def test_the_shipped_default_is_allowed(self):
        """A whitelist that rejects the config default would break every run."""
        import json

        config_path = os.path.join(
            REPO, "src", "deepmarkpy", "plugins", "attacks",
            "speech_enhancement_2", "config.json",
        )
        default = json.load(open(config_path))["model_name_se2"]
        assert default in self._engine_source(), (
            f"config default {default!r} is not in the whitelist"
        )


class TestStandaloneImagesCarryWhatTheyImport:
    """An app.py that imports deepmarkpy needs it installed in its image.

    Most services build on ml-services-base, which installs the package. The
    two that do not must install it themselves, and forgetting is invisible
    until the container starts: uvicorn dies on ModuleNotFoundError and the
    contract check reports it only as a 20-minute port timeout.
    """

    def _dockerfile(self, plugin_dir):
        return open(os.path.join(plugin_dir, "Dockerfile")).read()

    @pytest.mark.parametrize(
        "path", APPS, ids=lambda p: os.path.basename(os.path.dirname(p))
    )
    def test_deepmarkpy_import_is_satisfied(self, path):
        plugin_dir = os.path.dirname(path)
        app_src = open(path).read()
        if "from deepmarkpy" not in app_src and "import deepmarkpy" not in app_src:
            pytest.skip("does not import deepmarkpy")

        dockerfile = self._dockerfile(plugin_dir)
        from_base = "ml-services-base" in dockerfile
        installs_itself = "deepmarkpy-pkg" in dockerfile
        assert from_base or installs_itself, (
            f"{os.path.basename(plugin_dir)}/app.py imports deepmarkpy, but its "
            "image neither builds on ml-services-base nor installs the package. "
            "The container will die at import with ModuleNotFoundError."
        )
