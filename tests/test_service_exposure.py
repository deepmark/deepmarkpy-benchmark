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
        bound = ast.unparse(field.value)
        assert "MAX_AUDIO_B64_CHARS" in bound, (
            f"{path}: audio bound is not the shared MAX_AUDIO_B64_CHARS. "
            "Audio crosses the wire base64-encoded, so the cap must be in "
            "characters -- a sample-count cap would not bound the body."
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


class TestPackagedModulesReachTheImages:
    """Every module a service imports must be in the package the image ships.

    ml-services-base carries deepmarkpy and is built by hand, outside compose
    (README's install section). `docker-compose build` therefore reports
    success while every dependent image still holds whatever package the base
    was last built with -- and a service importing a module added since then
    dies at startup, which the contract check reports only as a port timeout.

    This checks the source side of that: a module imported by an app.py has to
    exist in the tree the base image installs from. It does not prove the base
    image is current -- only a rebuild does that -- but it catches the case
    where a module is referenced and simply is not there.
    """

    def _package_modules(self):
        pkg = os.path.join(REPO, "src", "deepmarkpy")
        found = set()
        for root, _dirs, files in os.walk(pkg):
            rel = os.path.relpath(root, os.path.dirname(pkg))
            for f in files:
                if f.endswith(".py"):
                    mod = os.path.join(rel, f[:-3]).replace(os.sep, ".")
                    found.add(mod.removesuffix(".__init__"))
        return found

    @pytest.mark.parametrize("path", APPS, ids=lambda p: os.path.basename(os.path.dirname(p)))
    def test_every_deepmarkpy_import_exists(self, path):
        modules = self._package_modules()
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("deepmarkpy"):
                assert node.module in modules, (
                    f"{os.path.basename(os.path.dirname(path))}/app.py imports "
                    f"{node.module}, which is not in src/deepmarkpy. The image "
                    "will fail at startup, and the contract check will report it "
                    "as a port timeout rather than an ImportError."
                )

    def test_the_wire_module_is_packaged(self):
        """It is imported by all 16 services and was added after the base was
        last built, which is exactly how the stale-base failure happened."""
        assert "deepmarkpy.core.wire" in self._package_modules()
