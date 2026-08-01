"""Every HTTP client must fail fast and say which service failed.

Clients are parsed with ``ast`` rather than imported: several live in plugin
packages whose ML dependencies are absent on the host.
"""

import ast
import os

import pytest

PLUGINS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "deepmarkpy", "plugins"
)

# Attack plugins that reach a service over HTTP.
HTTP_ATTACKS = [
    "vae", "diffusion", "neural_vocoder", "speech_enhancement_1",
    "speech_enhancement_2", "speech_tokenization", "opus_codec",
    "encodec", "descript_audio_codec", "network_transmission",
]


def _tree(plugin):
    path = os.path.join(PLUGINS_DIR, "attacks", plugin, "attack.py")
    assert os.path.exists(path), f"missing attack.py for {plugin}"
    return ast.parse(open(path).read()), open(path).read()


def _post_calls(tree):
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "post"
    ]


@pytest.mark.parametrize("plugin", HTTP_ATTACKS)
def test_every_post_has_a_timeout(plugin):
    tree, _ = _tree(plugin)
    calls = _post_calls(tree)
    assert calls, f"{plugin}: no requests.post found"
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        assert "timeout" in kwargs, (
            f"{plugin}: requests.post at line {call.lineno} has no timeout; "
            "a wedged service would hang the run indefinitely"
        )


@pytest.mark.parametrize("plugin", HTTP_ATTACKS)
def test_every_client_checks_http_status(plugin):
    _, src = _tree(plugin)
    assert "raise_for_status" in src, (
        f"{plugin}: no raise_for_status; a 5xx body would be parsed as a result"
    )


@pytest.mark.parametrize("plugin", HTTP_ATTACKS)
def test_null_audio_is_rejected_by_value_not_key(plugin):
    """A service reporting an error in a 200 body sends audio=null.

    Guarding on key presence lets that through to ``np.array(None)``, a 0-d
    object array that fails much later with an unrelated message.
    """
    _, src = _tree(plugin)
    assert 'response_data.get("audio") is None' in src or "audio\" not in response_data" not in src, (
        f"{plugin}: guards on key presence only, so a null audio value passes"
    )


@pytest.mark.parametrize("plugin", ["vae", "diffusion", "neural_vocoder",
                                    "speech_enhancement_1", "speech_enhancement_2"])
def test_failure_message_names_the_service(plugin):
    """The five clients that previously failed anonymously."""
    _, src = _tree(plugin)
    tree = ast.parse(src)
    cls = next(n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    assert cls in src.split("class ", 1)[1], f"{plugin}: class name not referenced"
    # The raised error and the logged failure both carry the attack's name.
    assert src.count(cls) >= 2, (
        f"{plugin}: failure paths do not name {cls}, so the user cannot tell "
        "which service failed"
    )
