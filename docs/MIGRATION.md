# Migrating to the packaged benchmark (v1.0.0)

The benchmark now ships as the `deepmarkpy` package. Benchmark behavior,
results, configs, CLI flags, and the docker-compose workflow are unchanged;
what changes is how you install, launch, and import it.

## Install

```bash
pip install -e .[all]          # from a checkout (dev)
# or, pinned, without a checkout:
pip install "deepmarkpy @ git+https://github.com/deepmark/deepmarkpy-benchmark@v1.0.0"
```

Extras: `[metrics]` (ViSQOL), `[native-attacks]` (pycodec2, audiocomplib),
`[all]`. NISQA remains a manual install (see README).

## Launch

| Before | Now |
|---|---|
| `python src/run.py --wav_files_dir ...` | `deepmark-benchmark --wav_files_dir ...` |

`python src/run.py` still works through a deprecation shim. All flags are
unchanged; `--report_dir` (default `./report`) and `--plugins_dir` are new
and optional.

## Imports

| Before | Now |
|---|---|
| `from core.base_attack import BaseAttack` | `from deepmarkpy.core.base_attack import BaseAttack` |
| `from core.base_model import BaseModel` | `from deepmarkpy.core.base_model import BaseModel` |
| `from benchmark import Benchmark` | `from deepmarkpy.benchmark import Benchmark` |
| `from plugin_manager import PluginManager` | `from deepmarkpy.plugin_manager import PluginManager` |
| `from utils.utils import ...` | `from deepmarkpy.utils.utils import ...` |

## Custom plugins

In-tree plugin directories keep working from a source checkout exactly as
before. With an installed package, drop your plugin directories
(`attack.py`/`model.py` + `config.json`) under any folder and point the
benchmark at it:

```bash
deepmark-benchmark --plugins_dir /path/to/my_plugins ...
# or
export DEEPMARK_PLUGINS_DIR=/path/to/my_plugins
```

## Consumers of plugin engines

The per-plugin inference modules are importable as
`deepmarkpy.plugins.{attacks,models}.<name>.inference` and expose a uniform
`Engine` class (see `docs/ENGINE_CONVENTIONS.md`); these paths are the
stable public API from v1.0.0.
