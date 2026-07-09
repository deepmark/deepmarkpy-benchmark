# Contributing

## Versioning

`deepmarkpy` follows SemVer from `v1.0.0` onward.

- Major: breaking changes to attack config keys, `apply()` kwargs, model interfaces, or contract snapshot removals/renames.
- Minor: new attacks, new models, new optional parameters, or additive contract snapshot changes.
- Patch: bug fixes and implementation changes that preserve the public contract.

The contract snapshot in `tests/data/attack_param_contract.json` is the interface gate. If a change modifies that snapshot, update `CHANGELOG.md` and choose the version bump according to the rules above.

## Local Checks

```bash
pip install -e ".[dev]"
pytest tests -q
```

Before publishing a release, build and test the wheel, not only the editable install:

```bash
python -m build
pip install dist/deepmarkpy-*.whl
```
