# Changelog

## v1.0.0 - 2026-07-09

- Canonicalized attack parameter configs to use clean attack-local names.
- Added namespaced CLI attack overrides such as `--gaussian_noise.snr_db`.
- Kept existing suffixed attack flags as deprecated compatibility aliases.
- Added a parameter contract snapshot test for attack config keys.
- Packaged the benchmark as `deepmarkpy` with the `deepmark-benchmark` console script.
- Moved source code under `src/deepmarkpy` and included plugin configs as package data.
- Added optional dependency extras for metrics, native tools, model families, server tools, and development.
- Added CI coverage for package install, tests, slim install behavior, and CLI startup.
