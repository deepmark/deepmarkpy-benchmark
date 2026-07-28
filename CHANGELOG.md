# Changelog

## Unreleased

- Restored isolated HTTP/Compose execution for all eight dependency-heavy AI
  attacks while keeping their numerical and model implementations packaged in
  `deepmarkpy`.
- Added a shared remote attack client and a lazy FastAPI service adapter.
- Replaced six per-folder AI Dockerfiles with isolated targets that install the
  current package and validate each environment with `pip check`.
- Added Compose services for Encodec and Descript Audio Codec.
- Completed the XCodec2 and ClearVoice extras, bounded the modern Torch stack,
  pinned the BigVGAN checkout, and removed aggregate extras that implied the
  mutually incompatible AI runtimes could share one environment.
- Preserved canonical attack parameters and legacy aliases across the HTTP
  boundary.

## v1.1.0 - 2026-07-17

- Replaced the HTTP proxy implementations for VAE, diffusion, speech tokenization,
  neural vocoder, and both speech-enhancement attacks with direct, lazily loaded
  packaged implementations suitable for in-process and SageMaker execution.
- Added per-attack optional dependency groups for the packaged AI attacks.
- Split the core PyTorch runtime from the optional modern Transformers stack so
  XCodec2 can retain its older, image-specific Transformers dependency.
- Broadened the torch runtime range to include XCodec2's tested torch 2.4 line;
  standard attack images continue to preinstall torch 2.7.1 in the base layer.
- Kept XCodec2 as an image-specific no-dependencies install because its pinned
  Transformers requirement conflicts with the shared torch dependency group.
- Kept ClearVoice image-specific because its NumPy `<2.0` constraint conflicts
  with newer NumPy releases; broadened the core NumPy range so the Python 3.11
  ClearVoice image and NumPy 2.x consumers are both supported.
- Preserved the existing canonical parameter contract and legacy aliases.

## v1.0.0 - 2026-07-09

- Canonicalized attack parameter configs to use clean attack-local names.
- Added namespaced CLI attack overrides such as `--gaussian_noise.snr_db`.
- Kept existing suffixed attack flags as deprecated compatibility aliases.
- Added a parameter contract snapshot test for attack config keys.
- Packaged the benchmark as `deepmarkpy` with the `deepmark-benchmark` console script.
- Moved source code under `src/deepmarkpy` and included plugin configs as package data.
- Added optional dependency extras for metrics, native tools, model families, server tools, and development.
- Added CI coverage for package install, tests, slim install behavior, and CLI startup.
