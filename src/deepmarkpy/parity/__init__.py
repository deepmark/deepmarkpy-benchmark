"""CPU/GPU parity: prove a component computes the same numbers on another device.

Nothing runs on a GPU in a benchmark until this package has recorded a PASS
for that (service, operation, image). The pieces:

- :mod:`~deepmarkpy.parity.compare`: element-wise and frame-wise differences
  between a reference output and a candidate.
- :mod:`~deepmarkpy.parity.tiers`: the tolerance tiers (EXACT, NUMERIC,
  QUANTIZED) and how each case is judged.
- :mod:`~deepmarkpy.parity.corpus`: the fixed inputs every run uses.
- :mod:`~deepmarkpy.parity.verdicts`: verdict records, keyed by image ID.
- :mod:`~deepmarkpy.parity.runner`: runs one engine in-process inside its
  service image. It is standalone, so it is mounted into images rather than
  imported from them.
"""
