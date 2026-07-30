"""Deprecated launcher kept for the pre-package workflow.

``python src/run.py ...`` keeps working; the supported entry points are the
``deepmark-benchmark`` console script and ``python -m deepmarkpy.run``.
"""

import warnings

from deepmarkpy.run import main

if __name__ == "__main__":
    warnings.warn(
        "python src/run.py is deprecated; use the deepmark-benchmark "
        "console script (pip install .) or python -m deepmarkpy.run",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
