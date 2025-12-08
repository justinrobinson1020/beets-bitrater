"""Consolidated thread-clamping for numerical libraries.

These libraries (numba, scipy, OpenBLAS, MKL, etc.) check environment
variables at import time to initialize thread pools.  Setting these vars
to "1" prevents thread explosion when joblib spawns parallel workers.
"""

import os

# Every threading env var we need to clamp.
THREAD_VARS: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",  # macOS Accelerate
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "KMP_BLOCKTIME": "0",
}


def clamp_threads() -> None:
    """Set thread limits using ``setdefault`` — safe for package init.

    Existing user overrides are preserved.
    """
    for var, value in THREAD_VARS.items():
        os.environ.setdefault(var, value)


def clamp_threads_hard() -> None:
    """Set thread limits using direct assignment — for worker processes.

    Overwrites any existing values to guarantee single-threaded execution.
    """
    for var, value in THREAD_VARS.items():
        os.environ[var] = value
