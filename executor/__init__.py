"""Alpha Engine executor package.

Importing arcticdb here guarantees it primes its bundled aws-c-common
allocator BEFORE any submodule pulls in pandas / pyarrow. Python runs
``executor/__init__.py`` exactly once, before any ``from executor.X``
or ``import executor.X``, so any submodule that adds an ``import pandas``
at top level cannot accidentally invert the order.

Background — the macOS-only segfault this prevents:

    Fatal error condition occurred in aws-c-common/source/allocator.c:121:
        allocator != ((void*)0)
    arcticdb_ext: Aws::S3::S3Client constructor

arcticdb's bundled aws-c-common (linked into ``arcticdb_ext.cpython-...so``)
collides with pyarrow's bundled copy (linked into ``libarrow.dylib``)
when arcticdb's S3Client tries to construct an endpoint provider after
pyarrow has already called ``aws_endpoints_ruleset_new_from_string``.
The dynamic linker's two-level namespace on macOS lets both copies load
simultaneously; whichever inits first wins, but the second's shared
allocator state stays uninitialized -> segfault on first ``get_library()``.

Linux (Lambda, EC2 Amazon Linux) is unaffected because the dynamic
linker resolves to a single shared aws-c-common.

This priming was previously inside ``executor/price_cache.py``, but that
file imports late in the chain (``main.py:46`` after ``main.py:42``'s
``from executor.risk_guard``). When ``risk_guard.py`` added a top-level
``import pandas`` in 2026-04-27's DataFrame migration, pandas/pyarrow
loaded before arcticdb_ext could prime, re-introducing the segfault for
``executor/main.py --simulate --dry-run`` on macOS dev machines.
Centralizing the priming in the package ``__init__`` removes the
ordering hazard for any future module that adds a pandas import.
"""

# noqa: F401 -- kept for its side effect on import ordering.
# Must run before any executor submodule imports pandas.
import arcticdb as _arcticdb  # noqa: F401
