#!/usr/bin/env python3
"""Scan src/ for Python files that contain Numba imports or decorators."""
 
import re
from pathlib import Path
 
NUMBA_PATTERNS = re.compile(
    r"""
    ^\s*import\s+numba              |  # import numba
    ^\s*from\s+numba\b              |  # from numba import ...
    @njit                           |  # @njit / @njit(...)
    @numba\.jit                     |  # @numba.jit(...)
    @numba\.njit                    |  # @numba.njit(...)
    @numba\.vectorize               |  # @numba.vectorize(...)
    @numba\.guvectorize             |  # @numba.guvectorize(...)
    @numba\.cfunc                   |  # @numba.cfunc(...)
    numba\.typed                       # numba.typed.List / Dict usage
    """,
    re.MULTILINE | re.VERBOSE,
)
 
def main():
    src = Path("src")
    if not src.is_dir():
        print("No src/ directory found. Run this from your project root.")
        raise SystemExit(1)
 
    hits = sorted(
        p for p in src.rglob("*.py")
        if NUMBA_PATTERNS.search(p.read_text(encoding="utf-8", errors="ignore"))
    )
 
    if not hits:
        print("No Numba usage found under src/.")
    else:
        print(f"Found {len(hits)} file(s) with Numba code:\n")
        for p in hits:
            print(f"  {p}")
 
if __name__ == "__main__":
    main()
 