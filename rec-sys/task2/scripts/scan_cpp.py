#!/usr/bin/env python3
"""Minimal local safety scanner for Track1 C++ submissions."""

from __future__ import annotations

import re
import sys
from pathlib import Path


FORBIDDEN_PATTERNS = {
    r"#\s*include\s*<fstream>": "file stream include",
    r"#\s*include\s*<filesystem>": "filesystem include",
    r"\bfopen\s*\(": "fopen",
    r"\bfreopen\s*\(": "freopen",
    r"\bifstream\b": "ifstream",
    r"\bofstream\b": "ofstream",
    r"\bfstream\b": "fstream",
    r"\bsystem\s*\(": "system",
    r"\bpopen\s*\(": "popen",
    r"\bremove\s*\(": "remove",
    r"\brename\s*\(": "rename",
    r"\bunlink\s*\(": "unlink",
    r"\brmdir\s*\(": "rmdir",
    r"\bexec[lvpe]*\s*\(": "exec",
    r"\bfork\s*\(": "fork",
    r'"/data': "hard-coded data path",
    r'"[^"]*\.npy[^"]*"': "hard-coded npy path",
    r'"[^"]*\.bin[^"]*"': "hard-coded bin path",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: scan_cpp.py <solution.cpp>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern, label in FORBIDDEN_PATTERNS.items():
        if re.search(pattern, text):
            print(f"forbidden C++ pattern: {label}")
            return 1

    print("C++ safety scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
