#!/usr/bin/env python3
"""Extract Track1 local data files from track1.zip."""

from __future__ import annotations

import zipfile
from pathlib import Path


TASK2_DIR = Path(__file__).resolve().parents[1]
ZIP_PATH = TASK2_DIR / "track1.zip"
OUT_DIR = TASK2_DIR / "track1" / "secure_data_full_1024"
ZIP_PREFIX = "students/secure_data_full_1024/"
FILES = [
    "meta.json",
    "P.npy",
    "Q.npy",
    "global_mean.npy",
    "incremental.npy",
    "test.npy",
    "judge_data.bin",
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for filename in FILES:
            member = ZIP_PREFIX + filename
            target = OUT_DIR / filename
            info = archive.getinfo(member)
            if target.exists() and target.stat().st_size == info.file_size:
                print(f"exists {target}")
                continue
            target.write_bytes(archive.read(member))
            print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
