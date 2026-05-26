#!/usr/bin/env python3
"""Compile and run a tiny C++ smoke test for Track1 solution.cpp."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SOLUTION = REPO_ROOT / "rec-sys" / "task2" / "track1" / "solution.cpp"


SMOKE_SOURCE = f"""
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "{SOLUTION.as_posix()}"

int main() {{
    const int users = 3;
    const int items = 3;
    const int dim = 4;
    std::vector<float> P(users * dim, 0.0f);
    std::vector<float> Q(items * dim, 0.0f);
    IncrementalSVD model;
    model.load_base_model(P.data(), Q.data(), users, items, dim, 3.0f);

    const float before = model.predict(0, 0);
    std::vector<Rating> batch1 = {{{{0, 0, 5.0f}}, {{1, 1, 1.0f}}}};
    std::vector<Rating> batch2 = {{{{0, 0, 5.0f}}, {{2, 2, 4.0f}}}};
    model.update(batch1);
    model.update(batch2);

    const float high = model.predict(0, 0);
    const float low = model.predict(1, 1);
    const float invalid = model.predict(-1, 0);

    std::cout << "before=" << before << " high=" << high
              << " low=" << low << " invalid=" << invalid << "\\n";
    assert(std::fabs(before - 3.0f) < 1e-5f);
    assert(high > before);
    assert(low < before);
    assert(std::fabs(invalid - 3.0f) < 1e-5f);
    return 0;
}}
"""


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="task2_cpp_smoke_") as tmp_name:
        tmp = Path(tmp_name)
        source = tmp / "smoke.cpp"
        exe = tmp / "smoke.exe"
        source.write_text(SMOKE_SOURCE, encoding="utf-8")
        compile_cmd = [
            "g++",
            "-std=c++17",
            "-O2",
            "-fopenmp",
            str(source),
            "-o",
            str(exe),
        ]
        subprocess.run(compile_cmd, check=True)
        subprocess.run([str(exe)], check=True)
    print("C++ smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
