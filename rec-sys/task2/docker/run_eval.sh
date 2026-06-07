#!/usr/bin/env bash
set -euo pipefail

cd "${WORKSPACE:-/workspace}"

RUNS="${1:-10}"
RUN_TIMEOUT="${RUN_TIMEOUT:-1800}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TASK2_CPP_CXX="${TASK2_CPP_CXX:-g++}"
export TASK2_CPP_MARCH="${TASK2_CPP_MARCH:-haswell}"
export TASK2_CPP_FLAGS="${TASK2_CPP_FLAGS:-}"

DATA_DIR="rec-sys/task2/track1/secure_data_full_1024"

if [[ ! -f "${DATA_DIR}/judge_data.bin" ]]; then
    if [[ -f "rec-sys/task2/track1.zip" ]]; then
        python3 rec-sys/task2/scripts/extract_track1_data.py
    else
        echo "missing ${DATA_DIR}/judge_data.bin"
        echo "put track1.zip at rec-sys/task2/track1.zip, or place judge_data.bin under ${DATA_DIR}/"
        exit 1
    fi
fi

python3 rec-sys/task2/scripts/scan_cpp.py rec-sys/task2/track1/solution.cpp
python3 rec-sys/task2/scripts/cpp_smoke.py
python3 rec-sys/task2/track1/benchmark.py \
    --solution rec-sys/task2/track1/solution.cpp \
    --language cpp \
    --benchmark-runs "${RUNS}" \
    --run-timeout "${RUN_TIMEOUT}"
