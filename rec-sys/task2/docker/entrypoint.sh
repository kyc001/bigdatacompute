#!/usr/bin/env bash
set -euo pipefail

DEV_USER="${DEV_USER:-ics}"
DEV_PASSWORD="${DEV_PASSWORD:-ics}"

mkdir -p /run/sshd

if id "${DEV_USER}" >/dev/null 2>&1; then
    echo "${DEV_USER}:${DEV_PASSWORD}" | chpasswd
    install -d -m 700 -o "${DEV_USER}" -g "${DEV_USER}" "/home/${DEV_USER}/.ssh"
    if [[ -n "${AUTHORIZED_KEYS:-}" ]]; then
        printf '%s\n' "${AUTHORIZED_KEYS}" > "/home/${DEV_USER}/.ssh/authorized_keys"
        chown "${DEV_USER}:${DEV_USER}" "/home/${DEV_USER}/.ssh/authorized_keys"
        chmod 600 "/home/${DEV_USER}/.ssh/authorized_keys"
    fi
fi

cat >/etc/profile.d/task2-env.sh <<'EOF'
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TASK2_CPP_CXX="${TASK2_CPP_CXX:-g++}"
export TASK2_CPP_MARCH="${TASK2_CPP_MARCH:-haswell}"
export TASK2_CPP_FLAGS="${TASK2_CPP_FLAGS:-}"
EOF

echo "task2 sshd ready: user=${DEV_USER}, workspace=/workspace, OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}, TASK2_CPP_MARCH=${TASK2_CPP_MARCH:-haswell}"
exec /usr/sbin/sshd -D -e
