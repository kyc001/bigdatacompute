FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    g++ \
    gcc \
    gdb \
    git \
    htop \
    less \
    libgomp1 \
    make \
    numactl \
    openssh-server \
    pkg-config \
    procps \
    python3 \
    python3-numpy \
    python3-pip \
    python3-venv \
    rsync \
    sudo \
    time \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash ics \
    && echo "ics:ics" | chpasswd \
    && usermod --append --groups sudo ics \
    && mkdir -p /run/sshd /workspace \
    && ssh-keygen -A

RUN printf '%s\n' \
    'PasswordAuthentication yes' \
    'KbdInteractiveAuthentication no' \
    'PermitRootLogin no' \
    'AllowTcpForwarding yes' \
    'GatewayPorts no' \
    'X11Forwarding no' \
    'UsePAM yes' \
    > /etc/ssh/sshd_config.d/99-task2-vscode.conf

COPY rec-sys/task2/docker/entrypoint.sh /usr/local/bin/task2-sshd-entrypoint
RUN chmod +x /usr/local/bin/task2-sshd-entrypoint

ENV OMP_NUM_THREADS=16 \
    OMP_THREAD_LIMIT=16 \
    OPENBLAS_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    TASK2_CPP_CXX=g++ \
    TASK2_CPP_MARCH=haswell \
    TASK2_CPP_FLAGS=

WORKDIR /workspace
EXPOSE 22

ENTRYPOINT ["/usr/local/bin/task2-sshd-entrypoint"]
