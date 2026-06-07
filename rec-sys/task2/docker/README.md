# Task2 Docker SSH Environment

This container is for local Track1 evaluation under a Linux GCC/OpenMP setup.

It defaults to:

- Ubuntu 24.04
- `gcc` / `g++` with OpenMP
- `OMP_NUM_THREADS=16`
- Docker memory limit `4g`
- benchmark compile target `-march=haswell`
- SSH user `ics`, password `ics`
- host port `127.0.0.1:20022`

Start it from the repository root:

```powershell
docker compose -f rec-sys/task2/docker/compose.yml up -d --build
```

If Docker Desktop exposes at least 16 CPUs and you also want a hard Docker CPU quota, start it with the override:

```powershell
docker compose -f rec-sys/task2/docker/compose.yml -f rec-sys/task2/docker/compose.16cpu.yml up -d --build
```

On a host where Docker exposes fewer than 16 CPUs, the override will fail at container creation time. The default compose file still sets the OpenMP thread limit to 16, which matches the benchmark call path, but Docker cannot schedule more physical CPUs than the VM exposes.

VSCode SSH config:

```sshconfig
Host task2-docker
    HostName 127.0.0.1
    Port 20022
    User ics
    RemoteForward 127.0.0.1:17890 127.0.0.1:17890
    ServerAliveInterval 30
    ServerAliveCountMax 4
    ExitOnForwardFailure yes
```

The same snippet is saved as `rec-sys/task2/docker/ssh_config.example`.

Open `/workspace` after connecting.

Run the judge-like flow inside the container:

```bash
cd /workspace
bash rec-sys/task2/docker/run_eval.sh 10
```

Data note: put `track1.zip` at `rec-sys/task2/track1.zip` before running, or place `judge_data.bin` under `rec-sys/task2/track1/secure_data_full_1024/`.

Change defaults without editing files:

```powershell
$env:DEV_PASSWORD = "your-password"
$env:TASK2_CPP_MARCH = "haswell"
docker compose -f rec-sys/task2/docker/compose.yml up -d --build
```

Stop it:

```powershell
docker compose -f rec-sys/task2/docker/compose.yml down
```
