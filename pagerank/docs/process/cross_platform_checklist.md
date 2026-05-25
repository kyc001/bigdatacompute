# cross_platform_checklist.md

## 说明

- 当前主验收平台：Windows
- WSL 可用于源码验证；本机 WSL 通过 Windows 版 micromamba 进入 `test` 环境
- `main.exe` 是 Windows 可执行文件，不代表 Linux/macOS 原生可执行文件
- 所有步骤默认在项目根目录执行

## Windows 验证清单

1. 激活开发环境并安装依赖：

```powershell
micromamba activate test
python -m pip install -r requirements.txt
```

2. 跑测试：

```powershell
python -m pytest -q
```

3. 跑源码版主程序：

```powershell
python main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32
```

4. 检查：

- `Res.txt` 已生成
- 前 100 行格式为 `NodeID Score`
- 最后一行 stdout 是合法 JSON
- JSON 中 `iters > 0`

5. 跑 benchmark：

```powershell
python scripts/benchmark.py --main main.py --data Data.txt --out bench.csv --interval 0.05 --runs 3 --modes 'csr,csr_block' --K 8 --dtype float32
```

6. 检查：

- `bench.csv` 存在
- CSV 列顺序为 `run_id,mode,K,dtype,peak_rss_mb,wall_sec,iters,top10_signature`
- `peak_rss_mb < 80`

PowerShell 会把未加引号的 `csr,csr_block` 拆成数组，`--modes` 一律使用
`'csr,csr_block'`。

7. 执行打包：

```powershell
Get-Content docs/packaging/compile-parameter.txt
pyinstaller --noconfirm --clean --onefile --name main main.py --exclude-module matplotlib --exclude-module tkinter --exclude-module PIL --exclude-module pytest
```

8. 运行打包后的 exe：

```powershell
.\dist\main.exe --data Data.txt --out Res_packaged.txt --mode csr_block --K 8 --dtype float32
```

9. 对比源码版与打包版：

- `Res.txt` 与 `Res_packaged.txt` 的 Top-10 节点顺序一致
- stdout JSON 的 `iters` 一致
- `top10_signature` 一致

10. 运行本地可移动验证包：

```powershell
cd portable_check
.\run_check.ps1
cd ..
```

11. 检查：

- 脚本输出 `PASS`
- `portable_check/Res_check.txt` 为 100 行
- Top-10 签名为 `75,8686,9678,5104,725,3257,468,7730,7175,5526`

`portable_check/` 中含 `Data.txt`，只用于本地换机验证，不默认加入最终提交 zip。

## WSL / Linux 验证清单（可选）

本机 WSL 原生 Python 未必包含 numpy/pytest，可直接调用 Windows 版 micromamba：

```bash
cd /mnt/d/Study/26sp/bigdata/pagerank大作业
/mnt/d/micromamba/micromamba.exe run -n test python -m pytest -q
/mnt/d/micromamba/micromamba.exe run -n test python main.py --data Data.txt --out Res_wsl.txt --mode csr_block --K 8 --dtype float32 --beta 0.85 --eps 1e-8
```

若在真正 Linux/macOS 上复现，需要创建对应平台的 Python 环境并运行：

```bash
python -m pytest -q
python main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32
```

随后按同等 PyInstaller 参数重新打包本平台可执行文件，并对比 `top10_signature`。
