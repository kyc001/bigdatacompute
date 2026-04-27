# cross_platform_checklist.md

## 说明

- 当前主验收平台：Windows
- Linux 部分只保留步骤清单，当前不作为优先执行项
- 所有步骤默认在项目根目录执行

## Windows 验证清单

1. 激活开发环境并安装依赖：

```powershell
micromamba activate test
python -m pip install -r requirements.txt
```

2. 跑测试：

```powershell
pytest -q
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
python benchmark.py --main main.py --data Data.txt --out bench.csv --interval 0.05 --runs 3 --modes csr,csr_block --K 8 --dtype float32
```

6. 检查：

- `bench.csv` 存在
- CSV 列顺序为 `run_id,mode,K,dtype,peak_rss_mb,wall_sec,iters,top10_signature`
- `peak_rss_mb < 80`

7. 执行打包：

```powershell
Get-Content compile-parameter.txt
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

## Linux 验证清单（可选）

1. 创建并激活 Python 环境，安装 `requirements.txt`
2. 运行：

```bash
pytest -q
python main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32
```

3. 检查 `Res.txt` 与 stdout JSON
4. 按 `compile-parameter.txt` 的同等参数执行 `pyinstaller`
5. 运行可执行文件并对比 `top10_signature`
