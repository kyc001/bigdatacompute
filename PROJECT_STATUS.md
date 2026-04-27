# PROJECT_STATUS.md

更新时间：2026-04-27 19:16 UTC+8

本文档记录当前整理后的项目状态。当前原则已经调整为：`main.py` 使用单文件主实现，
实验/分析脚本统一归档到 `scripts/`，文档材料集中到 `docs/`。

## 当前结构状态

| 区域 | 状态 | 说明 |
| --- | --- | --- |
| `main.py` | 已恢复 | 单文件主实现，保留 `load_graph / power_iteration / dump_top_k / run_pagerank` 等接口。 |
| `blocks.py` | 保留 | B 侧分块实现仍在根目录，供 `main.py --mode csr_block` 调用。 |
| `scripts/` | 已整理 | 可复现实验、分析、绘图脚本已归档到此目录。 |
| `docs/` | 已整理 | 交接文档、流程文档、报告草稿已集中存放。 |
| `portable_check/` | 新增 | 本地可移动 Windows 验证包，包含 exe、数据和 PowerShell 检查脚本；不默认进入最终 zip。 |
| 临时目录 | 已清理 | 删除 `.tmp_runtime/`、`.tmp_test*/`、`.pytest_cache/`、`__pycache__/`。 |
| 无用脚本 | 已清理 | 删除旧的无关临时文件；老师提供的 `memoryuse-python.py` 已保留并归档到 `scripts/`。 |

## 保留脚本

| 路径 | 用途 | 保留原因 |
| --- | --- | --- |
| `scripts/baseline_dense.py` | E1 稠密基线 | 报告需要和 CSR/Block 做对照。 |
| `scripts/benchmark.py` | 性能/RSS 采样 | 实验结果可复现的主入口。 |
| `scripts/sweep.py` | beta / eps 扫描 | 支撑 E5/E6。 |
| `scripts/run_e8.py` | dead-end 策略对比 | 支撑 E8。 |
| `scripts/analyze_dataset.py` | 数据集画像 | 支撑报告第 1 章和图表。 |
| `scripts/plot.py` | 图表生成 | 从实验结果生成 `report/fig` 和 summary JSON。 |
| `scripts/memoryuse-python.py` | 老师提供的内存测量脚本 | 作为 `scripts/benchmark.py` 之外的人工复核工具，保留原始逻辑。 |

## 实验摘要

数据来源：`experiments/report_summary.json` 与 `experiments/final_run_20260427/main_run.json`。

| 场景 | 峰值 RSS | 时间 | 迭代轮数 |
| --- | ---: | ---: | ---: |
| dense | 412.64 MB | 0.602 s | 14 |
| csr | 34.36 MB | 0.726 s | 15 |
| csr_block | 34.66 MB | 0.838 s | 15 |
| final_run csr_block | 29.58 MB | 0.828 s | 15 |

冻结运行 Top-10 signature：

```text
75,8686,9678,5104,725,3257,468,7730,7175,5526
```

## 2026-04-27 收尾验证

| 项目 | 状态 | 摘要 |
| --- | --- | --- |
| Windows pytest | 通过 | `11 passed` |
| Windows 源码主程序 | 通过 | `csr_block`，15 轮，Top-10 signature 一致，RSS 约 29.45 MB |
| benchmark smoke | 通过 | PowerShell 使用 `--modes 'csr,csr_block'`，`csr/csr_block` 均 15 轮 |
| sweep smoke | 通过 | `beta=0.85`、`eps=1e-8` 最小扫描生成 2 行结果 |
| WSL pytest | 通过 | `/mnt/d/micromamba/micromamba.exe run -n test`，`11 passed` |
| WSL 源码主程序 | 通过 | `csr_block`，15 轮，Top-10 signature 一致 |
| `portable_check/run_check.ps1` | 通过 | 输出 `PASS portable check` |
| 最终目录 exe | 通过 | `可执行文件/main.exe`，15 轮，Top-10 signature 一致 |

## 推荐复现顺序

```powershell
micromamba activate test
python -m pytest -q
python main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32 --beta 0.85 --eps 1e-8
python scripts/benchmark.py --main main.py --data Data.txt --out bench.csv --runs 3 --modes 'csr,csr_block'
python scripts/run_e8.py --data Data.txt --out experiments/E8.csv
cd portable_check
.\run_check.ps1
```

## 后续 TODO

1. 若老师明确要求提交数据集，再决定是否把 `Data.txt` 加入最终 zip；当前默认不提交数据。
2. 提交前由小组复核最终 zip、报告 PDF 和邮件正文。
3. 发送前可再运行一次 `portable_check/run_check.ps1` 做换机前确认。

## 风险

| 风险 | 等级 | 说明 |
| --- | --- | --- |
| PowerShell 参数拆分 | 中 | `--modes` 必须写成 `'csr,csr_block'`，否则逗号会被拆成两个参数。 |
| 可执行文件跨平台误解 | 中 | `main.exe` 是 Windows 单文件程序；Linux/macOS 需要源码或各自平台重新打包。 |
| 环境误用 | 低 | 请使用 `micromamba activate test`，不要用系统 `C:\\msys64\\mingw64\\bin\\python.exe`。 |
