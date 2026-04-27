# PROJECT_STATUS.md

更新时间：2026-04-27 19:05 UTC+8

本文档记录当前整理后的项目状态。当前原则已经调整为：`main.py` 使用单文件主实现，
实验/分析脚本统一归档到 `scripts/`，文档材料集中到 `docs/`。

## 当前结构状态

| 区域 | 状态 | 说明 |
| --- | --- | --- |
| `main.py` | 已恢复 | 单文件主实现，保留 `load_graph / power_iteration / dump_top_k / run_pagerank` 等接口。 |
| `blocks.py` | 保留 | B 侧分块实现仍在根目录，供 `main.py --mode csr_block` 调用。 |
| `scripts/` | 已整理 | 可复现实验、分析、绘图脚本已归档到此目录。 |
| `docs/` | 已整理 | 交接文档、流程文档、报告草稿已集中存放。 |
| 临时目录 | 已清理 | 删除 `.tmp_runtime/`、`.tmp_test*/`、`.pytest_cache/`、`__pycache__/`。 |
| 无用脚本 | 已删除 | 删除旧的 `memoryuse-cpp.py`、`memoryuse-python.py`，保留现有 benchmark 链路。 |

## 保留脚本

| 路径 | 用途 | 保留原因 |
| --- | --- | --- |
| `scripts/baseline_dense.py` | E1 稠密基线 | 报告需要和 CSR/Block 做对照。 |
| `scripts/benchmark.py` | 性能/RSS 采样 | 实验结果可复现的主入口。 |
| `scripts/sweep.py` | beta / eps 扫描 | 支撑 E5/E6。 |
| `scripts/run_e8.py` | dead-end 策略对比 | 支撑 E8。 |
| `scripts/analyze_dataset.py` | 数据集画像 | 支撑报告第 1 章和图表。 |
| `scripts/plot.py` | 图表生成 | 从实验结果生成 `report/fig` 和 summary JSON。 |

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

## 推荐复现顺序

```powershell
micromamba activate test
python -m pytest -q
python main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32 --beta 0.85 --eps 1e-8
python scripts/benchmark.py --main main.py --data Data.txt --out bench.csv --runs 3 --modes csr,csr_block
python scripts/run_e8.py --data Data.txt --out experiments/E8.csv
```

## 后续 TODO

1. 用 `micromamba activate test` 后重新跑完整测试和主程序冒烟。
2. 决定最终提交目录 `2413575_..._第一次作业/源码/` 是否同步本轮脚本归档结构。
3. 若报告正文引用根目录脚本路径，需要更新为 `scripts/...`。
4. PyInstaller 打包前确认 `docs/packaging/compile-parameter.txt` 与最终提交目录中的打包参数一致。

## 风险

| 风险 | 等级 | 说明 |
| --- | --- | --- |
| 脚本移动后旧路径引用失效 | 中 | README 已更新；报告草稿和历史 handover 中可能仍有旧路径。 |
| 最终提交目录未同步 | 中 | 当前根目录已整理，提交包目录需要人工确认是否重拷。 |
| 环境误用 | 低 | 请使用 `micromamba activate test`，不要用系统 `C:\\msys64\\mingw64\\bin\\python.exe`。 |
