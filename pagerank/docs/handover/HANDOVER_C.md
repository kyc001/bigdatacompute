# HANDOVER_C.md

本文档以“C 写给未来要接手或复查实验/报告部分的人”的口吻整理。请先读 `INTERFACE.md`，再按本文复现实验和统稿。

数据源：`experiments/report_summary.json`，最后核对 2026-04-27。

## 1. 项目当前状态

### 已完成

- C 侧脚本已完成并可复用：
  - `analyze_dataset.py`
  - `sweep.py`
  - `plot.py`
- 报告草稿已覆盖第 1、5、7 章：
  - `report_ch1_draft1.md`
  - `report_ch5.md`
  - `report_ch7.md`
  - `report_final.md`
- 实验数据与图表已生成：
  - `experiments/dataset_stats.json`
  - `experiments/degree_distribution.csv`
  - `experiments/E1_dense.csv`
  - `experiments/E3.csv`
  - `experiments/E4.csv`
  - `experiments/E7.csv`
  - `experiments/E8.csv`
  - `experiments/sweep_results.csv`
  - `experiments/report_summary.json`
  - `report/fig/*.png`
- A 的 `load_graph / power_iteration / dump_top_k` 已接入，E1/E2/E8 数据已进入最终汇总。
- B 的 `csr_block` 主路径、benchmark、PyInstaller 命名配置已对齐，提交目录已有 `main.exe / Res.txt / Report.pdf / README.txt`。

### 进行中

- 最终提交目录仍是占位名 `待填写学号姓名_第一次作业/`，需要人工填入真实学号姓名。
- `report/main.pdf` 本轮按要求不修改；若 Markdown 草稿继续调整，需要人工决定是否重编 LaTeX。
- 打包版与源码版的最终 Top-10 签名对拍仍需人工复核。

### 待开始

- 生成最终 zip 并由 B 队长复核。
- 发送提交邮件前核对附件、命名、收件人与报告分工说明。

## 2. 文件清单

| 路径 | 作用 | 数据来源 |
| --- | --- | --- |
| `analyze_dataset.py` | 数据集统计脚本 | `Data.txt` |
| `experiments/dataset_stats.json` | 数据集总体统计 | `analyze_dataset.py` |
| `experiments/degree_distribution.csv` | 入度/出度频数表 | `analyze_dataset.py` |
| `sweep.py` | beta/eps 参数扫描 | `main.py` CLI |
| `experiments/sweep_results.csv` | E5/E6 扫描结果 | `sweep.py` |
| `experiments/E1_dense.csv` | E1 稠密基线 | `baseline_dense.py` / benchmark |
| `experiments/E3.csv` | E3 K 值对比 | `benchmark.py` |
| `experiments/E4.csv` | E2/E4 CSR 与 CSR+Block 对比 | `benchmark.py` |
| `experiments/E7.csv` | E7 dtype 对比 | `benchmark.py` |
| `experiments/E8.csv` | E8 dead-end 策略对比 | `experiments/run_e8.py` |
| `experiments/report_summary.json` | 报告最终数字汇总 | `plot.py` |
| `plot.py` | 从 CSV/JSON 生成图表和 summary | `experiments/*.csv` |
| `report/fig/*.png` | 报告图表资源 | `plot.py` |
| `report_ch1_draft1.md` | 第 1 章草稿 | 数据集统计 |
| `report_ch5.md` | 第 5 章草稿 | `report_summary.json` |
| `report_ch7.md` | 第 7 章草稿 | 团队分工 |
| `report_final.md` | 最终统稿 Markdown 草稿 | A/B/C 各章 |
| `submission_checklist.md` | 提交前核验清单 | 最终提交目录 |
| `email_draft.md` | 提交邮件草稿 | 最终 zip 名称 |

## 3. 实验结果索引

| 实验 | 当前状态 | CSV/JSON | 报告位置 |
| --- | --- | --- | --- |
| 数据集分析 | 已完成 | `dataset_stats.json`、`degree_distribution.csv` | 第 1 章 |
| E1 稠密基线 | 已完成 | `E1_dense.csv`、`report_summary.json` | 第 5 章 5.4 |
| E2 CSR | 已完成 | `E4.csv`、`report_summary.json` | 第 5 章 5.4 |
| E3 分块 K | 已完成 | `E3.csv`、`report_summary.json` | 第 5 章 5.3 |
| E4 CSR+Block | 已完成 | `E4.csv`、`report_summary.json` | 第 5 章 5.4 |
| E5 beta 扫描 | 已完成 | `sweep_results.csv`、`report_summary.json` | 第 5 章 5.5 |
| E6 eps 扫描 | 已完成最终残差汇总 | `sweep_results.csv`、`report_summary.json` | 第 5 章 5.6 |
| E7 dtype 对比 | 已完成 | `E7.csv`、`report_summary.json` | 第 5 章 5.7 |
| E8 dead-end 策略 | 已完成 | `E8.csv`、`report_summary.json` | 第 5 章 5.8 |

## 4. 关键实验数字

| 组别 | 配置 | 峰值 RSS / MB | 时间 / s | 迭代轮数 |
| --- | --- | ---: | ---: | ---: |
| E1 | `dense` | 412.64 | 0.602 | 14 |
| E2 | `csr` | 34.36 | 0.726 | 15 |
| E4 | `csr_block` | 34.66 | 0.838 | 15 |
| E7 | `float32` | 34.22 | 0.789 | 15 |
| E7 | `float64` | 35.34 | 0.719 | 12 |

| K | 峰值 RSS / MB | 时间 / s | 迭代轮数 |
| ---: | ---: | ---: | ---: |
| 4 | 35.09 | 0.704 | 15 |
| 8 | 34.98 | 0.729 | 15 |
| 16 | 34.92 | 0.856 | 15 |

E8 结论：

- `ignore` 的 `rank_sum=0.602009`，违反概率守恒。
- `delete` 与补偿法的 Top-10 Jaccard 为 `0.666667`，改变了原图语义。
- 最终主实现应保持 `compensation` 质量补偿法。

## 5. 如何复现实验

从仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python analyze_dataset.py --data Data.txt
python sweep.py --main main.py --data Data.txt --out experiments/sweep_results.csv --result-dir experiments/sweep_outputs --mode csr_block --K 8 --dtype float32
python plot.py --summary-json experiments/report_summary.json --out-dir report/fig
python main.py --data Data.txt --beta 0.85 --eps 1e-8 --out Res.txt --mode csr_block --K 8 --dtype float32
```

## 6. 报告统稿注意事项

- 所有实验数字以 `experiments/report_summary.json` 为准。
- 图表必须可追溯到 CSV/JSON，不手填数据。
- `main.py` 只输出最终 `delta`，没有逐轮 residual trace；报告中只能写最终残差汇总。
- `report/main.pdf` 本轮未重编，若 Markdown 改动要进入 PDF，需要人工重新跑 LaTeX 并复核版式。
- 第 7 章必须保留团队分工表。

## 7. 提交清单核验结果

当前已具备：

- `待填写学号姓名_第一次作业/main.py`
- `待填写学号姓名_第一次作业/compile-parameter.txt`
- `待填写学号姓名_第一次作业/main.exe`
- `待填写学号姓名_第一次作业/Res.txt`
- `待填写学号姓名_第一次作业/Report.pdf`
- `待填写学号姓名_第一次作业/README.txt`

仍需人工确认：

- 最终目录和 zip 是否改成真实学号姓名。
- `Report.pdf` 是否需要按最新 Markdown 草稿重编。
- 打包版 `main.exe` 是否与源码版 `top10_signature` 一致。

## 8. 距离 DDL 的剩余时间与 Top-3 风险

当前核对日期为 `2026-04-27 UTC+8`，DDL 为 `2026-05-15 12:00 UTC+8`，约剩 `18 天 12 小时`。

Top-3 风险：

1. 最终目录仍是占位名，zip 命名需要人工填写真实成员信息。
2. Markdown 草稿已更新，但 `report/main.pdf` 本轮未改，需决定是否重编。
3. 打包版与源码版尚需最后一次对拍，避免 exe 产物与冻结 `Res.txt` 不一致。
