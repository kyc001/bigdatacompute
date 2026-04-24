# HANDOVER.md

本文档以“C 写给未来要接手或复查实验/报告部分的人”的口吻整理。请先读 `INTERFACE.md`，再按本文复现实验和统稿。

## 1. 项目当前状态

### 已完成

- 已阅读作业原始 `requirement.pdf`，要求包括：Python 入口 `main.py`、输出 `Res.txt`、处理 dead-end/spider-trap、强制 Sparse Matrix + Block Matrix、内存低于 `80 MB`、时间低于 `60 s`、zip 命名严格按成员信息。
- 已阅读 `分工.md` 与 `yourjob.md`，确认 C 负责数据分析、参数扫描、可视化、报告第 1/5/7 章、统稿、提交清单与邮件草稿。
- 已阅读队友仓库，B 已冻结 `INTERFACE.md`，并提供 `main.py`、`blocks.py`、`benchmark.py`、E3/E4/E7 CSV、第 4/6 章草稿。
- C 侧已完成：
  - `analyze_dataset.py`
  - `sweep.py`
  - `plot.py`
  - `report_ch1_draft1.md`
  - `report_ch5.md`
  - `report_ch7.md`
  - `report_final.md`
  - `submission_checklist.md`
  - `email_draft.md`
- 已生成真实数据：
  - `experiments/dataset_stats.json`
  - `experiments/degree_distribution.csv`
  - `experiments/sweep_results.csv`
  - `report/fig/degree_distribution.png`
  - `report/fig/memory_peak_bar.png`
  - `report/fig/k_memory_time.png`
  - `report/fig/beta_top10_similarity.png`
  - `report/fig/epsilon_residual_summary.png`

### 进行中

- A 的最终 `load_graph / power_iteration / dump_top_k` 尚未正式替换当前兼容实现。
- A 的 E1/E2/E8 和第 2/3 章还未接入。
- 最终 PDF 尚未导出。
- PyInstaller 打包和干净机器验证尚未完成。

### 待开始

- 用 A/B 最终 CSV 重新生成完整四张核心图。
- 将 `report_final.md` 导出为 `Report.pdf`。
- 按真实成员学号姓名打 zip，并由 B 队长复核后提交。

## 2. 文件清单

| 路径 | 作用 | 依赖输入 |
| --- | --- | --- |
| `analyze_dataset.py` | 数据集统计脚本，输出 JSON/CSV/PNG | `Data.txt` |
| `experiments/dataset_stats.json` | 数据集总体统计 | `analyze_dataset.py` |
| `experiments/degree_distribution.csv` | 入度/出度频数表 | `analyze_dataset.py` |
| `report/fig/degree_distribution.png` | 第 1 章度分布图 | `degree_distribution.csv` |
| `sweep.py` | 通过 CLI 扫描 beta/eps | `main.py`、`Data.txt`、`INTERFACE.md` |
| `experiments/sweep_results.csv` | E5/E6 扫描结果 | `sweep.py` |
| `experiments/sweep_outputs/` | 每次扫描的 Top-100 结果 | `sweep.py` |
| `plot.py` | 从 CSV 生成报告图 | `experiments/*.csv` |
| `report/fig/memory_peak_bar.png` | 内存峰值阶段图 | `experiments/E4.csv`，待 A/B 补完整 |
| `report/fig/k_memory_time.png` | K 与内存/时间双轴图 | `experiments/E3.csv` |
| `report/fig/beta_top10_similarity.png` | beta 与 Top-10 相似度图 | `sweep_results.csv` |
| `report/fig/epsilon_residual_summary.png` | eps 与迭代轮数/最终残差图 | `sweep_results.csv` |
| `report_ch1_draft1.md` | 第 1 章草稿 | 数据集统计与图 |
| `report_ch5.md` | 第 5 章草稿 | E3/E4/E5/E6/E7 CSV |
| `report_ch7.md` | 第 7 章草稿 | 团队分工 |
| `report_final.md` | 阶段统稿 Markdown | A/B/C 各章 |
| `submission_checklist.md` | 提交前核验清单 | 最终提交目录 |
| `email_draft.md` | 提交邮件草稿 | 最终 zip 名称 |
| `HANDOVER.md` | 本交接文档 | 当前全部产物 |

## 3. 实验结果索引

| 实验 | 当前状态 | CSV | 图 | 报告位置 |
| --- | --- | --- | --- | --- |
| E1 稠密基线 | 待 A 补 | 待补 | `memory_peak_bar.png` 待补柱 | 第 5 章 5.4 |
| E2 CSR | 阶段性用 B 的 `csr` 数据，待 A 确认 | `experiments/E4.csv` | `memory_peak_bar.png` | 第 5 章 5.4 |
| E3 分块 K | 已有 B 数据 | `experiments/E3.csv` | `k_memory_time.png` | 第 5 章 5.3 |
| E4 CSR+Block | 已有 B 数据 | `experiments/E4.csv` | `memory_peak_bar.png` | 第 5 章 5.4 |
| E5 beta 扫描 | 已完成 | `experiments/sweep_results.csv` | `beta_top10_similarity.png` | 第 5 章 5.5 |
| E6 eps 扫描 | 已完成汇总，缺逐轮 residual trace | `experiments/sweep_results.csv` | `epsilon_residual_summary.png` | 第 5 章 5.6 |
| E7 dtype 对比 | 已有 B 数据 | `experiments/E7.csv` | 第 5 章表格，暂未单独出图 | 第 5 章 5.7 |
| E8 dead-end 策略 | 待 A 补 | 待补 | 待补 | 第 5 章后续补充 |

## 4. 如何复现实验

从仓库根目录执行：

```powershell
cd D:\data_compute\bigdatacompute-master-extracted\bigdatacompute-master
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

运行测试：

```powershell
pytest -q
```

生成数据集统计：

```powershell
python analyze_dataset.py --data Data.txt
```

运行 E5/E6 扫描：

```powershell
python sweep.py --main main.py --data Data.txt --out experiments/sweep_results.csv --result-dir experiments/sweep_outputs --mode csr_block --K 8 --dtype float32
```

重新生成图表：

```powershell
python plot.py --e3 experiments/E3.csv --e4 experiments/E4.csv --sweep experiments/sweep_results.csv --out-dir report/fig
```

跑主提交版本：

```powershell
python main.py --data Data.txt --beta 0.85 --eps 1e-8 --out Res.txt --mode csr_block --K 8 --dtype float32
```

## 5. 数据分析关键发现

1. 数据集规模为 `10000` 个节点、`150000` 条边，平均入度和出度均为 `15.00`。
2. dead-end 节点数为 `1500`，占比 `15.00%`，必须使用 dangling mass 补偿。
3. 弱连通分量共有 `501` 个，最大弱连通分量包含 `9500` 个节点，占比 `95.00%`。
4. E5 中 `beta=0.80/0.85/0.90` 的 Top-10 集合一致，`beta=0.70/0.95` 各出现 1 个节点替换。
5. E6 中 `eps=1e-6/1e-8/1e-10` 的 Top-10 完全一致，但迭代轮数从 `9` 增至 `15` 再到 `27`。

## 6. 报告统稿规范

- 中文正文建议使用宋体或等价中文正文字体，英文与代码使用等宽字体。
- 公式用独立块展示，变量如 `beta`、`eps`、`N`、`M` 保持全文一致。
- 图编号统一为“图 1-1”“图 5-1”等，表编号统一为“表 5-1”等。
- 所有实验数字必须能追溯到 CSV 或 JSON。
- 图片统一使用 PNG，建议 300 dpi。
- 参考文献统一放到报告末尾，网页或课程材料注明访问日期。
- 最终 PDF 前清除“待补”“阶段性”等字样，除非确实作为限制说明保留。

## 7. 与 A、B 的协作点

- `sweep.py` 依赖 A 写的 `main.py` CLI，调用方式必须遵守 B 的 `INTERFACE.md` 第 5 节。
- `plot.py` 只读 CSV，不 import `main.py`。
- 第 5 章的内存/时间数据来自 B 的 `benchmark.py` 输出，目前已有 E3/E4/E7。
- A 需要补 E1/E2/E8、报告第 2/3 章，并确认最终 `Res.txt`。
- B 需要完成 PyInstaller 打包、跨平台验证和最终 zip 复核。
- 如果最终报告必须画逐轮残差曲线，需要 A/B 增加真实 residual trace CSV；当前 stdout JSON 只有最终 `delta`。

## 8. 提交清单核验结果

最新清单见 `submission_checklist.md`。当前阶段未通过的关键项：

- 最终 zip 尚未生成。
- `Report.pdf` 尚未导出。
- `main.exe` 尚未生成并在干净机器验证。
- A 的 E1/E2/E8 与第 2/3 章尚未接入。
- 完整 E1 → E2 → E3 → E4 内存柱状图仍缺 A/B 最终真实数据。

## 9. 距离 DDL 的剩余时间与 Top-3 风险

当前日期按环境为 `2026-04-24`，DDL 为 `2026-05-15 12:00 UTC+8`，约剩 `20 天 14 小时`。

Top-3 风险：

1. A 的最终算法实现未及时接入，导致 E1/E2/E8、报告第 2/3 章和最终 `Res.txt` 延后。
2. PyInstaller 打包临近 DDL 才做，可能遇到依赖、路径或 exe 运行环境问题。
3. 报告图表与 CSV 版本不一致，特别是后续 A/B 更新实验后忘记重跑 `plot.py`。

建议下一步：先让 A 合入最终算法并跑 `pytest -q`，再由 B 重新跑 benchmark，最后由 C 重跑 `sweep.py` 和 `plot.py` 完成终稿。
