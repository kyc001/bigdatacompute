# PageRank Coursework

本项目用于完成“大数据计算及应用”课程 PageRank 作业：读取 `Data.txt`
中的有向边表，计算 PageRank，并把 Top-100 节点写入 `Res.txt`。

当前代码策略是：**保留 `main.py` 单文件主实现**，所有冻结接口、CLI 调度、
CSR 迭代、稠密基线调用入口和结果输出都可以从 `main.py` 找到。分块实现按
B 侧职责保留在 `blocks.py`，实验与分析脚本统一归档到 `scripts/`。

## 环境

使用课程/本机环境：

```powershell
micromamba activate test
```

依赖文件：

```powershell
python -m pip install -r requirements.txt
```

## 快速运行

运行主提交模式：

```powershell
python main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32 --beta 0.85 --eps 1e-8
```

运行纯 CSR：

```powershell
python main.py --data Data.txt --out Res.txt --mode csr --dtype float32 --beta 0.85 --eps 1e-8
```

运行单元测试：

```powershell
python -m pytest -q
```

## 可复现实验脚本

所有开发期实验脚本集中在 `scripts/`，避免污染根目录。

| 脚本 | 用途 |
| --- | --- |
| `scripts/baseline_dense.py` | E1 稠密基线，对照 CSR 内存与时间。 |
| `scripts/benchmark.py` | 多次运行 `main.py`，采样 RSS 和运行时间，输出 CSV。 |
| `scripts/sweep.py` | E5/E6 参数扫描，扫 `beta` / `eps`。 |
| `scripts/run_e8.py` | E8 dead-end 策略对比：补偿、忽略、删除。 |
| `scripts/analyze_dataset.py` | 数据集画像，输出节点/边/degree/dead-end 等统计。 |
| `scripts/plot.py` | 从实验 CSV/JSON 生成报告图表和 summary JSON。 |
| `scripts/memoryuse-python.py` | 老师提供的运行时内存测量辅助脚本，作为 benchmark 之外的人工复核工具。 |

示例命令：

```powershell
python scripts/baseline_dense.py --data Data.txt --out Res_dense.txt --dtype float32
python scripts/benchmark.py --main main.py --data Data.txt --out bench.csv --runs 3 --modes 'csr,csr_block'
python scripts/sweep.py --main main.py --data Data.txt --out experiments/sweep_results.csv
python scripts/run_e8.py --data Data.txt --out experiments/E8.csv
python scripts/analyze_dataset.py --data Data.txt --out-json experiments/dataset_stats.json
python scripts/plot.py --summary-json experiments/report_summary.json --out-dir report/fig
```

PowerShell 会把未加引号的 `csr,csr_block` 当成数组拆开，所以 `--modes`
参数在 PowerShell 中请使用 `'csr,csr_block'`。

## 可移动验证包

`portable_check/` 是本地换机验证目录，不作为最终 zip 提交内容。目录中包含
`main.exe`、`Data.txt`、`run_check.ps1` 和说明文件，可在另一台 Windows
机器或支持 Windows 可执行文件互操作的 WSL 环境中快速复核可执行文件：

```powershell
cd portable_check
.\run_check.ps1
```

脚本会生成 `Res_check.txt`，检查 Top-10 签名是否为
`75,8686,9678,5104,725,3257,468,7730,7175,5526`，并确认结果文件为 100 行。

## 最终提交目录

`2413575_柯云超_2412235_匡航逸_2413507_蒋林瀞_第一次作业/` 已按提交用途重建：

- `源码/`：`main.py`、`blocks.py`、`scripts/`、`tests/`、依赖与打包参数。
- `实验结果/`：`Res.txt`、实验摘要和冻结运行记录。
- `实验报告/`：新版 `Report.pdf`。
- `可执行文件/`：重新打包后的 `main.exe`。

该目录默认不包含 `Data.txt`。

## 目录结构

```text
.
├─ main.py                         # 单文件主实现与固定 CLI 入口
├─ blocks.py                       # Block Matrix / memmap 分块迭代
├─ mock_graph.py                   # 测试辅助图与 reference 工具
├─ scripts/                        # 可复现实验、分析、绘图脚本
│  └─ memoryuse-python.py           # 老师提供的内存测量辅助脚本
├─ tests/                          # pytest 回归测试
├─ experiments/                    # 实验结果、summary、冻结运行记录
├─ docs/                           # 过程文档、交接文档、报告草稿
├─ report/                         # LaTeX 报告、PDF、图表资源
├─ portable_check/                  # 本地可移动 Windows 验证包，不随最终 zip 提交
├─ INTERFACE.md                    # B 冻结的接口契约
├─ PROJECT_STATUS.md               # 当前状态与后续计划
├─ requirements.txt
└─ Data.txt
```

## 输出约定

`Res.txt` 每行格式：

```text
NodeID Score
```

- 最多输出 100 行，节点不足 100 个时输出全部节点。
- `Score` 固定保留 10 位小数。
- 排序规则：PageRank 降序；分数相同时按原始 `NodeID` 升序。
- `main.py` stdout 最后一行必须是合法 JSON，供 `scripts/benchmark.py`
  和 `scripts/sweep.py` 解析。

## 文档

- 当前状态：[PROJECT_STATUS.md](PROJECT_STATUS.md)
- 冻结接口：[INTERFACE.md](INTERFACE.md)
- 文档索引：[docs/README.md](docs/README.md)
- 交接文档：[docs/handover/](docs/handover/)
- 报告草稿：[docs/report_drafts/](docs/report_drafts/)
- 协作流程：[docs/process/](docs/process/)
- 打包命令：[docs/packaging/compile-parameter.txt](docs/packaging/compile-parameter.txt)
