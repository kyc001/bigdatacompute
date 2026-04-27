# HANDOVER.md

本文档以“队长 B 写给未来要接手或复查整个工程的人”的口吻整理当前状态。请先读 `INTERFACE.md`，再读本文件。

> 最近一次全链路复核时间：**2026-04-25（UTC）**。本次复核覆盖 A/B/C 三侧脚本可运行性、`pytest`、`csr` 与 `csr_block` 真数运行、benchmark 单轮验证与 C 侧数据脚本可执行性。

## 0. 本次复核记录（2026-04-25）

- `python -m pytest -q`：11 条测试全部通过。
- `python main.py --mode csr ...`：可完成收敛，`iters=15`，输出 Top-10 签名为 `75,8686,9678,5104,725,3257,468,7730,7175,5526`。
- `python main.py --mode csr_block ...`：可完成收敛，`iters=15`，Top-10 签名与 `csr` 一致。
- `python benchmark.py --runs 1 --modes dense,csr,csr_block ...`：三种模式均可执行并输出 RSS / wall time / iter。
- `python analyze_dataset.py --data Data.txt ...`：可输出数据统计 JSON。
- `python sweep.py --mode csr_block --betas 0.85 --eps-list 1e-8 ...`：可输出 CSV（2 行）。
- `python plot.py --summary-json experiments/report_summary.json --out-dir ...`：可生成图表文件并刷新 summary。

## 1. 项目当前状态

### 已完成

- `INTERFACE.md` 已冻结为 `v1.0`
- 队长侧工程基建已补齐：
  - `README.md`
  - `CONTRIBUTING.md`
  - `requirements.txt`
  - `code_review_checklist.md`
  - `.gitignore`
- B 侧核心代码已实现：
  - `blocks.py`
  - `main.py`
  - `benchmark.py`
  - `mock_graph.py`
- A 侧核心算法代码已接入并通过回归：
  - `load_graph / power_iteration / dump_top_k`
  - `baseline_dense.py`
  - `tests/test_algorithm.py`
  - `report_ch2.md / report_ch3.md`
  - `HANDOVER_A.md`
- C 侧数据流程脚本可执行：
  - `analyze_dataset.py`
  - `sweep.py`
  - `plot.py`
- pytest 回归测试当前 **11/11 通过**
- `csr` 与 `csr_block` 在真实 `Data.txt` 上可运行，Top-10 签名一致
- E1 / E2 / E8 关键数据已有摘要文件 `experiments/report_summary.json`
- 第 4、6、7 章草稿及总报告骨架文件已在仓库中

### 进行中

- 报告最终文字润色、图表替换、总稿拼接与交叉校对
- 最终提交前的打包产物复验（Windows 主验收）

### 待开始

- 按课程模板填入真实学号姓名并生成最终提交压缩包
- 提交前复跑一次全流程并冻结 `Res.txt` 与实验附件

## 2. 文件清单

| 路径 | 作用 | 版本 |
| --- | --- | --- |
| `INTERFACE.md` | 全队冻结接口契约 | `v1.0-frozen` |
| `README.md` | 项目简介、目录、快速开始、分工 | `v0.1` |
| `CONTRIBUTING.md` | Git 分支、commit、PR 规范 | `v0.1` |
| `.gitignore` | 忽略运行产物、打包产物、实验中间文件 | `v0.1` |
| `requirements.txt` | 开发环境依赖 | `v0.1` |
| `code_review_checklist.md` | A / C 代码审查清单 | `v0.1` |
| `mock_graph.py` | 小图手算 mock + 100 节点随机图 + 朴素 reference | `v0.1` |
| `blocks.py` | `build_blocks` 与 `iterate_by_block` | `v0.1` |
| `main.py` | 主入口、A/B 集成版 `load_graph / power_iteration / dump_top_k` 与多 mode 运行 | `v1.2` |
| `benchmark.py` | RSS / wall time 采样与 CSV 输出 | `v0.1` |
| `tests/conftest.py` | pytest 导入路径配置 | `v0.1` |
| `tests/test_pagerank_behaviors.py` | B 侧行为回归测试（与 A 侧测试合计 11 条） | `v0.2` |
| `tests/test_algorithm.py` | A 侧算法回归测试 | `v1.0` |
| `baseline_dense.py` | E1 稠密基线（仅实验对照） | `v1.0` |
| `report_ch2.md` | 第 2 章草稿：算法原理 | `v1.0-draft` |
| `report_ch3.md` | 第 3 章草稿：算法实现细节 | `v1.0-draft` |
| `HANDOVER_A.md` | A 侧交接文档 | `v1.0` |
| `report_ch4.md` | 第 4 章草稿：内存优化 | `v0.1-draft` |
| `report_ch6.md` | 第 6 章草稿：工程实现与部署 | `v0.1-draft` |
| `compile-parameter.txt` | PyInstaller 打包命令 | `v0.1` |
| `cross_platform_checklist.md` | Windows 主验收 + Linux 备用检查清单 | `v0.1` |
| `HANDOVER.md` | 本交接文档（A/B/C 联调后更新） | `v0.2` |

补充说明：

- `experiments/` 目录下存放 E3 / E4 / E7 运行时生成的 CSV 与结果文件，属于开发期中间产物，已被 `.gitignore` 忽略
- `Res*.txt`、`bench*.csv`、`dist/`、`build/` 等都不应视为正式源码交付

## 3. 接口契约

### A 的函数签名

```python
def load_graph(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]
def power_iteration(row_ptr, col_idx, out_deg, N, beta, eps, dtype=np.float32, max_iter=200) -> tuple[np.ndarray, int, float]
def dump_top_k(path: str, ranks: np.ndarray, k: int = 100) -> str
```

### B 的函数签名

```python
def build_blocks(edges, K: int, tmp_dir: str) -> list[dict]
def iterate_by_block(blocks, out_deg, N, beta=0.85, eps=1e-8, dtype=np.float32, max_iter=200, tmp_dir=None) -> tuple[np.ndarray, int, float]
```

### CLI 契约

```bash
python main.py --data Data.txt --beta 0.85 --eps 1e-8 --out Res.txt --mode csr_block --K 8 --dtype float32
```

### `Res.txt` 契约

- 每行 `NodeID Score`
- `Score` 固定 10 位小数
- 共 100 行，按 Score 降序；同分按 NodeID 升序

### stdout JSON 契约

`main.py` 最后一行必须是 JSON，至少包含：

```json
{"peak_rss_mb": 0.0, "wall_sec": 0.0, "iters": 0}
```

当前实现还会额外输出：

- `mode`
- `K`
- `dtype`
- `delta`
- `n_nodes`
- `n_edges`
- `top10_signature`
- `out`

### dead-end 公式

冻结公式如下：

```text
r_new[v] = (1 - beta) / N
         + beta * dangling_mass / N
         + beta * sum(r[u] / out_deg[u] for every edge u -> v, out_deg[u] > 0)
```

其中：

```text
dangling_mass = sum(r[i] for i in dead_end_nodes)
```

## 4. Git 协作规范

### 分支模型

- `main`：稳定、可提交版本
- `dev`：团队集成分支
- `feature-*`：成员功能分支

### Commit 格式

```text
<type>(<scope>): <summary>
```

示例：

- `feat(block): add memmap-based iterate_by_block`
- `docs(interface): freeze stdout json contract`
- `test(mock): add dead-end regression cases`

### PR 流程

1. 从 `dev` 拉最新代码
2. 新建 `feature-*` 分支
3. 本地跑 `pytest -q`
4. 提交 PR 到 `dev`
5. 在描述中写清目标 / 变更 / 风险 / 验证 / 待办
6. 至少 1 人 review 通过后再合并
7. 阶段性稳定后由队长把 `dev` 合并到 `main`

## 5. 如何本地跑通

以下步骤以 Windows + `micromamba test` 环境为主。

### 5.1 clone 与进入目录

```bash
git clone <your-repo-url> pagerank-coursework
cd pagerank-coursework
```

### 5.2 激活环境并安装依赖

```powershell
micromamba activate test
python -m pip install -r requirements.txt
```

### 5.3 跑 mock / pytest

```powershell
pytest -q
python mock_graph.py --show all
```

### 5.4 跑真实数据

```powershell
python main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32
```

预期：

- 生成 `Res.txt`
- stdout 最后一行是 JSON

### 5.5 跑 benchmark

```powershell
python benchmark.py --main main.py --data Data.txt --out bench.csv --interval 0.05 --runs 3 --modes csr,csr_block --K 8 --dtype float32
```

### 5.6 打包

```powershell
Get-Content compile-parameter.txt
pyinstaller --noconfirm --clean --onefile --name main main.py --exclude-module matplotlib --exclude-module tkinter --exclude-module PIL --exclude-module pytest
```

### 5.7 运行打包产物

```powershell
.\dist\main.exe --data Data.txt --out Res_packaged.txt --mode csr_block --K 8 --dtype float32
```

### 5.8 对拍源码版与打包版

- 对比 `Res.txt` 与 `Res_packaged.txt` 的 Top-10 节点顺序
- 对比 stdout JSON 中的 `iters`
- 对比 `top10_signature`

## 6. 已知问题与权衡

1. `dump_top_k` 的原始 ID 恢复依赖模块级 `_LAST_NODE_IDS`（由 `load_graph` 更新），属于“隐式状态”，若未来做同进程并发多图，需要改为显式传递映射。
2. `build_blocks(edges_array, K, tmp_dir)` 在输入是纯边数组且图中存在“只作为 dead-end、完全不出现在边表中的节点”时，无法自动推断完整 `N`；这类场景必须按契约传 `(edges, N)`。
3. `dense` 模式只用于 mock / 小图 / 对照实验，不适合作为真实大图主提交流程。
4. 实验数据在不同机器上的绝对时间 / RSS 会波动，应关注相对趋势和 Top-10 稳定性。
5. Windows 打包链路为主验收路径，Linux 侧仍以源码运行验证为主。

## 7. 下一步建议

1. 队长在 Windows 上执行最终 PyInstaller 验收，并保留 exe 与源码版对拍记录（`top10_signature`、`iters`、`delta`）。
2. 统一报告口径：把 `HANDOVER.md`、`HANDOVER_A.md`、`HANDOVER_C.md` 中重复指标对齐到同一版摘要数据。
3. 提交前冻结最终参数（`beta=0.85, eps=1e-8, mode=csr_block, K=8, dtype=float32`）并补充“变更即复测”规则。
4. 最终 zip 前复跑：`pytest -q`、`main.py csr_block`、`benchmark.py`、`sweep.py`（最小集）。
5. 若时间允许，再做一轮 `K` 与 `dtype` 敏感性复核，确认报告图表与文字结论一致。

## 8. 与 A、C 的集成点

### A 已接入并完成的点

- `main.py` 已使用 A 的真实 `load_graph / power_iteration / dump_top_k`
- A 侧测试文件 `tests/test_algorithm.py` 已纳入回归
- `report_ch2.md / report_ch3.md / HANDOVER_A.md` 已就位
- 本次复核已重新跑通：
  - `pytest -q`
  - `python main.py --data Data.txt --out .tmp_check_res_csr.txt --mode csr --K 8 --dtype float32`
  - `python main.py --data Data.txt --out .tmp_check_res_csr_block.txt --mode csr_block --K 8 --dtype float32`

### C 已接入并完成的点

- `sweep.py` 可直接调 CLI 跑参数扫描并输出 CSV / 结果目录
- `analyze_dataset.py` 可输出数据统计 JSON
- `plot.py` 可消费 summary 并生成图表
- 报告侧已有第 1、5、7 章草稿与图表资源，进入最终统稿阶段

## 9. 实验数据摘要

环境口径：

- 平台：Windows 开发机
- Python：`micromamba test`
- 参数：`beta = 0.85`，`eps = 1e-8`
- 统计方式：每组配置跑 3 次，取平均 `peak_rss_mb` 与平均 `wall_sec`

### E3：K 变化

| 配置 | 平均峰值 RSS / MB | 平均时间 / s | 收敛轮数 |
| --- | ---: | ---: | ---: |
| `csr_block, K=4, float32` | `35.16` | `0.376` | `15` |
| `csr_block, K=8, float32` | `35.99` | `0.423` | `15` |
| `csr_block, K=16, float32` | `34.97` | `0.457` | `15` |

结论：

- 当前数据规模下，`K` 增大并没有继续明显压低 RSS
- `K=4` 最快，`K=16` 最慢
- 作为主提交默认值，`K=8` 仍可接受，且与接口示例一致

### E4：CSR vs CSR+Block

| 配置 | 平均峰值 RSS / MB | 平均时间 / s | 收敛轮数 |
| --- | ---: | ---: | ---: |
| `csr, K=8, float32` | `33.42` | `0.492` | `15` |
| `csr_block, K=8, float32` | `35.89` | `0.418` | `15` |

结论：

- 当前实现下，`csr_block` 比纯 `csr` 更快
- 纯 `csr` 的 RSS 略低
- 主提交版本仍建议使用 `csr_block`

### E7：dtype 对比

| 配置 | 平均峰值 RSS / MB | 平均时间 / s | 收敛轮数 |
| --- | ---: | ---: | ---: |
| `csr_block, K=8, float32` | `35.92` | `0.403` | `15` |
| `csr_block, K=8, float64` | `36.27` | `0.366` | `12` |

结论：

- 两者 `top10_signature` 一致
- `float64` 当前机器上因为轮数更少，平均时间略短
- 但为了保留更大的内存安全余量，主提交仍建议固定 `float32`

## 10. 队长视角的团队状态

### A

- 当前状态：`已接入 / 已完成主交付`
- 风险：
  - `dump_top_k` 目前通过 `_LAST_NODE_IDS` 恢复原始 ID，后续若并发多图需改显式映射
  - 继续保持与 `INTERFACE.md` 冻结公式一致，避免后续“优化”引入行为漂移
- 后续协作重点：
  - 配合最终打包前复测
  - 配合统一报告口径与实验数字

### C

- 当前状态：`已接入 / 已完成数据脚本主链路`
- 风险：
  - 需确保报告中的数字都可追溯到脚本产物（CSV/JSON），避免手工改图
  - 最终统稿时需统一 A/B/C 三份 handover 的指标口径
- 后续协作重点：
  - 总报告润色与章节拼接
  - 最终提交附件组织与校验

### B（当前我这边）

- 当前状态：`核心模块与工程基建已就绪，进入收尾验收`
- 已完成：接口、工程规范、分块实现、benchmark、测试、报告草稿、交接文档、A/C 集成复核
- 剩余动作：Windows 打包验收、全仓最终复测、提交包冻结

## 11. 风险清单

按当前复核日期 `2026-04-25` 估算，距离 DDL `2026-05-15 12:00 UTC+8` 约还有 **20 天**（精确值随当前时刻变化）。

### 风险 1：打包版与源码版行为不一致

- 表现：exe 在路径、编码、依赖裁剪后输出与源码版有差异
- 影响：最终提交可执行文件不稳定，评分风险高
- 应对：
  - 打包后强制对拍 `top10_signature / iters / delta`
  - 保留对拍日志并纳入最终交接附件

### 风险 2：报告指标口径不统一

- 表现：`HANDOVER.md`、`HANDOVER_A.md`、`HANDOVER_C.md` 与报告正文数值不一致
- 影响：答辩与复查时可追溯性不足
- 应对：
  - 统一以最新脚本复跑结果为唯一数据源
  - 统一更新所有 handover 与报告表格

### 风险 3：最后阶段改参数未复测

- 表现：临近提交调整 `K / dtype / eps` 后未全链路重跑
- 影响：`Res.txt`、图表、结论出现互相矛盾
- 应对：
  - 建立“参数变更即复测”的提交前检查表
  - 提交包仅使用最后一次冻结参数生成的产物

## 12. 接手建议

如果你现在接手整个工程，我建议按以下顺序继续：

1. 先读 `INTERFACE.md`
2. 跑 `pytest -q`
3. 跑一次 `python main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32`
4. 看 `benchmark.py` 和 `experiments/` 里的原始数据
5. 完成 Windows 打包对拍，冻结最终提交文件清单

只要冻结接口不被破坏并按清单复测，当前仓库已经可以进入最终提交阶段。
