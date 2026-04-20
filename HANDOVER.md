# HANDOVER.md

本文档以“队长 B 写给未来要接手或复查整个工程的人”的口吻整理当前状态。请先读 `INTERFACE.md`，再读本文件。

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
- pytest 回归测试已补到 6 条，当前全部通过
- E3 / E4 / E7 已在 Windows 开发环境下跑出一版数据摘要
- 第 4 章与第 6 章 Markdown 草稿已写完
- 打包命令与跨平台检查清单已补齐

### 进行中

- A 的真实 `load_graph / power_iteration / dump_top_k` 还未接入，目前 `main.py` 里是 B 的兼容实现
- C 的 `sweep.py / plot.py / analyze_dataset.py` 尚未接入当前 CLI / CSV 契约
- 报告最终文字润色、图表替换、总稿拼接还没开始

### 待开始

- 队长本人在 Windows 上手动跑最终 PyInstaller 验收
- A 完成 E1 / E2 / E8，并把算法章节并入报告
- C 完成 E5 / E6、图表生成、报告统稿与 zip 打包

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
| `main.py` | 主入口、兼容版 `load_graph / power_iteration / dump_top_k` | `v0.1` |
| `benchmark.py` | RSS / wall time 采样与 CSV 输出 | `v0.1` |
| `tests/conftest.py` | pytest 导入路径配置 | `v0.1` |
| `tests/test_pagerank_behaviors.py` | 6 条核心回归测试 | `v0.1` |
| `report_ch4.md` | 第 4 章草稿：内存优化 | `v0.1-draft` |
| `report_ch6.md` | 第 6 章草稿：工程实现与部署 | `v0.1-draft` |
| `compile-parameter.txt` | PyInstaller 打包命令 | `v0.1` |
| `cross_platform_checklist.md` | Windows 主验收 + Linux 备用检查清单 | `v0.1` |
| `HANDOVER.md` | 本交接文档 | `v0.1` |

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

当前兼容实现还会额外输出：

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
pyinstaller --noconfirm --clean --onefile --name pagerank main.py --exclude-module matplotlib --exclude-module tkinter --exclude-module PIL --exclude-module pytest
```

### 5.7 运行打包产物

```powershell
.\dist\pagerank.exe --data Data.txt --out Res_packaged.txt --mode csr_block --K 8 --dtype float32
```

### 5.8 对拍源码版与打包版

- 对比 `Res.txt` 与 `Res_packaged.txt` 的 Top-10 节点顺序
- 对比 stdout JSON 中的 `iters`
- 对比 `top10_signature`

## 6. 已知问题与权衡

1. 当前 `main.py` 里仍是 B 的兼容版 `load_graph / power_iteration / dump_top_k`，不是 A 的最终实现。
2. `build_blocks(edges_array, K, tmp_dir)` 在输入是纯边数组且图中存在“只作为 dead-end、完全不出现在边表中的节点”时，无法自动推断完整 `N`；这类场景必须按契约传 `(edges, N)`。
3. `dense` 模式只用于 mock / 小图 / 对照实验，不适合作为真实大图主提交流程。
4. 当前实验数据全部来自 Windows 开发环境的 3 次重复平均；如果换机器，绝对时间与 RSS 会变化，但相对趋势应保持。
5. Linux 侧只保留 checklist，没有纳入当前优先验收范围。

## 7. 下一步建议

1. 让 A 先按 `INTERFACE.md` 对接真实 `load_graph / power_iteration / dump_top_k`，并保留当前 pytest 全量回归。
   - 预估工时：`0.5~1 天`
2. 让 C 用冻结的 CLI / CSV schema 写通 `sweep.py` 与 `plot.py`，优先吃 benchmark 输出而不是直接 import 内部函数。
   - 预估工时：`0.5 天`
3. 队长本人在 Windows 上手动执行最终打包与验收清单，确认源码版与 exe 版 `top10_signature` 一致。
   - 预估工时：`0.5 天`
4. 把 E3 / E4 / E7 的 CSV 交给 C 画图，并把 `report_ch4.md` / `report_ch6.md` 的占位图替换成正式图片。
   - 预估工时：`0.5~1 天`
5. 视 A 的实现情况决定是否把 `block` 与 `csr_block` 模式做进一步职责拆分；若不拆，也至少在 README 里说明当前二者等价路径。
   - 预估工时：`2~3 小时`

## 8. 与 A、C 的集成点

### 需要 A 接入的地方

- 用 A 的真实 `load_graph` 替换 `main.py` 当前兼容实现
- 用 A 的真实 `power_iteration` 替换 `main.py` 当前纯 CSR 兼容实现
- 用 A 的真实 `dump_top_k` 接管结果落盘与 `top10_signature`
- A 接入后必须重新跑：
  - `pytest -q`
  - `python main.py --data Data.txt --out Res.txt --mode csr`
  - `python main.py --data Data.txt --out Res.txt --mode csr_block --K 8`

### 需要 C 接入的地方

- `sweep.py` 只调用 CLI，不要 import `main.py`
- `plot.py` 只消费 benchmark CSV
- 需要 C 接手的产物：
  - E3 / E4 / E7 图表
  - 报告第 1、5、7 章
  - 最终 PDF / zip 组织

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

- 当前状态：`待开始 / 待接入`
- 风险：
  - 如果 A 的 `power_iteration` 没按冻结 dead-end 公式实现，结果会与 B 的 block 版不一致
  - 如果 `dump_top_k` 排序规则和 10 位小数格式不一致，会直接破坏 benchmark 与结果对拍
- 我对 A 的期待：
  - 先按签名实现
  - 先过 pytest
  - 再做 E1 / E2 / E8

### C

- 当前状态：`待开始`
- 风险：
  - 如果 C 直接 import 内部函数而不是走 CLI，后续一旦重构，实验脚本会集体失效
  - 如果图表不是从 CSV 自动生成，报告数字容易失去可追溯性
- 我对 C 的期待：
  - 先写 `sweep.py`
  - 再写 `plot.py`
  - 最后接管报告统稿与打包

### B（当前我这边）

- 当前状态：`核心模块与工程基建已就绪`
- 已完成：接口、工程规范、分块实现、benchmark、测试、报告草稿、交接文档
- 剩余动作：等待 A / C 接入后做一次最终总装联调

## 11. 风险清单

按当前会话日期 `2026-04-20` 估算，距离 DDL `2026-05-15 12:00 UTC+8` 约还有 `25` 天左右；若按 `2026-04-20 00:00` 起算，则约为 `25 天 12 小时`。

### 风险 1：A 的实现与冻结接口偏离

- 表现：签名不一致、dead-end 公式不一致、`dump_top_k` 输出格式不一致
- 影响：B 的 block 版与 A 的 csr 版无法对拍，C 的实验脚本也会失效
- 应对：
  - A 合并前强制跑 pytest
  - 代码审查严格对照 `INTERFACE.md`

### 风险 2：临近 DDL 才做打包验收

- 表现：PyInstaller 缺 hook、路径问题、exe 运行但结果文件格式不对
- 影响：最终提交版本可能无法在无 Python 环境机器上通过
- 应对：
  - 至少提前一周在 Windows 上跑完整清单
  - 用 `top10_signature` 做源码版与打包版对拍

### 风险 3：C 的实验数据与源码版本脱节

- 表现：报告里的数字来自旧 CSV 或手工拷贝
- 影响：报告与代码不一致，返工成本高
- 应对：
  - 统一只以 benchmark CSV 为数据源
  - 画图脚本不允许手填数字

## 12. 接手建议

如果你现在接手整个工程，我建议按以下顺序继续：

1. 先读 `INTERFACE.md`
2. 跑 `pytest -q`
3. 跑一次 `python main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32`
4. 看 `benchmark.py` 和 `experiments/` 里的原始数据
5. 再决定是先催 A 对接，还是先让 C 接走实验可视化

只要冻结接口不被破坏，当前 B 侧产物已经足够支撑后续团队并行推进。
