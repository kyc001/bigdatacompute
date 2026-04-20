# 第 6 章 工程实现与部署

## 6.1 工程目标与设计原则

本项目并不是单纯完成一个能跑的 PageRank 脚本，而是要交付一套便于团队协作、便于实验复现、便于最终打包提交的工程。为此，本章关注的问题不再是“公式是否正确”，而是“代码能否被三位成员并行维护、实验能否被脚本化复现、最后能否被无 Python 环境的机器直接执行”。

工程实现阶段遵循四条原则。第一，接口先行。队长 B 在开发最早期冻结 `INTERFACE.md`，把 A 的算法接口、B 的分块接口、C 的实验调用方式全部钉死，降低后续并行开发时的返工概率。第二，主程序最小化。`main.py` 只负责 CLI 解析、调度、结果输出和 JSON 摘要，不把实验逻辑、画图逻辑或临时调试逻辑混进主入口。第三，开发与提交分离。`benchmark.py`、`pytest`、`pyinstaller` 都属于开发期工具链，不应污染最终提交包的最小运行路径。第四，所有实验结果必须可追溯。C 的 `sweep.py` 和 `plot.py` 只能基于 CLI 与 CSV 工作，不能直接 import 内部函数，否则一旦内部实现变化，实验脚本会同步失稳。

## 6.2 项目结构设计

按照当前分工，仓库被组织为以下几类文件。

- 契约与协作文档：
  - `INTERFACE.md`
  - `README.md`
  - `CONTRIBUTING.md`
  - `code_review_checklist.md`
  - `HANDOVER.md`
- 核心实现：
  - `main.py`
  - `blocks.py`
  - `mock_graph.py`
- 工具链：
  - `benchmark.py`
  - `requirements.txt`
  - `compile-parameter.txt`
  - `cross_platform_checklist.md`
- 质量保障：
  - `tests/`
- 报告草稿：
  - `report_ch4.md`
  - `report_ch6.md`

这种结构的好处是职责边界清晰。A 可以只关注 `load_graph / power_iteration / dump_top_k`；B 可以集中维护分块逻辑、benchmark 与打包；C 则只需要面向 CLI 和 CSV 工作，而不必深入主程序内部实现。对于课程项目而言，这种分层可以显著降低“一个人临时改了函数签名，另外两个人的脚本全挂掉”的风险。

## 6.3 CLI 与实验自动化

为了让 C 的批量实验脚本具备稳定调用点，`main.py` 的命令行被固定为：

```bash
python main.py --data Data.txt --beta 0.85 --eps 1e-8 --out Res.txt --mode csr_block --K 8 --dtype float32
```

其中 `--mode` 统一保留 `dense / csr / block / csr_block` 四个入口，含义分别对应小图对照、纯 CSR、纯分块、以及主提交版本 `CSR + Block`。即使某些模式只用于开发或 mock，对外接口也保持稳定，这样 C 的 `sweep.py` 就只需枚举参数组合，而不需要知道实现细节。

更关键的是，`main.py` 最后一行 stdout 固定输出 JSON，例如：

```json
{
  "peak_rss_mb": 35.6,
  "wall_sec": 0.38,
  "iters": 15,
  "mode": "csr_block",
  "K": 8,
  "dtype": "float32",
  "top10_signature": "75,8686,..."
}
```

这样，`benchmark.py` 不需要解析人类可读日志，也不需要 import 主程序内部对象，只需读取最后一行 JSON 即可完成 `wall_sec / iters / top10_signature` 的采集。与之配套，benchmark 的 CSV schema 也被冻结为：

```text
run_id,mode,K,dtype,peak_rss_mb,wall_sec,iters,top10_signature
```

这让实验自动化链路形成闭环：`main.py` 产出 JSON，`benchmark.py` 产出 CSV，C 的画图脚本只消费 CSV。

## 6.4 依赖管理与开发环境

课程项目最容易出问题的地方之一是“同一份代码在不同成员机器上行为不一致”。为降低这种风险，本项目用 `requirements.txt` 固定开发依赖：

- `numpy`
- `scipy`
- `psutil`
- `pytest`
- `pyinstaller`

其中要强调两点。第一，`scipy` 是给 A 的 CSR 相关实现预留的，主流程禁止直接 `import scipy`，只能按契约使用 `from scipy.sparse import csr_matrix`。第二，`benchmark.py` 依赖 `psutil`，但它属于开发工具，不进入提交时的算法说明。换句话说，依赖管理要区分“算法允许使用的库”和“工程工具链需要的库”，二者不能混为一谈。

Windows 开发阶段建议统一使用同一套虚拟环境，例如 `micromamba activate test`。所有测试、benchmark 和打包命令都在同一环境内执行，避免出现“pytest 用的是一个 Python，PyInstaller 用的是另一个 Python”的隐蔽问题。

## 6.5 质量保障：mock、pytest 与 code review

工程质量保障由三层组成。

第一层是 mock。`mock_graph.py` 既提供 5 节点手算图，也提供 100 节点、500 边随机图。前者用于验证 dead-end 补偿公式和逐轮数值是否正确，后者用于验证分块版与朴素 dense reference 的最终收敛结果是否一致。

第二层是自动化测试。当前 `tests/` 至少覆盖以下场景：

- 小图手算值与最终 block / csr 结果一致
- 100 节点随机图的分块结果与朴素 reference 一致
- `K ∈ {1, 4, 8, 16}` 时结果稳定
- 全 dead-end 病态图仍能收敛到均匀分布
- spider-trap 图能正常收敛
- benchmark 能解析 stdout JSON 并写出固定 schema CSV

第三层是人工审查。队长提供 `code_review_checklist.md`，分别列出对 A 的算法代码和对 C 的脚本代码的审查要点。这样做的价值在于：即使 reviewer 不是该模块作者，也可以按清单快速定位高风险点，比如 dead-end 公式是否一致、排序规则是否与 `Res.txt` 契约一致、CSV 列是否被脚本改坏等。

## 6.6 Git 协作规范

为防止多人开发时相互覆盖，本项目采用三层分支模型：

- `main`：稳定、可提交、可打包版本
- `dev`：团队日常集成分支
- `feature-*`：各成员功能分支

提交信息统一采用：

```text
<type>(<scope>): <summary>
```

例如 `feat(block): add memmap-based iterate_by_block`。这一规范看似琐碎，但对课程项目非常重要，因为最终整理实验与报告时，队长需要快速回溯某组结果对应的是哪一版实现。若提交信息过于随意，如“改了一点”“试试看”，后期几乎无法做责任归因与版本定位。

PR 流程也被要求写清目标、改动、风险、验证、待办五项内容。这样做的目的不是模仿工业界流程本身，而是通过模板逼迫每次改动都留下最小可追溯记录，避免临近 DDL 时“谁改过这里已经说不清”的情况。

## 6.7 PyInstaller 打包流程

本项目最终要求在无 Python 环境的机器上运行，因此需要将 `main.py` 打成单文件可执行程序。打包命令单独写入 `compile-parameter.txt`，并显式排除无关模块，例如 `matplotlib`、`tkinter`、`PIL`、`pytest`，以减小打包体积并减少 hook 干扰。

推荐流程如下：

1. 先在开发环境中运行：

```bash
python main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32
```

确认结果文件格式与 stdout JSON 都正常。

2. 再执行：

```bash
pyinstaller --noconfirm --clean --onefile --name pagerank main.py ...
```

3. 进入 `dist/` 目录运行生成的可执行文件，复测一遍 `Data.txt`。

4. 检查：
   - `Res.txt` 是否生成
   - Top-10 签名是否与源码版一致
   - JSON 中的 `iters` 是否一致
   - 运行时间是否在接受范围内

之所以先跑源码版再打包，是因为 PyInstaller 的问题通常不在算法，而在依赖收集、路径解析、临时目录权限和编码。若直接从打包版本排查，定位成本会高很多。

## 6.8 Windows 为主的验证清单

结合当前项目安排，真实验收平台以 Windows 为主。因此本工程交付的重点不是“在本仓库里生成所有平台产物”，而是把 Windows 下的脚本、命令和检查项写清楚，保证队长可以在自己的机器上重复执行。对应步骤包括：

- 安装依赖
- 运行 pytest
- 跑 `main.py`
- 跑 `benchmark.py`
- 执行 PyInstaller
- 运行打包后的 exe
- 对比 Top-10 签名

Linux 侧仅保留 checklist，不作为当前交付的主验证对象。这样既满足课程对跨平台可移植性的说明要求，又不会让时间被非关键路径的环境问题消耗掉。

<!-- 图6.1：从接口冻结到实验 CSV 再到打包产物的工程流水线 -->
<!-- 图6.2：main.py / benchmark.py / tests / PyInstaller 的关系图 -->

## 6.9 本章结论

本章的核心结论是：课程作业也需要最小可维护工程。`INTERFACE.md` 负责冻结协作边界，`README + CONTRIBUTING + code_review_checklist` 负责约束团队行为，`benchmark.py + CSV schema` 负责保证实验可复现，`PyInstaller + compile-parameter.txt + checklist` 负责保证最终交付可执行。对本项目而言，真正重要的不只是“把 PageRank 算出来”，更是“让这套实现可以被 A、B、C 三人拆开协作，最后又能稳定合到一起”。这也是第 6 章相对于算法章节最重要的工程价值。
