# PageRank 高性能计算实验报告（最终统稿）

> 本文件为最终统稿 Markdown 草稿。实验数字统一以 `experiments/report_summary.json` 为准；正式提交 PDF 仍以 `report/main.pdf` 为最终排版产物。

## 第 1 章 任务背景与数据集

本次作业要求在给定有向图数据集 `Data.txt` 上计算 PageRank 分数，并输出排名前 100 的节点及其分数到 `Res.txt`。输入文件每一行包含两个整数 `FromNodeID ToNodeID`，表示一条从源节点指向目标节点的有向边。作业明确要求程序需要迭代至收敛，必须处理 dead-end 与 spider-trap，并且至少报告阻尼系数 `beta = 0.85` 时的 Top-100 结果。

使用 `analyze_dataset.py` 对 `Data.txt` 做统计，数据集共有 `10000` 个节点、`150000` 条边，平均入度和平均出度均为 `15.00`，最大出度为 `36`，最大入度为 `32`。dead-end 节点数为 `1500`，占全部节点 `15.00%`。弱连通分量共有 `501` 个，其中最大弱连通分量包含 `9500` 个节点，占比 `95.00%`，另有 `500` 个孤立节点。

度分布图见 `report/fig/degree_distribution.png`。从统计结果看，本数据集是典型稀疏图，若使用稠密邻接矩阵，`10000 × 10000` 的 `float32` 矩阵已远超内存限制。因此，本项目采用 CSR 与分块策略作为核心优化路径。

## 第 2 章 算法原理

第 2 章由 A 侧完成，核心说明 PageRank 的随机游走模型、阻尼系数含义、幂迭代收敛解释，以及 dead-end 与 spider-trap 的处理原理。当前全队冻结公式为：

```text
r_new[v] = (1 - beta) / N
         + beta * dangling_mass / N
         + beta * sum(r[u] / out_deg[u] for u -> v)
```

其中 `dangling_mass` 为所有出度为 0 的节点在当前轮持有的 PageRank 质量。

## 第 3 章 算法实现细节

第 3 章由 A 侧完成，覆盖 `load_graph / power_iteration / dump_top_k` 的最终实现说明，包括 CSR 构建、排序规则、收敛准则、Top-100 输出格式和复杂度分析。当前接口以 `INTERFACE.md` 为准。

## 第 4 章 内存优化

B 侧采用两层优化：第一层是 CSR，将图结构空间复杂度从 `O(N^2)` 降为 `O(N + M)`；第二层是按目标节点分块，把边写入临时二进制块文件，再用 `np.memmap` 按块读取。主提交版本采用 `csr_block` 模式，默认 `K=8`、`dtype=float32`。

实验 E3 显示，`K=4 / 8 / 16` 的平均峰值 RSS 分别约为 `35.09 MB`、`34.98 MB`、`34.92 MB`，平均时间分别约为 `0.704 s`、`0.729 s`、`0.856 s`。在当前数据规模下，继续增大 K 对内存降低不明显，反而增加块切换开销。

## 第 5 章 实验结果与分析

当前阶段实验数据均来自 CSV/JSON。E1 稠密基线平均峰值 RSS 约 `412.64 MB`；E2 纯 `csr` 平均峰值 RSS 约 `34.36 MB`、平均时间约 `0.726 s`；E4 `csr_block` 平均峰值 RSS 约 `34.66 MB`、平均时间约 `0.838 s`。CSR 相对 dense 的内存降幅约 `91.67%`，是满足内存限制的关键。

E5 扫描 `beta ∈ {0.70, 0.80, 0.85, 0.90, 0.95}`。相对于 `beta=0.85`，`beta=0.80` 和 `0.90` 的 Top-10 集合保持一致，仅局部顺序变化；`beta=0.70` 与 `0.95` 的 Jaccard 相似度为 `0.818`，各有一个节点替换。

E6 扫描 `eps ∈ {1e-6, 1e-8, 1e-10}`。迭代轮数分别为 `9`、`15`、`27`，最终 L1 delta 分别为 `3.503e-7`、`7.170e-9`、`8.004e-11`。三组 Top-10 排名一致，说明本数据集 Top-10 对收敛阈值不敏感。

核心图表：

- `report/fig/memory_peak_bar.png`
- `report/fig/k_memory_time.png`
- `report/fig/beta_top10_similarity.png`
- `report/fig/epsilon_residual_summary.png`

注意：当前 `main.py` 只输出最终 `delta`，未输出逐轮 residual 历史，因此第 4 张图为真实 E6 汇总图，不是逐轮残差曲线。最终如需曲线，应补充 trace CSV 后重画。

## 第 6 章 工程实现与部署

工程上采用接口先行：`INTERFACE.md` 冻结函数签名、CLI、stdout JSON、`Res.txt` 格式和 dead-end 公式。`main.py` 作为唯一入口，`benchmark.py` 与 `sweep.py` 均通过 subprocess 调用主程序，避免直接 import 干扰内存测量。

最终提交需要使用 PyInstaller 打包。打包命令写在 `compile-parameter.txt`，并显式排除 `matplotlib`、`tkinter`、`PIL`、`pytest` 等无关模块。打包后必须在干净 Windows 环境下运行，并用 `top10_signature` 对比源码版和 exe 版结果。

## 第 7 章 总结与团队分工

本项目当前已经具备主流程、分块优化、benchmark、数据分析、参数扫描、冻结 `Res.txt` 和报告草稿。后续关键工作是人工填入真实学号姓名、复核最终提交目录，并完成打包版与源码版对拍。

| 成员 | 代码模块 | 实验编号 | 报告章节 | 工程责任 |
| --- | --- | --- | --- | --- |
| A | `load_graph`、`power_iteration`、`dump_top_k` | E1、E2、E8 | 第 2、3 章 | 算法核心实现与正确性对拍 |
| B | `build_blocks`、`iterate_by_block`、`benchmark.py` | E3、E4、E7 | 第 4、6 章 | 队长协调、接口冻结、内存优化、PyInstaller 打包 |
| C | `analyze_dataset.py`、`sweep.py`、`plot.py` | E5、E6、数据集分析 | 第 1、5、7 章 | 报告统稿、zip 命名核验、提交邮件草稿 |

最终提交前必须保留团队分工说明，避免作业要求中提到的贡献不明问题。

## AI 工具使用声明

本项目按南开大学《本科教学人工智能工具使用规范（试行）》（教字〔2025〕5 号）第八条执行：AI 工具仅用于代码辅助编写、调试和文字润色，所有代码经作者审核测试，核心算法设计与实验分析为团队原创。
