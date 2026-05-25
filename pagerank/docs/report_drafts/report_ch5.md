# 第 5 章 实验结果与分析

## 5.1 实验设计与数据来源

本章汇总团队最终阶段已经完成的实验数据。所有图表均由脚本从 CSV/JSON 生成，不手工填写图中数据。统一数据来源为 `experiments/report_summary.json`。

- 数据集分析：`experiments/dataset_stats.json` 与 `experiments/degree_distribution.csv`
- E1、E3、E4、E7、E8：`experiments/E1_dense.csv`、`experiments/E3.csv`、`experiments/E4.csv`、`experiments/E7.csv`、`experiments/E8.csv`
- E5、E6：C 使用 `sweep.py` 通过 `main.py` CLI 实测得到的 `experiments/sweep_results.csv`
- 图表输出：`report/fig/*.png`

## 5.2 数据集分析结果

数据集共有 `10000` 个节点、`150000` 条边，平均入度与平均出度均为 `15.00`。最大出度为 `36`，最大入度为 `32`，说明图整体较稀疏，且不存在极端高连接度的超级节点。若采用稠密矩阵，空间复杂度为 `O(N^2)`，对一万节点图会形成数百 MB 的基础内存占用；而 CSR 将图结构压缩为 `row_ptr / col_idx / out_deg` 三类数组，空间复杂度降为 `O(N + M)`。

数据集中 dead-end 节点数为 `1500`，占比 `15.00%`。这是算法实现中最需要关注的结构特征之一。如果不补偿这些节点持有的概率质量，PageRank 向量的总和会逐轮下降，最终得到的排名不再对应合法概率分布。当前主程序和分块版本均采用 `INTERFACE.md` 冻结的质量补偿公式。

弱连通分量共有 `501` 个，最大弱连通分量包含 `9500` 个节点，占比 `95.00%`，另有 `500` 个孤立节点。这说明图主体结构集中，Top 排名大概率来自最大弱连通分量；孤立节点和小分量则主要依赖 teleport 与 dead-end 补偿获得基础分数。

## 5.3 E3：块数 K 对内存与时间的影响

B 对 `csr_block` 模式在 `K = 4 / 8 / 16` 下各运行 3 次。平均结果如下：

| K | 平均峰值 RSS / MB | 平均时间 / s | 迭代轮数 |
| ---: | ---: | ---: | ---: |
| 4 | 35.09 | 0.704 | 15 |
| 8 | 34.98 | 0.729 | 15 |
| 16 | 34.92 | 0.856 | 15 |

图 `report/fig/k_memory_time.png` 展示了块数与内存、时间之间的权衡。当前数据规模下，增大 `K` 并没有显著降低内存峰值，`K=16` 的 RSS 只比 `K=8` 略低，但耗时更高。原因在于本数据集只有十五万条边，Python 解释器、NumPy 运行时和进程采样开销已经占据相当比例；当图本身不大时，过细分块会增加文件切换和块循环成本。

数据源：`experiments/report_summary.json`。

因此，当前阶段可保留 `K=8` 作为主提交默认值：它与 `INTERFACE.md` 示例一致，运行时间与内存都处于安全范围内，也为更大图保留了分块扩展能力。

## 5.4 E1/E2/E4：稠密、CSR 与 CSR+Block 对比

`report_summary.json` 汇总了稠密基线、纯 CSR 与 `csr_block` 三种实现的最终平均结果：

| 模式 | 平均峰值 RSS / MB | 平均时间 / s | 迭代轮数 |
| --- | ---: | ---: | ---: |
| `dense` | 412.64 | 0.602 | 14 |
| `csr` | 34.36 | 0.726 | 15 |
| `csr_block` | 34.66 | 0.838 | 15 |

从内存角度看，CSR 相对 dense 的峰值 RSS 从 `412.64 MB` 降至 `34.36 MB`，下降约 `91.67%`，这是稀疏化带来的核心收益。`csr_block` 的 RSS 与纯 CSR 接近，说明在当前数据规模下，分块更多体现为可扩展的工程约束，而不是进一步压低当前小图 RSS。分块的主要价值在于把活动更新区限制在目标块内，当边数进一步增大时，内存上界会比普通 CSR 迭代更可控。

数据源：`experiments/report_summary.json`。

## 5.5 E5：阻尼系数 beta 对 Top-10 排名的影响

C 使用 `sweep.py` 扫描 `beta ∈ {0.70, 0.80, 0.85, 0.90, 0.95}`，固定 `eps=1e-8`、`mode=csr_block`、`K=8`、`dtype=float32`。以 `beta=0.85` 的 Top-10 为基准，计算 Jaccard 相似度和交集上的 Kendall tau，结果如下：

| beta | 迭代轮数 | Top-10 Jaccard | Kendall tau | Top-10 签名 |
| ---: | ---: | ---: | ---: | --- |
| 0.70 | 13 | 0.818 | 0.833 | 见 `experiments/sweep_results.csv` |
| 0.80 | 15 | 1.000 | 0.956 | 见 `experiments/sweep_results.csv` |
| 0.85 | 15 | 1.000 | 1.000 | `75,8686,9678,5104,725,3257,468,7730,7175,5526` |
| 0.90 | 16 | 1.000 | 0.956 | 见 `experiments/sweep_results.csv` |
| 0.95 | 16 | 0.818 | 0.944 | 见 `experiments/sweep_results.csv` |

图 `report/fig/beta_top10_similarity.png` 显示，`beta` 在 `0.80` 到 `0.90` 区间时 Top-10 集合非常稳定，仅发生局部顺序变化；当 `beta=0.70` 或 `0.95` 时，Top-10 集合出现 1 个节点替换。直观解释是：较低 beta 增强随机跳转项，削弱链接结构；较高 beta 强化沿边传播，让图结构局部差异更明显。当前数据集中前几名节点较稳定，说明高排名节点有较强结构优势。

数据源：`experiments/report_summary.json` 与 `experiments/sweep_results.csv`。

## 5.6 E6：收敛阈值 eps 对结果与迭代轮数的影响

C 使用 `sweep.py` 扫描 `eps ∈ {1e-6, 1e-8, 1e-10}`，固定 `beta=0.85`、`mode=csr_block`、`K=8`、`dtype=float32`。结果如下：

| eps | 迭代轮数 | 最终 L1 delta | Top-10 Jaccard |
| ---: | ---: | ---: | ---: |
| `1e-6` | 9 | `3.503e-7` | 1.000 |
| `1e-8` | 15 | `7.170e-9` | 1.000 |
| `1e-10` | 27 | `8.004e-11` | 1.000 |

图 `report/fig/epsilon_residual_summary.png` 展示了阈值变严格后迭代轮数增加的趋势。`eps=1e-6` 到 `1e-8` 时，轮数从 9 增至 15；进一步到 `1e-10` 时，轮数增至 27。三组实验的 Top-10 集合和顺序均与 `eps=1e-8` 保持一致，说明在本数据集上，Top-10 排名对收敛阈值不敏感。

需要说明的是，当前 `INTERFACE.md` 只要求 `main.py` 在 stdout JSON 中输出最终 `delta`，没有暴露逐轮 residual 历史。因此本阶段图表是“最终残差与迭代轮数汇总”，不是完整逐轮残差曲线。若最终报告必须展示逐轮曲线，需要 A/B 在不破坏 CLI 的前提下增加可选 residual trace 输出，再由 `plot.py` 读取真实 trace CSV 生成曲线。

数据源：`experiments/report_summary.json`。

## 5.7 E7：dtype 对比

B 的 E7 对比了 `float32` 与 `float64`：

| dtype | 平均峰值 RSS / MB | 平均时间 / s | 迭代轮数 |
| --- | ---: | ---: | ---: |
| `float32` | 34.22 | 0.789 | 15 |
| `float64` | 35.34 | 0.719 | 12 |

当前机器上 `float64` 因收敛轮数较少，平均时间略低；但它的内存峰值高于 `float32`。考虑作业评分重点之一是内存峰值，并且最终提交要求在不同机器上留出安全余量，主提交版本仍建议固定 `float32`。从 Top-10 签名看，两种 dtype 当前没有造成可见排名差异。

数据源：`experiments/report_summary.json`。

## 5.8 E8：dead-end 策略对比

E8 对比了质量补偿、忽略 dead-end、删除 dead-end 三种策略：

| 策略 | 时间 / s | 迭代轮数 | rank_sum | Top-10 Jaccard |
| --- | ---: | ---: | ---: | ---: |
| `compensation` | 0.580 | 15 | 1.000 | 1.000 |
| `ignore` | 1.570 | 60 | 0.602 | 1.000 |
| `delete` | 0.499 | 17 | 1.000 | 0.667 |

忽略 dead-end 时，最终 `rank_sum` 只有 `0.602`，说明概率质量明显泄漏，已经不是合法概率分布。删除 dead-end 虽然保持了概率和，但 Top-10 与补偿法的 Jaccard 只有 `0.667`，说明它改变了原图语义。质量补偿法同时保持节点全集和概率守恒，因此是最终实现采用的策略。

数据源：`experiments/report_summary.json`。

## 5.9 结论

综合当前真实数据，可以得到四点结论。第一，数据集稀疏性明显，CSR 是满足内存限制的必要选择。第二，dead-end 节点占比达到 `15.00%`，必须按统一公式做质量补偿。第三，`beta` 对 Top-10 集合有一定影响，但核心高排名节点在 `0.80` 到 `0.90` 区间保持稳定。第四，收敛阈值越严格，迭代轮数明显增加，但 Top-10 排名在本数据集上保持稳定。

最终提交仍需由人工在打包版上复核源码版与 exe 版的 Top-10 签名一致性。
