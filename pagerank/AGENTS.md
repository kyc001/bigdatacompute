你是一个资深 Python 工程师，正在协助我完成一个高性能计算课程作业的 A 角色工作（队内只负责算法核心，不兼队长；队长由 B 担任）。
请严格按下面的要求推进，完成后输出一份完整的 HANDOVER_A.md 供我交接给下一个人。

## 一、项目背景

- 作业：在给定图数据集 Data.txt（每行 "FromNodeID ToNodeID"）上计算 PageRank，输出 Top-100 节点到 Res.txt。
- 强制要求：
  - 语言：Python（入口文件必须命名为 main.py）。
  - 内存峰值 ≤ 80 MB（目标 ≤ 65 MB）。
  - 运行时间 ≤ 60 秒（目标 ≤ 30 秒）。
  - 必须实现稀疏矩阵（CSR）+ 分块矩阵（Block Matrix）两种内存优化。
  - 正确处理 dead-ends（悬挂节点）和 spider-traps。
  - 迭代直到收敛，ε = 1e-8，β = 0.85 为必报场景。
  - 禁止调用现成 PageRank API（networkx.pagerank、igraph.pagerank、scipy.sparse.linalg.eigs、graph-tool 等）。
- 最终用 PyInstaller 打成单文件可执行程序。

## 二、我负责的 A 角色任务（纯算法核心）

1. **核心代码模块**（main.py 的算法主体）：
   - load_graph(path)：读 Data.txt，构造 CSR（row_ptr / col_idx / out_deg / N），处理节点 ID 非连续的情况（做 ID 映射）。
   - power_iteration(row_ptr, col_idx, out_deg, N, beta, eps)：幂迭代，带 dead-end 质量补偿，返回 float32 的 r 向量。
   - dump_top_k(r, k, out_path, id_map)：写 Top-100 到 Res.txt，格式 "NodeID Score"，Score 保留 10 位小数。
   - baseline_dense.py：稠密版本（仅用于 E1 基线对比，不提交）。
2. **实验**（3 个）：
   - E1：稠密基线（建立对照）。
   - E2：稀疏 CSR 版本（对比 E1 的内存降幅）。
   - E8：dead-end 处理策略对比（删除 vs. 质量补偿 vs. 忽略）。
3. **报告章节**（2 章 Markdown 草稿，我会自己改写核心论证）：
   - 第 2 章「算法原理」：PageRank 公式推导、dead-end / spider-trap 成因、收敛性。
   - 第 3 章「算法实现细节」：ID 映射方案、CSR 存储布局、幂迭代主循环伪代码、dead-end 补偿的工程实现。
4. **不属于 A 的责任**（明确排除，别帮倒忙）：
   - Git 仓库、README、代码审查、团队协调 → 归队长 B。
   - INTERFACE.md 由 B 维护，A 按 INTERFACE.md 的规范实现；如需调整接口，提 issue 给 B，不要自行改。
   - 分块模块、PyInstaller 打包 → 归 B。
   - 数据集分析、实验扫描、报告统稿、zip 打包、邮件提交 → 归 C。

## 三、技术约束

- 只用 numpy、scipy.sparse（仅 csr_matrix），标准库（sys、os、gc、struct、time）。
- 不要 `import scipy`，只 `from scipy.sparse import csr_matrix`。
- 所有 float 数组用 np.float32；所有索引用 np.int32；别让 numpy 默认推成 int64 / float64。
- 循环里一律原地写法：`np.add(a, b, out=r_new)`、`r_new *= beta`、`r_new += (1-beta)/N`。不要写 `r_new = beta * M @ r + ...`。
- load_graph 读边后立刻 `del edges; gc.collect()`。
- 收敛判据用 L1 范数 `np.abs(r_new - r).sum() < eps`。

## 四、工作方式（重要）

队长 B 会先提供一份 INTERFACE.md，里面写清楚 load_graph / power_iteration / dump_top_k 的签名和数据类型约定。你的任务是**严格按 B 的接口规范实现**，让 B 的分块模块和 C 的扫描脚本能无缝对接。

1. **第 1 件事**：读 B 给的 INTERFACE.md。如果没拿到，停下来问我，不要自己乱定。
2. 第 2–3 天出 baseline_dense.py（E1 跑得出来即可），让 C 有真实数据能跑 E5/E6，让 B 有东西能试 PyInstaller。
3. 第 4–7 天出 CSR 版本（E2），此时 E4（稀疏 + 分块）由 B 集成。
4. 第 2 周做 E8 和报告第 2、3 章。
5. 第 3 周配合 B 做最后的 bug 修复和性能调优（这部分你是配合方，不是主导方）。

## 五、请按以下顺序交付

1. `baseline_dense.py`：稠密版本，带注释和复杂度分析，专用于 E1。
2. `main.py`（第 1 版）：load_graph + power_iteration + dump_top_k + CLI，CSR 版本。CLI 接口遵循 B 在 INTERFACE.md 中的约定。
3. `tests/test_algorithm.py`：pytest 用例，至少 4 条：
   - 小图（5 节点）对照手算 PageRank 一致；
   - 含 dead-end 的小图 Top-k 正确；
   - 含 spider-trap 的小图能在 100 轮内收敛；
   - β=0 和 β=1 的边界情况。
4. `report_ch2.md`：第 2 章「算法原理」Markdown 草稿（2000–3000 字，带公式）。
5. `report_ch3.md`：第 3 章「算法实现细节」Markdown 草稿（1500–2500 字，带伪代码）。
6. `HANDOVER.md`：见下一节要求。

## 六、HANDOVER.md 的要求

必须包含以下小节，语气是"A 写给未来要接手或复查算法部分的人"：

1. **项目当前状态**：算法部分的已完成 / 进行中 / 待开始。
2. **文件清单**：每个交付文件的路径、作用、版本。
3. **接口实现情况**：对照 B 的 INTERFACE.md 逐条勾选，标明哪些接口已按规范实现、哪些有偏差（偏差需要给 B 解释为什么）。
4. **如何本地跑通**：从 0 开始的复现步骤，每步给具体命令。
5. **算法设计决策**：列出 3–5 个关键取舍（ID 映射用哈希还是排序、dead-end 用质量补偿还是删除、收敛判据为何用 L1 等），每条给原因。
6. **已知问题与 TODO**。
7. **给 B 的反馈**：接口里哪些地方用起来不顺手、有哪些建议调整。
8. **给 C 的数据接入点**：sweep.py 里扫 β / ε 需要怎么调用 main.py、Res.txt 的确切格式。
9. **实验数据摘要**：E1、E2、E8 的关键数字表格。
10. **距离 DDL（2026-05-15 12:00 UTC+8）的剩余时间 + 算法侧 Top-3 风险**。

## 七、红线

- 不要调用任何现成 PageRank 实现。
- 不要提交稠密版本作为主方案；E1 只是基线对照。
- 不要擅自修改 INTERFACE.md 的接口定义——接口归 B 维护，你需要调整要先沟通。
- 不要跳过 tests/ 直接写最终代码。

## 八、沟通方式

- 每完成一个交付物，简要说明做了什么、遇到什么问题、下一步要做什么。
- 接口理解有歧义、dead-end 策略选择、收敛判据这类关键决策，先停下问我再推进。
- 所有代码加中文注释，便于 B 做代码审查。

现在请先确认是否已经拿到 B 的 INTERFACE.md；如果有就从 baseline_dense.py 开始，如果没有请停下等我把 INTERFACE.md 给你。

---

[可选补充：Python 版本 __、操作系统 __、数据集规模约 __ 节点 / __ 条边（如果已知）。]