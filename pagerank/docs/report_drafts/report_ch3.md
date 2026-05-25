# 第 3 章 算法实现细节

## 3.1 总体实现思路

本项目 A 侧的核心实现围绕三个接口展开：

1. `load_graph(path)`：读取 `Data.txt`，构造按源点分组的 CSR；
2. `power_iteration(...)`：在 CSR 上执行幂迭代；
3. `dump_top_k(path, ranks, k)`：按约定格式输出结果。

实现目标不是“先写出一个能跑的 PageRank”，而是在 Python 环境下同时满足接口统一、内存受控、实验可复现三项要求。因此整条链路都遵循两个原则：

- 所有索引数组统一为 `np.int32`；
- 所有排名相关浮点数组统一为 `np.float32`。

这样做的直接收益是内存可控，且与 B 的分块模块、C 的扫参脚本保持同一套数据约定。

## 3.2 节点编号与 ID 映射方案

课程给定的边文件只有两列：`FromNodeID ToNodeID`。理论上存在两种可能：

1. 节点编号本身已经是从 `0` 开始的连续编号；
2. 节点编号是稀疏的，需要在程序内部做压缩映射。

但是这里有一个实际工程细节：如果文件里某些编号从未出现在任何边中，它们可能代表“完全孤立节点”。若简单按照“见到的节点集合”做紧凑映射，就会把这些节点直接丢掉，导致节点总数 `N` 变小，进而改变 PageRank 分布。

因此当前实现采用折中策略：

1. 先流式读入原始边表，只保存原始 `src/dst`；
2. 若观察到编号从 `0` 起步且整体编号密度较高，则保留 `0..max_id` 的编号空间，把缺席编号视为孤立点；
3. 若编号并非从 `0` 起步，或跨度远大于已观测节点数，则退化为紧凑重映射。

这样既能兼顾课程数据集中的孤立点，又能在明显非连续编号场景下完成压缩。由于冻结接口没有把 `id_map` 暴露给 `dump_top_k`，实现中额外维护了一个模块级反向映射 `_LAST_NODE_IDS`，供输出阶段恢复原始 NodeID。

## 3.3 CSR 存储布局

本实现使用经典的 Compressed Sparse Row（CSR）布局，但含义按“源节点的出边列表”组织：

- `row_ptr[u] : row_ptr[u+1]` 表示源点 `u` 的出边区间；
- `col_idx[row_ptr[u] : row_ptr[u+1]]` 存放这些出边的目标点；
- `out_deg[u] = row_ptr[u+1] - row_ptr[u]`。

它的优点是结构紧凑、易于序列化、适合与 B 的分块模块拼接。相较于稠密矩阵 `N x N` 的存储方式，CSR 只为真实存在的边分配空间，其空间复杂度为

\[
O(N + M),
\]

其中 `N` 是节点数，`M` 是边数。对于当前 `10000` 节点、`150000` 边的数据集，CSR 版本内存占用显著低于稠密方案。

建图过程分两步：

1. 统计每个源点的出度，得到 `out_deg`；
2. 对 `out_deg` 做前缀和，得到 `row_ptr`，再借助写指针把目标点写入 `col_idx`。

这样不需要对全部边做额外排序，只需一次线性遍历即可完成压缩。

## 3.4 `load_graph` 的工程实现

`load_graph` 不直接把整个文本读成一个大字符串，而是逐行扫描。这样做有三个好处：

1. 格式校验可以精确到行号；
2. 不依赖 `numpy` 的隐式 dtype 推断；
3. 便于后续在读入后立刻释放临时边缓冲。

核心流程如下：

```text
load_graph(path):
    逐行读取 (raw_src, raw_dst)
    校验：两列、整数、非负、可落入 int32
    缓存原始边表
    根据编号分布决定：
        保留 0..max_id
        或构造紧凑映射
    统计 out_deg
    前缀和生成 row_ptr
    按写指针填充 col_idx
    删除临时边缓冲并 gc.collect()
    返回 row_ptr, col_idx, out_deg, N
```

这里特别注意两点：

1. 所有返回数组必须已经是 `np.int32`，不能依赖调用方再做一次转换；
2. 临时边缓冲在建好 CSR 后立即释放，避免高峰内存叠加。

## 3.5 幂迭代主循环

`power_iteration` 的输入是 CSR 三元组与参数 `(beta, eps)`，输出为 `(ranks, iters, delta)`。实现没有使用任何现成 PageRank API，而是直接按照冻结公式做原地更新。

伪代码如下：

```text
power_iteration(row_ptr, col_idx, out_deg, N, beta, eps):
    r      <- [1/N] * N
    r_new  <- empty(N)
    delta_buf <- empty(N)
    live_mask <- out_deg > 0
    dead_mask <- out_deg == 0
    inv_out_deg[live_mask] <- 1 / out_deg[live_mask]

    repeat t = 1 .. max_iter:
        dangling_mass <- sum(r[dead_mask])
        base <- (1 - beta) / N + beta * dangling_mass / N
        r_new.fill(base)

        for each live source node u:
            contrib <- beta * r[u] / out_deg[u]
            for each edge u -> v:
                r_new[v] += contrib

        delta_buf <- abs(r_new - r)
        delta <- sum(delta_buf)
        if delta < eps:
            return r_new, t, delta

        swap(r, r_new)
```

实际代码中，为了减少临时对象分配，做了几处工程化处理：

1. `r_new`、`delta_buffer` 只申请一次，循环内复用；
2. `inv_out_deg` 预先计算，避免每轮重复除法；
3. `np.add.at` 用于把单个源点的贡献散射到多个目标点；
4. 差值范数通过缓冲区原地计算，避免每轮新建一份差值数组。

## 3.6 dead-end 补偿的工程实现

dead-end 的统一实现点只有一个：每轮开始先求

\[
dangling\_mass=\sum_{i:out\_deg[i]=0} r[i].
\]

然后把

\[
\beta \cdot dangling\_mass / N
\]

直接并入所有节点共享的 `base` 项。这样每一轮都只需一次 dead-end 质量求和，不必对每个 dead-end 节点单独发边，既符合冻结公式，又能减少实现复杂度。

与两种对照策略相比：

- “忽略 dead-end”会导致总概率质量逐渐下降；
- “删除 dead-end”需要先做递归剪枝，再回填结果，流程更复杂，也会改变原图语义；
- “质量补偿”既保持原图节点全集，又能维持概率质量守恒，是课程要求下最稳妥的工程方案。

## 3.7 Top-k 输出与接口对接

`dump_top_k` 的职责有两层：

1. 按 `Score` 降序、`NodeID` 升序写出 Top-100；
2. 返回 `top10_signature`，供 `benchmark.py` 与 `sweep.py` 做回归比较。

输出文件每行格式固定为：

```text
NodeID Score
```

其中 `Score` 固定保留 10 位小数。由于前面提到的内部重映射，写文件时不能直接把内部数组下标当成 NodeID，而必须先通过 `_LAST_NODE_IDS` 还原。也正因为如此，当前实现默认假定同一进程中“最近一次 `load_graph` 的图”和“当前要输出的 `ranks`”是同一份图，这是后续接口优化时最值得 B 关注的一点。

## 3.8 本章小结

本章的核心结论是：PageRank 的公式并不复杂，真正困难的是在 Python 的内存约束下把它落成稳定、可复现、可对接的实现。A 侧方案通过“流式读图 + CSR 压缩 + 原地幂迭代 + dead-end 统一补偿”完成了主算法闭环，并给 B 的分块版本提供了稳定的输入数据结构，也给 C 的实验脚本提供了统一的 CLI 与结果格式。
