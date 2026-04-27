# HANDOVER_A

这份文档是 A 写给后续接手或复查算法部分的人，重点覆盖当前实现状态、接口对齐情况、实验数字和后续风险。文中所有结论均基于 `2026-04-25 16:43:40 UTC+8` 这一时间点的仓库快照。

数据源：`experiments/report_summary.json`，最后核对 2026-04-27。

## 1. 项目当前状态

### 已完成

- `baseline_dense.py` 已完成，可作为 E1 稠密基线独立运行。
- `main.py` 的 `load_graph / power_iteration / dump_top_k / CLI` 已完成，`csr` 与 `csr_block` 路径已实测跑通真实数据 `Data.txt`。
- `tests/test_algorithm.py` 已完成，覆盖小图手算、dead-end、spider-trap、`beta=0/1` 边界和非连续 ID 回归。
- `report_ch2.md`、`report_ch3.md` 已完成初稿。
- E1、E2、E8 的关键数字已测得，可直接写入报告或继续复测。

### 进行中

- 最终提交前仍需人工完成打包版与源码版对拍，以及真实学号姓名目录命名。

### 待开始

- 若队内决定进一步压极限性能，可继续做 `np.add.at` 热点优化或 K 值敏感性测试，但这不影响当前 A 侧交付闭环。

## 2. 文件清单

| 路径 | 作用 | 当前版本 |
| --- | --- | --- |
| `baseline_dense.py` | E1 稠密基线脚本；可单独输出 Top-100 与 JSON 摘要 | v1.0 |
| `main.py` | A 侧主算法入口；支持 `dense / csr / csr_block`，并可对接 B 的分块实现 | v1.2 |
| `tests/test_algorithm.py` | A 侧 pytest 用例 | v1.0 |
| `report_ch2.md` | 报告第 2 章草稿：算法原理 | v1.0 |
| `report_ch3.md` | 报告第 3 章草稿：实现细节 | v1.0 |
| `HANDOVER_A.md` | A 侧交接文档 | v1.0 |

### 本次顺手修过、但不属于 A 主交付的辅助文件

| 路径 | 调整原因 |
| --- | --- |
| `benchmark.py` | 把临时目录切到工作区内，避免系统临时目录权限问题影响实验 |
| `tests/test_pagerank_behaviors.py` | 同样改为工作区内临时目录，并修正 benchmark 用例的 Windows 编码兼容性 |

## 3. 接口实现情况

以下对照的是 B 冻结的 `INTERFACE.md`。

### `load_graph(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]`

- [x] 返回值个数、顺序、dtype 符合约定。
- [x] `row_ptr / col_idx / out_deg` 全部为 `np.int32`。
- [x] 空文件、格式错误、负编号、文件不存在均会抛异常。
- [x] 已处理“编号可能不是紧凑连续”的情况。

实现备注：
- 严格来说，`INTERFACE.md` 没给 `id_map` 返回位，但任务说明又要求 A 处理非连续 ID。当前做法是：`load_graph` 内部维护 `_LAST_NODE_IDS` 反向映射，后续由 `dump_top_k` 读取并恢复原始 NodeID。
- 为了不丢掉“完全没有出现在边表中的孤立节点”，若检测到编号从 `0` 起步且整体密度较高，会保留 `0..max_id` 的编号空间；否则做紧凑映射。这个策略在当前数据集上是正确的，但若未来出现“从 0 开始但跨度极大且没有节点清单”的数据，需要 B 再确认语义。

### `power_iteration(...) -> tuple[np.ndarray, int, float]`

- [x] 签名符合约定。
- [x] dead-end 公式与 `INTERFACE.md` 完全一致，使用质量补偿法。
- [x] 收敛判据是 L1 范数。
- [x] 默认 `dtype=np.float32`，也支持 `np.float64` 以便测试对照。
- [x] 输出向量和在 `1e-5` 范围内接近 1。

### `dump_top_k(path: str, ranks: np.ndarray, k: int = 100) -> str`

- [x] 签名符合约定。
- [x] 输出按 `Score` 降序、`NodeID` 升序。
- [x] `Score` 固定保留 10 位小数。
- [x] 返回 `top10_signature`。

实现备注：
- [!] 因为冻结接口没有 `id_map` 参数，所以这里用模块级 `_LAST_NODE_IDS` 恢复原始 NodeID。这是唯一一个“接口外隐式状态”，需要 B 知道。

### `main.py` CLI

- [x] 已支持 `--data --beta --eps --out --mode --K --dtype --max-iter`。
- [x] `stdout` 最后一行是合法 JSON，至少包含 `peak_rss_mb / wall_sec / iters`。
- [x] `csr`、`csr_block` 已对真实数据实跑。

实现备注：
- `block` 与 `csr_block` 当前都走“先 CSR，再调 B 的 `build_blocks / iterate_by_block`”这条链路；从现有接口看是合法的，因为 `build_blocks` 本来就接受 CSR 展开的边迭代器。
- `main.py` 和 `benchmark.py` 的临时目录已经固定在工作区内的 `.tmp_runtime`，避免系统临时目录权限影响复现。

## 4. 如何本地跑通

### 从 0 开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 跑测试

```bash
python -m pytest tests/test_algorithm.py tests/test_pagerank_behaviors.py -q
```

3. 跑 E1 稠密基线

```bash
python baseline_dense.py --data Data.txt --beta 0.85 --eps 1e-8 --out Res_dense.txt --dtype float32
```

4. 跑 A 的 CSR 主版本

```bash
python main.py --data Data.txt --beta 0.85 --eps 1e-8 --out Res.txt --mode csr --K 8 --dtype float32
```

5. 跑 A+B 的 `csr_block`

```bash
python main.py --data Data.txt --beta 0.85 --eps 1e-8 --out Res_block.txt --mode csr_block --K 8 --dtype float32
```

6. 跑 benchmark 多轮对照

```bash
python benchmark.py --main main.py --data Data.txt --out bench.csv --interval 0.05 --runs 3 --modes dense,csr,csr_block --K 8 --dtype float32
```

7. 给 C 跑 sweep

```bash
python sweep.py --main main.py --data Data.txt --out experiments/sweep_results.csv --result-dir experiments/sweep_outputs --mode csr_block --K 8 --dtype float32
```

## 5. 算法设计决策

1. ID 映射不是一刀切做紧凑压缩，而是“高密度保留 `0..max_id`，明显稀疏时再重映射”。
原因：当前数据集存在孤立节点，若只按边表里出现过的 ID 建图，会把孤立点直接丢掉，`N` 会变错。

2. 主方案统一使用 dead-end 质量补偿，而不是删除法或忽略法。
原因：这是 `INTERFACE.md` 明确冻结的公式；同时它能保持节点全集不变、概率总质量不泄漏，也最方便和 B 的分块实现保持一致。

3. 收敛判据采用 L1 范数。
原因：接口已冻结为 L1；另外它直接反映两轮全图概率质量的总移动量，解释性最好。

4. CSR 用“按源点分组”的布局，而不是额外构造完整入邻接表。
原因：接口已固定为 `row_ptr / col_idx / out_deg`；B 的 `build_blocks` 也可以直接消费 CSR 展开的边流，便于联调。

5. `dump_top_k` 通过模块级反向映射恢复原始 NodeID，而没有改接口。
原因：任务说明要求 A 兜住非连续 ID，但冻结接口没给 `id_map` 参数；在不改签名的前提下，这是最小代价方案。

## 6. 已知问题与 TODO

1. `_LAST_NODE_IDS` 是进程级隐式状态。如果未来一个进程里并发加载多张图、交错调用 `dump_top_k`，会有错配风险。当前课程作业的 CLI 单次只跑一张图，因此不构成当前阻塞。
2. 若未来数据集从 `0` 开始但编号跨度巨大、同时又没有单独的节点清单，当前“保留 `0..max_id`”策略可能把一些本不存在的编号也当成孤立点。需要届时和 B 一起补充更明确的节点输入约定。
3. `baseline_dense.py` 只是实验基线，真实峰值内存约 412.64 MB，远超课程主提交通道的内存限制，不能作为最终方案。
4. 目前没有把 E8 的“删除 / 忽略”策略做成正式 CLI 模式，因为冻结接口要求主版本必须统一使用质量补偿。E8 的比较结果已经整理在本交接文档里，若后续要复测，可以单独写实验脚本，不要混入 `main.py` 主路径。

## 7. 给 B 的反馈

1. `INTERFACE.md` 让 A 处理非连续 ID，但 `dump_top_k` 又没有 `id_map` 参数，这迫使 A 侧只能用隐藏状态恢复原始 NodeID。建议下个版本把 `id_map` 明确纳入接口，或者在接口文档里正式允许模块级缓存。
2. `blocks.py` 对 `dtype` 的预期是 `np.float32` 这种类型对象，而不是 `np.dtype('float32')`。我已经在 `main.py` 适配了，但接口文档没写清楚，建议补一句。
3. `block` 和 `csr_block` 的语义现在很接近，因为 `build_blocks` 已经支持 CSR 展开的边流。若 B 后面希望这两个 mode 真正区分数据路径，建议在 `INTERFACE.md` 里把差异写死。

## 8. 给 C 的数据接入点

### C 该怎么调 `main.py`

推荐命令：

```bash
python main.py --data Data.txt --beta 0.85 --eps 1e-8 --out Res.txt --mode csr_block --K 8 --dtype float32
```

若扫参：

- 扫 `beta`：改 `--beta`
- 扫 `eps`：改 `--eps`
- 扫 `K`：改 `--K`

### `Res.txt` 的确切格式

- 每行一条记录：`NodeID Score`
- `Score` 固定保留 10 位小数
- 最多 100 行；若 `N < 100`，则输出 `N` 行
- 排序规则：先 `Score` 降序，再 `NodeID` 升序

### stdout 最后一行 JSON

C 现在可以直接解析这些字段：

- `peak_rss_mb`
- `wall_sec`
- `iters`
- `delta`
- `mode`
- `K`
- `dtype`
- `top10_signature`

## 9. 实验数据摘要

### E1 / E2（基于 `benchmark.py`，3 次平均，`beta=0.85, eps=1e-8, dtype=float32`）

| 实验 | 模式 | 平均峰值内存 / MB | 平均时间 / s | 平均迭代轮数 | Top-10 是否稳定 |
| --- | --- | ---: | ---: | ---: | --- |
| E1 | dense | 412.639 | 0.602077 | 14 | 是 |
| E2 | csr | 34.364 | 0.726331 | 15 | 是 |

结论：

- CSR 相对 dense 的峰值内存下降约 `91.67%`。
- 在当前这份 `10000` 节点、`150000` 边的合成数据上，dense 借助底层 BLAS 速度并不慢，但内存完全不满足课程主提交约束，因此只能作为对照，不可作为主方案。

### `csr_block`（补充，不计入 A 的主实验项，但已联调）

| 模式 | 平均峰值内存 / MB | 平均时间 / s | 平均迭代轮数 | Top-10 签名 |
| --- | ---: | ---: | ---: | --- |
| csr_block | 34.660 | 0.838039 | 15 | `75,8686,9678,5104,725,3257,468,7730,7175,5526` |

### E8 dead-end 策略对比（`beta=0.85, eps=1e-8`，同一份 `Data.txt`）

| 策略 | 时间 / s | 迭代轮数 | rank_sum | Top-10 与补偿法 Jaccard | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| 质量补偿 | 0.5798 | 15 | 1.000000 | 1.0000 | 主方案 |
| 忽略 dead-end | 1.5696 | 60 | 0.602009 | 1.0000 | 概率质量明显泄漏 |
| 删除 dead-end | 0.4987 | 17 | 1.000000 | 0.6667 | 先递归删掉了 1500 个节点，改变了原图语义 |

解释：

- “忽略”法在当前数据上 Top-10 恰好没变，但总质量只剩约 `0.602`，理论上已经不是合法概率分布，不能作为正式实现。
- “删除”法会直接改变节点全集，Top-10 与主方案只有 `2/3` 交集，不适合作为课程主结果。
- 因此主路径必须坚持质量补偿法，不能为了省实现量改成忽略法，也不要把删除法混到 `main.py` 正式模式里。

## 10. 距离 DDL 的剩余时间 + 算法侧 Top-3 风险

### 剩余时间

- DDL：`2026-05-15 12:00:00 UTC+8`
- 当前核对日期：`2026-04-27 UTC+8`
- 剩余：约 `18 天 12 小时`

### 算法侧 Top-3 风险

1. `dump_top_k` 的原始 ID 恢复依赖 `_LAST_NODE_IDS`，属于隐式状态。
影响：若后续有人把算法模块改成“同进程并发处理多图”，输出可能错乱。

2. B 侧分块实现对 `block/csr_block` 语义的假设仍需在最终说明中保持一致。
影响：后续合并或重构时，最容易出现“代码能跑、但接口细节不一致”的回归。

3. 未来如果数据源更换，且节点编号规则与当前假设不一致，`load_graph` 的编号策略需要重新确认。
影响：最坏情况下会把不存在的编号当孤立点，或者反过来丢失孤立点，直接改写 `N` 和排名分布。
