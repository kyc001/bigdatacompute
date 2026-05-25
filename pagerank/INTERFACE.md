# INTERFACE.md

## Freeze 信息

- 负责人：B
- 状态：`FROZEN`
- 版本：`v1.0`
- `v1.0.1 (2026-04-27)`: 增补可选参数 `chunk_size`，默认 `1 << 15`，向后兼容；不影响默认行为
- 生效范围：A / B / C 全队
- 变更规则：本文件冻结后，不允许直接改签名、改输出格式、改 dead-end 公式。若必须调整，先提 issue，再在下一个版本中变更。

## 1. 统一数据约定

- 输入文件：`Data.txt`
  - 每行两个整数，格式为 `FromNodeID ToNodeID`
  - 表示一条从源节点指向目标节点的有向边
- 节点编号：
  - 默认假设数据集中节点编号已经是从 `0` 开始的连续整数
  - 若未来发现节点编号不连续，由 A 在 `load_graph` 内完成重映射
- 索引 dtype：统一使用 `np.int32`
- 浮点 dtype：统一使用 `np.float32`
- 阻尼系数：`beta = 0.85`
- 收敛阈值：`eps = 1e-8`
- 收敛准则：统一采用 `L1` 范数
  - `delta = ||r_new - r||_1`
  - 当 `delta < eps` 时停止

## 2. dead-end 与 spider-trap 统一公式

### 2.1 dead-end 统一策略

全队统一使用“质量补偿法”，实现位置归 A 的 `power_iteration`，B 的分块版复用同一份公式。

设：

- `D = { i | out_deg[i] == 0 }`
- `dangling_mass = sum(r[i] for i in D)`

则每轮迭代中，对所有节点 `v` 都补偿：

```text
beta * dangling_mass / N
```

完整更新公式写死如下：

```text
r_new[v] = (1 - beta) / N
         + beta * dangling_mass / N
         + beta * sum(r[u] / out_deg[u] for every edge u -> v, out_deg[u] > 0)
```

说明：

- 不允许改成删点法或后处理归一化法
- spider-trap 不单独特判，由 teleport + dead-end 补偿共同保证收敛

## 3. A 角色必须实现的接口

### 3.1 `load_graph`

```python
def load_graph(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    返回:
        row_ptr: np.ndarray[np.int32], shape = (N + 1,)
        col_idx: np.ndarray[np.int32], shape = (M,)
        out_deg: np.ndarray[np.int32], shape = (N,)
        N: int
    """
```

语义：

- 使用 CSR，按“源节点”建行
- `row_ptr[u] : row_ptr[u + 1]` 是源节点 `u` 的所有出边
- `col_idx[row_ptr[u] : row_ptr[u + 1]]` 中保存目标节点编号
- `out_deg[u] == row_ptr[u + 1] - row_ptr[u]`

异常约定：

- 文件不存在：抛 `FileNotFoundError`
- 文件为空、格式错误、节点编号非法：抛 `ValueError`
- 输出数组 dtype 不符合约定：视为实现 bug，不允许 silently 改签名

### 3.2 `power_iteration`

```python
def power_iteration(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    out_deg: np.ndarray,
    N: int,
    beta: float,
    eps: float,
    dtype: np.dtype = np.float32,
    max_iter: int = 200,
) -> tuple[np.ndarray, int, float]:
    """
    返回:
        ranks: np.ndarray[np.float32]
        iters: int
        delta: float
    """
```

语义：

- 必须使用第 2 节规定的 dead-end 补偿公式
- 必须处理 spider-trap
- 必须在 `dtype=np.float32` 下正确工作
- 返回向量满足 `abs(ranks.sum() - 1.0) <= 1e-5`

### 3.3 `dump_top_k`

```python
def dump_top_k(path: str, ranks: np.ndarray, k: int = 100) -> str:
    """
    把 Top-k 写入 path，并返回 top10_signature 字符串。
    """
```

语义：

- 输出按 Score 降序
- 若 Score 相同，按 NodeID 升序
- `Score` 保留 10 位小数
- 返回值 `top10_signature` 供 benchmark 与回归测试使用
- 建议格式：`node0,node1,node2,...,node9`

## 4. B 角色暴露的接口

### 4.1 `build_blocks`

```python
def build_blocks(edges, K: int, tmp_dir: str) -> list[dict]:
    """
    按目标节点把边分桶到 K 个 .bin 文件。
    """
```

参数约定：

- `edges` 支持两种形式：
  - `np.ndarray`，形状为 `(M, 2)`，列含义为 `(src, dst)`
  - `(edge_iterable, N)`，用于从 CSR 惰性展开边流
- `K >= 1`
- `tmp_dir` 必须可写

返回值：

- 返回长度为 `K` 的列表
- 每个元素至少包含：
  - `block_id`
  - `path`
  - `node_start`
  - `node_stop`
  - `node_count`
  - `edge_count`

块文件格式：

- 每条记录两个 `int32`
  - `src`
  - `local_dst = dst - node_start`

### 4.2 `iterate_by_block`

```python
def iterate_by_block(
    blocks: list[dict],
    out_deg: np.ndarray,
    N: int,
    beta: float = 0.85,
    eps: float = 1e-8,
    dtype: np.dtype = np.float32,
    max_iter: int = 200,
    tmp_dir: str | None = None,
    chunk_size: int = 1 << 15,
) -> tuple[np.ndarray, int, float]:
    """
    返回:
        ranks: np.ndarray[np.float32]
        iters: int
        delta: float
    """
```

语义：

- 必须是真分块：边先落盘，再按块 `memmap` 读取
- 每轮只能让当前块的 `r_new[node_start:node_stop]` 片段作为活动更新区
- 不允许把整个 `r_new` 作为普通内存数组常驻
- dead-end 补偿公式必须与 A 的 `power_iteration` 完全一致

## 5. `main.py` 命令行接口

统一命令行约定如下：

```bash
python main.py \
  --data Data.txt \
  --beta 0.85 \
  --eps 1e-8 \
  --out Res.txt \
  --mode csr_block \
  --K 8 \
  --dtype float32
```

参数说明：

- `--data`：输入数据文件路径
- `--beta`：阻尼系数
- `--eps`：收敛阈值
- `--out`：输出结果文件路径
- `--mode`：`dense | csr | block | csr_block`
- `--K`：块数，供 `block / csr_block` 使用
- `--dtype`：`float32 | float64`
- `--max-iter`：可选，默认 `200`

模式约定：

- `dense`：只用于 mock / 小图 / 对照实验，不用于最终主提交版本
- `csr`：A 的纯 CSR 版本
- `block`：B 的分块版本
- `csr_block`：A 的 CSR 输入 + B 的分块迭代，作为主提交版本

## 6. `Res.txt` 输出格式

- 文件路径由 `--out` 指定
- 每行格式：

```text
NodeID Score
```

- `Score` 固定保留 `10` 位小数
- 共 `100` 行；若节点数不足 `100`，则输出 `min(100, N)` 行
- 排序规则：
  - 第一关键字：`Score` 降序
  - 第二关键字：`NodeID` 升序

## 7. stdout 最后一行 JSON 规范

`main.py` 标准输出的最后一行必须是合法 JSON，供 `benchmark.py` 与 C 的 `sweep.py` 解析。

最少必须包含：

```json
{
  "peak_rss_mb": 0.0,
  "wall_sec": 0.0,
  "iters": 0
}
```

允许增加但不得删除的推荐字段：

```json
{
  "peak_rss_mb": 32.4,
  "wall_sec": 0.28,
  "iters": 15,
  "mode": "csr_block",
  "K": 8,
  "dtype": "float32",
  "delta": 7.1e-09,
  "top10_signature": "7903,2660,2296,5420,8417,2179,2339,1293,5370,1591"
}
```

约束：

- 最后一行必须是纯 JSON，不允许再混入其他日志
- benchmark 只解析最后一行

## 8. benchmark.py 约定

命令行：

```bash
python benchmark.py \
  --main main.py \
  --data Data.txt \
  --out bench.csv \
  --interval 0.05 \
  --runs 3 \
  --modes dense,csr,block,csr_block
```

CSV schema 固定为：

```text
run_id,mode,K,dtype,peak_rss_mb,wall_sec,iters,top10_signature
```

说明：

- `peak_rss_mb` 由 `psutil` 采样 RSS 峰值
- `wall_sec`、`iters`、`top10_signature` 从 `main.py` 最后一行 JSON 解析
- 若某个 mode 运行失败：
  - 该行仍写入 CSV
  - `top10_signature` 填 `ERROR`

## 9. C 角色需要遵守的接口点

- `sweep.py` 只通过 CLI 调 `main.py`
- `plot.py` 只读 benchmark 输出的 CSV，不直接 import `main.py`
- 报告中的实验图表统一以 CSV 为数据源，不手填数字

## 10. 错误处理约定

- 参数错误：抛 `ValueError`
- 文件不存在：抛 `FileNotFoundError`
- 文件系统写入失败：保持原始 `OSError`
- 算法模块不吞异常；benchmark 负责把失败记录到 CSV

## 11. 不允许做的事

- 不允许调用任何现成 PageRank API
- 不允许在循环里写 `r_new = beta * M @ r + ...`
- 不允许把 dead-end 处理写成另一套公式
- 不允许修改 `Res.txt` 格式与 stdout JSON 格式
