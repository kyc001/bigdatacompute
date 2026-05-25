# code_review_checklist.md

## 对 A 的算法代码审查

- `load_graph` 是否严格返回 `(row_ptr, col_idx, out_deg, N)`，且 dtype 为 `int32`
- `power_iteration` 是否使用了 `INTERFACE.md` 里冻结的 dead-end 质量补偿公式
- 收敛判据是否为 `L1` 范数，并严格使用 `eps = 1e-8`
- 是否存在现成 PageRank API、特征向量求解器或隐藏的矩阵库捷径
- 是否出现了循环内大临时数组表达式，如 `r_new = beta * M @ r + ...`
- `dump_top_k` 是否输出 10 位小数，且同分时按 NodeID 升序
- 对空文件、坏格式、非法节点编号是否抛了约定异常
- 输出向量是否满足 `abs(sum(r) - 1.0) <= 1e-5`

## 对 C 的实验与脚本代码审查

- `sweep.py` 是否只通过 CLI 调 `main.py`，没有绕过接口直接 import 内部函数
- `plot.py` 是否只依赖 CSV，不手工硬编码实验结果
- benchmark 聚合时是否保留 `mode / K / dtype / peak_rss_mb / wall_sec / iters / top10_signature`
- 图表标题、坐标轴、单位是否完整，是否区分时间和内存
- 是否区分开发环境结果与最终报告结果，避免混入旧数据
- 画图脚本是否能在没有 GUI 的环境下运行
- 输出文件命名是否稳定，便于报告和 zip 打包引用
- 报告引用的数字是否能回溯到原始 CSV
