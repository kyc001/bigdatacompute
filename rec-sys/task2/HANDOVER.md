# Task2 Track1 交接说明

## HANDOVER 维护规范

这份文档是“实验台账 + 队友交接”，不是最终报告。后续同学继续调参或改代码时，请按下面规则维护：

1. **只记录有信息量的内容。** 写清楚做了什么、为什么做、结果怎样、得到什么 insight。不要堆页数、不要复制大段代码、不要放无用图片、不要写明显的 LLM 套话。
2. **保留失败实验。** 失败实验如果说明了“为什么这个方向不值得继续”，就很有价值。不要因为结果差就删掉，除非确认数据或命令本身错了。
3. **每次实验尽量补齐字段。** 推荐格式：日期/负责人、目的或假设、代码/参数改动、运行命令、RMSE、耗时、结论、下一步。
4. **区分本地结果和 OJ 结果。** 本地 CPU、线程调度和 OJ 不同；本地时间只用于比较优化方向。写时间时要标明 `benchmark-runs`。
5. **核心代码只贴短片段。** 报告里引用 5-15 行关键代码即可，优先贴残差计算、shrinkage、并行归并、预测函数，不要整段复制 `solution.cpp`。
6. **改 `solution.cpp` 后必须验证。** 至少运行：
   ```powershell
   pixi run task2-track1-cpp-scan
   pixi run task2-track1-cpp-smoke
   pixi run task2-track1-cpp-benchmark-10
   ```
7. **如果改 benchmark/runner，要单独说明。** 本任务曾经踩过 `judge_data.bin` 读取格式错误，评测工具错了会让所有实验结论失真。
8. **不要提交大数据。** `.npy`、`.bin`、`.npz` 等本地数据/缓存保持忽略。

## 当前选择

Track1 是增量 SVD 性能优化题，正式实现改为 C++，主文件是：

```text
rec-sys/task2/track1/solution.cpp
```

压缩包里的 `solution.py` 只是官方模板/参考，不作为当前主提交；当前工作目录已经移除它，避免 benchmark 自动模式误跑 Python。Python 相关入口统一用 pixi 管理。

常用命令：

```powershell
pixi run task2-track1-extract
pixi run task2-track1-cpp-scan
pixi run task2-track1-cpp-smoke
pixi run task2-track1-cpp-benchmark
```

如果当前 PowerShell 还残留带引号的 `PIXI_HOME`，先执行：

```powershell
$env:PIXI_HOME = 'C:\Users\kyc\.pixi'
```

## 已完成内容

1. 已阅读 Track1 题面并整理中文要求文档：`rec-sys/task2/要求文档.md`。
2. 已抽出官方模板到 `rec-sys/task2/track1/`。
3. 已把本地大数据文件加入 `.gitignore`，避免误提交 `.npy` 和 `.bin`。
4. 已实现 C++ 版 `IncrementalSVD`：
   - `load_base_model()` 保存官方预训练 `P/Q` 指针；
   - `update()` 用增量评分统计 user/item 残差偏置；
   - `predict()` 使用 `global_mean + user_bias + item_bias` 的 ultra-fast 预测；
   - 预测结果会截断到 `[0.5, 5.0]`。
5. 已补齐本地 C++ benchmark 配置：
   - `rec-sys/task2/runner/cpp/main.cpp`
   - `rec-sys/task2/scripts/scan_cpp.py`
   - `rec-sys/task2/scripts/cpp_smoke.py`
   - `rec-sys/task2/scripts/extract_track1_data.py`
   - `pixi.toml` 中的 `task2-track1-*` 任务

## 当前算法思路

完整 1024 维 SGD 很贵，尤其是要在 200 万增量评分和 200 万测试评分上跑。进一步实验发现，在本地 full test 上，使用基础 SVD dot 只能把 RMSE 从 `0.93161` 轻微压到 `0.93131`，但会把 10 轮耗时从约 `0.48s` 拉高到约 `2.75s`。由于 Track1 有效后主要按运行时间竞争，当前 C++ 版本选择 ultra-fast bias-only 路线，不直接改 `P/Q`，也不在 `update()` 或 `predict()` 里做 1024 维 dot。

当前残差定义为：

```text
residual = rating - global_mean
```

然后分别累计到用户和物品上：

```text
user_bias[u] = 0.75 * user_residual_sum[u] / (user_count[u] + 20)
item_bias[i] = 1.00 * item_residual_sum[i] / (item_count[i] + 5)
```

这些参数来自本地采样实验。采样 30 万条测试数据时，基础 RMSE 约 `1.02118`，残差偏置后约 `0.93011`，采样下降约 `0.091`。这个数只代表本地采样，不等于最终 OJ 结果。

当前版本还做了一个重要的工程优化：每个 OpenMP 线程保留本地 user/item 累加数组避免锁竞争，但用 touched index 列表只归并本批实际出现过的用户和物品，避免每批全量扫描 `num_users * threads` 和 `num_items * threads`。

注意：曾经的本地 C++ runner 把 `judge_data.bin` 里的评分行按三个 `float` 读入，导致 `int32 user_id/item_id` 被误读成极小浮点数，RMSE 测量失真。现在 `rec-sys/task2/runner/cpp/main.cpp` 已修正为按 `int32, int32, float` 读取，benchmark 结果才和 `.npy` 数据一致。

## 实验过程与 insights

这一节给队友写报告用，建议只摘取有信息量的部分，不要整段照抄。

### 实验结果汇总

| 编号 | 日期 | 方法 | 验证命令 | RMSE | 10 轮耗时 | 结论 |
|---|---|---|---|---:|---:|---|
| T2-E0 | 2026-05-28 | 旧本地 runner + dot residual + dot predict，但二进制读取错误 | `pixi run task2-track1-cpp-benchmark` | 1.019420 | - | 无效结果。reader 把 `int32 user/item` 当成 float，导致测量失真。 |
| T2-E1 | 2026-05-28 | 修正 runner，`rating - base_svd_prediction` 残差，预测保留 `dot(P[u], Q[i])` | `pixi run task2-track1-cpp-benchmark-10` | 0.931279 | 约 7.39s | 精度最好一点，但 update 和 predict 都做 1024 维 dot，时间不适合排名。 |
| T2-E2 | 2026-05-28 | 去掉 update dot，`rating - global_mean` 残差，预测仍保留 `dot(P[u], Q[i])` | 离线 sweep + C++ benchmark | 0.931307 | 约 2.75s | RMSE 只差 `0.000028`，说明增量数据的 user/item 偏置已经解释了主要收益。 |
| T2-E3 | 2026-05-28 | touched-list 稀疏归并，只归并本批出现过的 user/item | `pixi run task2-track1-cpp-benchmark-10` | 0.931307 | 约 2.75s 以下 | 减少全量扫描开销，对不同机器有波动，但方向正确。 |
| T2-E4 | 2026-05-28 | 当前版本，`rating - global_mean` 残差，预测只用 `global_mean + user_bias + item_bias` | `pixi run task2-track1-cpp-benchmark-10` | 0.931613 | 约 0.48-0.64s | RMSE 轻微变差，但速度提升最大；Track1 有效后按时间竞争，当前推荐提交。 |

说明：当前版本 benchmark 中“更新前 RMSE”是当前 `predict()` 在 update 前的 RMSE，即 global-mean baseline；它不是官方 SVD dot baseline。但当前更新后 RMSE `0.931613` 仍明显低于官方基础 SVD 的约 `1.021712`，所以不是靠抬高基线取巧。

### 详细实验记录

#### T2-E0：旧本地 runner 读错二进制，导致误判

- 目的：先跑通 C++ 版本，确认 `load_base_model`、`update`、`predict` 接口能被本地 benchmark 调用。
- 当时方法：`update()` 计算 `global_mean + dot(P[u], Q[i])`，用真实评分减去 base SVD 预测作为残差，再累计 user/item bias；`predict()` 继续使用 `global_mean + dot(P[u], Q[i]) + user_bias + item_bias`。
- 当时结果：本地 1 轮显示 `RMSE 1.023281 -> 1.019420`，只下降 `0.003861`。
- 问题定位：离线 `.npy` 全量分析显示同类 residual-bias 方法应该到约 `0.931`，差距过大。继续检查 `judge_data.bin` 后发现 C++ runner 按 3 个 `float` 读取评分行，但真实布局是 `int32 user_id, int32 item_id, float32 rating`。
- insight：评测工具也是实验链路的一部分。reader 错误会让大部分用户/物品 id cast 成 0，导致 RMSE 看起来“有效但很弱”，实际结论完全不可信。
- 处理：修正 `rec-sys/task2/runner/cpp/main.cpp`，按 `int32, int32, float` 读取，并把这个数据契约记录到 `.trellis/spec/backend/quality-guidelines.md`。

#### T2-E1：base-SVD residual + base-SVD predict

- 目的：建立“精度优先”的强基线，判断只做 residual bias 能否明显改善官方基础 SVD。
- 方法：
  - `update()` 对每条增量评分计算 `pred = clamp(global_mean + dot(P[u], Q[i]))`；
  - `residual = rating - pred`；
  - 分别累计到用户和物品；
  - `predict()` 使用 `global_mean + dot(P[u], Q[i]) + user_bias + item_bias`。
- 结果：修正 runner 后，full test 约 `RMSE 1.021712 -> 0.931279`，10 轮约 `7.39s`。
- 结论：残差偏置非常有效，说明增量数据确实提供了强 user/item 修正信号。但该版本在 update 和 predict 阶段都要做大量 1024 维 dot，时间不适合竞争性排名。
- 报告可写 insight：从“完整 SGD”退一步，只学习残差偏置，已经能取得大部分精度收益；优化目标应从“更复杂的模型”转向“同等有效下更少计算”。

#### T2-E2：去掉 update 阶段的 base dot

- 目的：验证 update 中的 1024 维 dot 是否真的必要。
- 方法：
  - 把 `residual = rating - clamp(global_mean + dot(P[u], Q[i]))` 改成 `residual = rating - global_mean`；
  - `predict()` 暂时仍保留 `global_mean + dot(P[u], Q[i]) + user_bias + item_bias`。
- 结果：full test 约 `0.931307`，相比 T2-E1 只差约 `0.000028`；10 轮时间降到约 `2.75s`。
- 结论：update 阶段的 base SVD dot 成本很高，但对最终 RMSE 几乎没有贡献。增量评分相对于全局均值的偏移，已经能估计出足够好的 user/item bias。
- 报告可写 insight：用近似残差换掉精确残差，是一次“几乎不损失 RMSE、显著减少计算”的有效近似。

#### T2-E3：OpenMP 本地累加 + touched-list 稀疏归并

- 目的：减少并行统计中的锁竞争和无效扫描。
- 方法：
  - 每个线程维护自己的 `local_user_sum/local_item_sum/local_user_count/local_item_count`；
  - 线程内第一次遇到某个 user/item 时，把 id 放进 `local_touched_users/local_touched_items`；
  - 并行区结束后，只遍历 touched list，把本批实际出现过的 id 归并到全局 bias。
- 为什么这样做：
  - 如果直接在全局数组上 `atomic`，每条评分都要同步；
  - 如果每批结束后全量扫描所有 user/item，再乘以线程数，会浪费大量时间；
  - touched list 利用了评分矩阵稀疏性。
- 结论：这是工程优化，不改变算法输出。不同机器上收益有波动，但逻辑上比全量归并更适合稀疏数据。
- 报告可写 insight：并行化不只是加 OpenMP。局部缓冲避免写冲突，稀疏归并减少无用循环，两者配合才有意义。

#### T2-E4：当前 ultra-fast bias-only predict

- 目的：压缩最大热点 `predict()`。benchmark 对约 200 万测试评分逐个调用 `predict()`，如果每次都做 1024 维 dot，会成为主要耗时。
- 方法：
  - 保留 `P/Q` 指针以符合接口；
  - 不再在 `predict()` 里计算 `dot(P[u], Q[i])`；
  - 直接返回 `clamp(global_mean + user_bias[user] + item_bias[item])`。
- 结果：
  - RMSE 约 `0.931613`，比 T2-E2/T2-E3 稍差；
  - 10 轮总耗时约 `0.48-0.64s`，平均单轮约 `0.05-0.06s`；
  - 仍明显优于官方基础 SVD 约 `1.021712`，满足有效性要求。
- 结论：如果最终规则是“有效提交按时间排名”，当前版本是最合适的提交主线。如果隐藏榜更强调 RMSE，可以回退到 T2-E3。
- 风险：如果 OJ 的隐藏 test 分布和本地 test 差异很大，bias-only predict 可能比保留 base dot 的版本更不稳。保守备选是 T2-E3。

### 成功的 insight

1. **不要盲目做完整 SGD。** 1024 维向量、200 万增量评分、200 万测试评分意味着完整 SGD 和 full dot 都非常贵。题目虽然叫增量 SVD，但竞赛指标更偏向“有效改进后的时间”，所以先找低成本误差修正比重训隐向量更划算。
2. **增量数据主要提供 user/item 偏置信号。** 离线 full test 扫描发现，用 `rating - global_mean` 做残差几乎达到用 `rating - base_svd_prediction` 的 RMSE，说明本地数据里主要可利用信号是用户和物品的系统性偏高/偏低。
3. **预测阶段的 dot 是最大热点。** 保留 `dot(P[u], Q[i])` 会让 200 万次预测每次做 1024 维乘加。去掉 dot 后 RMSE 只从约 `0.931307` 到 `0.931613`，但 10 轮时间从约 `2.75s` 到约 `0.5s`。
4. **并行统计要避免锁，也要避免全量归并。** 每线程本地数组可以避免 atomic/critical，但如果每批都全量扫描所有用户和物品，也会浪费时间。touched-list 只处理本批出现过的 index，更符合稀疏评分数据。
5. **本地评测工具本身也要验证。** 修正 `judge_data.bin` 读取格式前，RMSE 下降只有 `0.003861`；修正后同类方法下降约 `0.09`。这个发现可以作为报告里“调试与验证”的有效内容。

### 失败或放弃的方向

1. **完整 1024 维 SGD 更新 P/Q：** 没有作为主线实现。理论上更贴近矩阵分解，但每条样本要读写两条 1024 维向量，内存带宽和时间都不划算；并且本地 residual-bias 已经能大幅降低 RMSE。
2. **保留基础 SVD dot 做最终预测：** 精度略好，但时间差距太大。若隐藏榜更看重 RMSE，可以作为备选；若最终按有效提交时间排名，当前 bias-only 更合理。
3. **复杂 SVD/Golub-Kahan 分解：** 你给的 `矩阵SVD分解介绍.pdf` 对并行思想有帮助，但本题已经给出预训练 `P/Q`，不需要重新对原始大矩阵做 SVD 分解。真正有用的是其中关于热点循环、矩阵/向量计算、OpenMP 并行的思路。

### 报告可引用的核心代码片段

残差统计主线：

```cpp
const float residual = r.rating - global_mean;
us[r.user] += residual;
is[r.item] += residual;
++uc[r.user];
++ic[r.item];
```

shrinkage 抑制低频用户/物品过拟合：

```cpp
user_bias[u] = user_weight * user_sum[u] /
               (static_cast<float>(user_count[u]) + user_shrink);
item_bias[i] = item_weight * item_sum[i] /
               (static_cast<float>(item_count[i]) + item_shrink);
```

ultra-fast 预测：

```cpp
float score = global_mean + user_bias[user_id] + item_bias[item_id];
return std::min(5.0f, std::max(0.5f, score));
```

本地 runner 的二进制读取修正：

```cpp
ratings[row] = Rating{
    read_one<std::int32_t>(in),
    read_one<std::int32_t>(in),
    read_one<float>(in),
};
```

## 后续调参方向

优先调这四个常量：

```cpp
user_shrink
item_shrink
user_weight
item_weight
```

建议搜索范围：

```text
user_shrink: 10, 20, 40, 80
item_shrink: 2, 5, 10, 20
user_weight: 0.5, 0.75, 1.0
item_weight: 0.75, 1.0, 1.25
```

如果隐藏集 RMSE 不够，再考虑加入“少量维度因子更新”，例如只更新前 32/64/128 维，观察 RMSE 和时间的折中。

## 已验证

本地已经完成：

```powershell
g++ -std=c++17 -O3 -fopenmp -c rec-sys/task2/track1/solution.cpp
```

并用小型 C++ 测试验证了：

1. `load_base_model()` 可以正常载入矩阵指针。
2. `update()` 后，高分样本预测会上升，低分样本预测会下降。
3. 越界用户/物品会返回 `global_mean`。

完整 benchmark 已通过 pixi 跑通：

```powershell
pixi run task2-track1-cpp-benchmark
pixi run task2-track1-cpp-benchmark-10
```

本机 1 轮结果：

```text
更新前 RMSE: 1.023239
更新后 RMSE: 0.931613
RMSE 下降:   0.091626
结果有效性:  有效 PASS
update+predict 耗时: 0.136 秒
```

本机 10 轮结果：

```text
更新前 RMSE: 1.023239
更新后 RMSE: 0.931613
RMSE 下降:   0.091626
结果有效性:  有效 PASS
10 轮总耗时: 约 0.48-0.64 秒
平均单轮:   约 0.05-0.06 秒
```

说明：本地 runner 的时间只用于比较优化方向，最终仍以 OJ 环境为准。

## 提交提醒

正式提交时重点检查：

1. 上传/提交的是 `solution.cpp`。
2. 代码里没有文件 I/O、系统调用、硬编码数据路径。
3. 本地完整 benchmark 用 `pixi run task2-track1-cpp-benchmark` 跑，不要误跑 Python 默认模板。
4. 报告由队友写时，可以把“bias-only 残差偏置 + shrinkage + OpenMP 本地统计 + touched-list 稀疏归并”作为算法说明主线。
5. 如果隐藏榜更看重 RMSE 而不是有效后时间，可以回退到 `global_mean + dot(P[u], Q[i]) + user_bias + item_bias` 版本；本地 full test 约 `0.931307`，但 10 轮约 `2.75s`。
