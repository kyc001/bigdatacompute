# Task2 Track1 交接说明

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
   - `predict()` 使用 `global_mean + dot(P[u], Q[i]) + user_bias + item_bias`；
   - 预测结果会截断到 `[0.5, 5.0]`。
5. 已补齐本地 C++ benchmark 配置：
   - `rec-sys/task2/runner/cpp/main.cpp`
   - `rec-sys/task2/scripts/scan_cpp.py`
   - `rec-sys/task2/scripts/cpp_smoke.py`
   - `rec-sys/task2/scripts/extract_track1_data.py`
   - `pixi.toml` 中的 `task2-track1-*` 任务

## 当前算法思路

完整 1024 维 SGD 很贵，尤其是要在 200 万增量评分和 200 万测试评分上跑。当前 C++ 版本不直接改 `P/Q`，而是统计增量数据中“基础 SVD 预测值”和真实评分之间的残差：

```text
residual = rating - clamp(global_mean + dot(P[u], Q[i]))
```

然后分别累计到用户和物品上：

```text
user_bias[u] = 0.75 * user_residual_sum[u] / (user_count[u] + 20)
item_bias[i] = 1.00 * item_residual_sum[i] / (item_count[i] + 5)
```

这些参数来自本地采样实验。采样 30 万条测试数据时，基础 RMSE 约 `1.02118`，残差偏置后约 `0.93011`，采样下降约 `0.091`。这个数只代表本地采样，不等于最终 OJ 结果。

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
```

本机 1 轮结果：

```text
更新前 RMSE: 1.023281
更新后 RMSE: 1.019420
RMSE 下降:   0.003861
结果有效性:  有效 PASS
update+predict 耗时: 0.517 秒
```

说明：本地 runner 的时间只用于比较优化方向，最终仍以 OJ 环境为准。

## 提交提醒

正式提交时重点检查：

1. 上传/提交的是 `solution.cpp`。
2. 代码里没有文件 I/O、系统调用、硬编码数据路径。
3. 本地完整 benchmark 用 `pixi run task2-track1-cpp-benchmark` 跑，不要误跑 Python 默认模板。
4. 报告由队友写时，可以把“残差偏置 + shrinkage + OpenMP 并行统计”作为算法说明主线。
