# 任务一交接说明

## HANDOVER 维护规范

这份文档用于实验台账和队友交接，不是最终报告。后续继续调参时请按下面规则维护：

1. **记录实验，不堆材料。** 每次新增内容要写清楚目的、参数、命令、RMSE、耗时、结论。不要复制大段代码、不要写泛泛的模型介绍、不要堆页数。
2. **失败实验也要保留。** 只要能说明“为什么不继续这个方向”，就写进实验记录，方便报告分析和避免重复踩坑。
3. **统一验证口径。** 默认使用 `validation_ratio=0.2, seed=42` 的验证集结果；如果换 seed 或交叉验证，要明确说明。
4. **区分预测文件和验证实验。** `outputs/` 下的预测文件用于提交；验证 RMSE 来自本地 split，不等于最终线上成绩。
5. **更新推荐模型时同步更新三处。** `pixi.toml`、`当前推荐模型`、`实验记录` 都要同步，避免队友照旧命令跑错。
6. **报告只引用关键代码/公式。** 任务一报告可以写 baseline、MF、blend、ensemble 的思路，不需要贴完整 `models.py`。

## 当前进度

任务一已经搭好可运行框架，代码放在 `rec-sys/task1` 下。项目现在不绑定任何环境管理器，队友只需要能运行 Python，并安装 `numpy` 即可。

目前已经完成：

- 解析 `data/train.txt` 和 `data/test.txt`
- 统计数据集基础信息
- 分割训练集/验证集并计算 RMSE
- 实现 bias baseline 模型
- 实现矩阵分解 MF 模型
- 实现 baseline + MF 融合模型
- 实现 residual MF 模型，作为后续可继续尝试的方向
- 实现 ensemble 模型，把强 bias+MF、不同随机种子的 MF、user residual KNN 做保守等权融合
- 生成符合 `ResultForm.txt` 风格的预测文件

你负责的是框架和模型跑通。队友可以在这个基础上继续调参、整理实验表和写报告。

## 目录结构

```text
rec-sys/task1/
├── data/                 # 原始数据，不要改
├── outputs/              # 生成的预测文件，已被 gitignore 忽略
├── src/
│   ├── cli.py            # 命令行入口逻辑
│   ├── data.py           # 数据读取、写出、统计、划分
│   ├── metrics.py        # RMSE 和计时
│   ├── models.py         # baseline / MF / blend / residual / ensemble 模型
│   └── pipeline.py       # 训练、评估、预测流程
├── requirements.txt      # Python 依赖
├── run.py                # 简单启动入口
└── HANDOVER.md           # 本交接文档
```

这里已经把之前多余的 `src/recsys_task1/` 层去掉了，现在就是普通课程作业脚本结构，队友看起来会更直观。

## 环境准备

任选一种 Python 环境即可，例如系统 Python、conda、venv、uv 等。

如果环境里还没有 `numpy`，在仓库根目录执行：

```powershell
python -m pip install -r rec-sys/task1/requirements.txt
```

之后所有命令都从仓库根目录运行。

## 常用命令

查看基础统计：

```powershell
python rec-sys/task1/run.py stats
```

查看训练/测试覆盖率和冷启动情况：

```powershell
python rec-sys/task1/run.py analysis
```

跑 baseline：

```powershell
python rec-sys/task1/run.py evaluate --model baseline --shrinkage 5 --bias-iterations 5
```

跑当前推荐的集成模型：

```powershell
python rec-sys/task1/run.py evaluate --model ensemble
```

生成当前推荐预测文件：

```powershell
python rec-sys/task1/run.py predict --model ensemble --output rec-sys/task1/outputs/predictions_ensemble.txt
```

默认输出文件：

```text
rec-sys/task1/outputs/predictions_ensemble.txt
```

`outputs/` 已经在 `.gitignore` 中，不会误提交。

## 当前推荐模型

当前验证集上最好的配置是集成模型：

```powershell
python rec-sys/task1/run.py evaluate --model ensemble
```

含义：

- 两个强 `baseline + MF` 融合模型负责主预测，使用更强的 bias baseline（`shrinkage=2, bias_iterations=20`）和不同 MF 维度/epoch；
- 一个不同随机种子的 MF 作为低相关性组件；
- 一个 user residual KNN 根据相似用户在同一物品上的残差做邻域修正；
- 四个组件等权平均，避免在单一验证集上用大正负权重过拟合。

目前结果：

```text
Validation RMSE: 16.7694
Train time: about 60-85s on the current machine
```

相比之前的模型：

```text
baseline shrinkage=5: RMSE 17.4690
MF tuned:             RMSE 17.3591
blend tuned:          RMSE 16.9196
ensemble:             RMSE 16.7694
```

所以当前建议优先提交 `predictions_ensemble.txt`。如果时间非常紧，需要快速重跑，可以退回 `task1-blend-tuned`。

## 数据分析结论

数据规模：

```text
训练用户数: 598
训练物品数: 9077
训练评分数: 90854
测试用户数: 610
测试 pair 数: 9982
评分范围: 10 - 100
平均评分: 69.88
稀疏度: 0.9833
```

冷启动情况：

```text
测试集中有 12 个训练集中没出现过的新用户
测试集中有 647 个训练集中没出现过的新物品
测试 pair 中 9028 / 9982 是用户和物品都在训练集中出现过
完全已知 pair 比例约 90.44%
```

分析：

- 大部分测试 pair 都不是完全冷启动，所以 MF 有发挥空间。
- 仍然有新用户和新物品，所以不能只依赖 MF，必须保留 baseline fallback。
- 融合模型比单独 MF 更稳，因为 baseline 能兜住冷启动和低频样本。

## 实验记录

| 模型 | 参数 | Validation RMSE | 训练时间 | 备注 |
|---|---|---:|---:|---|
| baseline | shrinkage=10 | 17.5441 | 0.11s | 初始 baseline |
| baseline | shrinkage=5, bias_iter=1 | 17.4690 | 0.07s | 旧版一次偏置估计 |
| baseline | shrinkage=5, bias_iter=5 | 17.2293 | 0.43s | 迭代偏置后明显提升 |
| mf | epochs=2, factors=8, lr=0.005, reg=0.05 | 17.3591 | 2.06s | 单独 MF 快速最佳 |
| blend | epochs=2, factors=8, lr=0.005, weight=0.5 | 17.2132 | 2.42s | 融合后明显提升 |
| blend | epochs=2, factors=8, lr=0.005, weight=0.6 | 17.2102 | 4.23s | 权重扫描较好 |
| blend | epochs=2, factors=10, lr=0.006, weight=0.6 | 17.1439 | 4.91s | 继续提升 |
| blend | epochs=2, factors=12, lr=0.006, weight=0.6 | 17.0984 | 2.25s | 更高维度有效 |
| blend | epochs=3, factors=10, lr=0.004, weight=0.6 | 17.0523 | 5.43s | 旧一轮最佳 |
| blend | epochs=3, factors=16, lr=0.004, shrinkage=5, weight=0.6 | 16.9637 | 3.85s | 迭代偏置后继续提升 |
| blend | epochs=3, factors=16, lr=0.004, shrinkage=5, weight=0.5 | 16.9557 | 3.79s | 降低 MF 权重更稳 |
| blend | epochs=3, factors=16, lr=0.004, shrinkage=3, weight=0.5 | 16.9196 | 4.01s | 当前最佳配置 |
| residual | epochs=2, factors=8, lr=0.005 | 17.2919 | 2.37s | 可继续研究，但目前不如 blend |
| blend | factors=32, epochs=4, lr=0.003, reg=0.05, shrinkage=2, bias_iter=20, weight=0.45 | 16.8509 | 5.79s | 强单模型 |
| blend | factors=24, epochs=5, lr=0.0025, reg=0.08, shrinkage=2, bias_iter=20, weight=0.45 | 16.8555 | 6.86s | 与 32 维模型互补 |
| user residual KNN | baseline shrinkage=2, bias_iter=20, k=40, shrinkage=1 | 16.9393 | - | 单独不如 blend，但与 MF 互补 |
| ensemble | blend32 + blend24e5 + mf16(seed=123) + userKNN 等权 | 16.7694 | 60-85s | 当前最佳，已生成预测文件 |

## 实验过程与 insights

这部分给队友写任务一报告时参考，不需要照搬全部内容。

### 实验日志格式

后续追加实验时，建议按这个格式写：

```text
编号:
目的:
命令:
参数:
结果:
结论:
下一步:
```

### 详细实验记录

#### T1-E1：bias baseline 调参

- 目的：建立稳健低成本基线，处理冷启动和低频用户/物品。
- 方法：全局均值 + 用户偏置 + 物品偏置，交替更新 user/item bias。
- 结果：
  - `shrinkage=5, bias_iter=1`：RMSE 约 `17.4690`；
  - `shrinkage=5, bias_iter=5`：RMSE 约 `17.2293`；
  - `shrinkage=2, bias_iter=20`：RMSE 约 `17.1764`。
- 结论：强 bias baseline 是最稳定的信号源，后续 MF 和 KNN 都应保留 baseline fallback。
- insight：这个数据集很稀疏，很多物品评分很少，偏置项比纯隐向量更稳。

#### T1-E2：单独 MF 与 baseline+MF blend

- 目的：验证矩阵分解是否能捕捉用户-物品交互。
- 方法：SGD 训练 biased MF，然后与 baseline 做加权平均。
- 结果：
  - 单独 MF 最好约 `17.16-17.36`，波动较大；
  - blend 从早期 `17.21` 逐步调到 `16.9196`；
  - 强 bias baseline + 更高维 MF 的单模型可到约 `16.8509`。
- 结论：MF 有用，但单独使用不够稳；与 baseline 融合后明显更好。
- insight：baseline 负责低频和冷启动兜底，MF 负责已知用户/物品的个性化。

#### T1-E3：residual MF

- 目的：尝试先用 baseline 解释主信号，再用 MF 学残差。
- 结果：当前 residual MF 配置约 `17.2919`，不如 blend。
- 结论：这个方向暂时不作为推荐提交；可能需要更细调学习率、残差权重和正则，但优先级低于 ensemble。

#### T1-E4：user residual KNN

- 目的：利用用户之间的相似残差，补充 MF 之外的邻域信号。
- 方法：在强 baseline 上计算残差矩阵，做用户余弦相似度；预测时用相似用户在同一物品上的残差修正。
- 结果：单独 KNN 最好约 `16.9393`，不如强 blend。
- 结论：KNN 单独不强，但和 MF 的错误模式不同，适合作为 ensemble 组件。
- insight：邻域方法在小用户数场景下很便宜，这里用户只有 598 个，可以直接构造用户相似矩阵。

#### T1-E5：ensemble 与权重选择

- 目的：降低单模型方差，利用不同配置/随机种子的互补性。
- 方法：等权融合：
  - `blend32`：`factors=32, epochs=4, lr=0.003, reg=0.05, shrinkage=2, bias_iter=20, weight=0.45`；
  - `blend24e5`：`factors=24, epochs=5, lr=0.0025, reg=0.08, shrinkage=2, bias_iter=20, weight=0.45`；
  - `mf16(seed=123)`；
  - `user residual KNN(k=40)`。
- 结果：验证 RMSE `16.7694`，当前最佳。
- 为什么不用验证集线性回归权重：离线试过直接拟合权重，RMSE 可以更低，但权重出现很大的正负值，明显有过拟合风险，不适合作为交接主线。
- 结论：当前推荐提交 `predictions_ensemble.txt`。如果时间紧或环境慢，可以退回 `task1-blend-tuned`。

### 成功的 insight

1. **强 bias baseline 很重要。** 从 `bias_iter=1` 到多轮交替更新，baseline RMSE 从约 `17.4690` 提升到约 `17.1764`，说明用户/物品偏置是这个小数据集里最稳定的信号。
2. **MF 单独不够稳，但和 baseline 融合有效。** 单独 MF 最好约 `17.16-17.36`，比强 baseline 并不总是好；加入 baseline 兜底后，blend 可以降到 `16.85-16.92`。
3. **不同模型误差互补比单模型继续堆参数更有效。** `blend32`、`blend24e5`、不同 seed 的 `mf16` 和 user residual KNN 单独都不是碾压级，但等权 ensemble 到 `16.7694`，说明它们犯错位置不完全一致。
4. **保守等权比大正负线性回归更适合交接。** 离线尝试过用验证集直接拟合线性权重，RMSE 可以更低，但权重出现很大的正负数，明显容易过拟合，所以没有写入主模型。

### 失败或放弃的方向

- **更高 epoch 的 MF 不一定更好。** 之前 `epochs=8` 的 MF 反而变差，可能是学习率偏大或过拟合。
- **residual MF 暂时不如 blend。** residual MF 单独约 `17.29`，后续可以继续调，但当前不作为推荐提交。
- **user residual KNN 单独不强。** KNN 最好约 `16.94`，比强 blend 差；它的价值主要在 ensemble 中提供和 MF 不同的邻域信号。

## 队友继续调参建议

优先围绕当前 ensemble 继续小范围搜索：

```powershell
python rec-sys/task1/run.py evaluate --model ensemble
```

建议尝试：

- 在 `pipeline.py` 的 `ensemble` 分支里调整组件权重，建议先试 `blend32/blend24e5/mf16/userKNN = 0.25/0.25/0.25/0.25` 附近的小改动。
- 增加不同 seed 的 MF 组件前要注意训练时间会线性增加。
- 可以继续调 `UserResidualKNNModel` 的 `neighbors`、`residual_shrinkage`，但单独 KNN 不是最强，主要价值是和 MF 互补。
- 快速提交或环境较慢时，用旧命令 `pixi run task1-blend-tuned` 和 `pixi run task1-predict-baseline` 做备选。

注意：epoch 不是越大越好，之前 `epochs=8` 的 MF 反而明显变差，可能过拟合或者学习率偏大。

## 报告写作建议

报告可以按这个逻辑写：

1. 数据集统计：直接使用 `python rec-sys/task1/run.py stats` 和 `python rec-sys/task1/run.py analysis` 的输出。
2. Baseline：全局均值 + 用户偏置 + 物品偏置。
3. MF：用户和物品映射到隐向量，用 SGD 最小化评分误差。
4. Blend：将 baseline 和 MF 加权融合，兼顾稳定性和个性化。
5. Ensemble：多个 MF 配置、不同随机种子和 user residual KNN 等权融合，降低单模型方差。
6. 实验分析：说明单独 MF 有提升，但集成模型 RMSE 最低；原因是测试集大多数 pair 已知，但仍存在冷启动和低频物品。

## 注意事项

- 不要改 `data/` 下的原始文件。
- 不要覆盖 `data/ResultForm.txt`，它只是格式示例。
- 默认预测会四舍五入成整数分数。
- 如果提交系统允许小数，可以给 `predict` 命令加 `--float-output`。
- 最终提交前务必确认预测文件行数和格式。
