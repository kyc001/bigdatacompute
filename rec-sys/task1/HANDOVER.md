# 任务一交接说明

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
│   ├── models.py         # baseline / MF / blend / residual 模型
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

跑当前推荐的融合模型：

```powershell
python rec-sys/task1/run.py evaluate --model blend --epochs 3 --factors 16 --lr 0.004 --reg 0.05 --shrinkage 3 --blend-weight 0.5 --bias-iterations 5
```

生成当前推荐预测文件：

```powershell
python rec-sys/task1/run.py predict --model blend --epochs 3 --factors 16 --lr 0.004 --reg 0.05 --shrinkage 3 --blend-weight 0.5 --bias-iterations 5 --output rec-sys/task1/outputs/predictions_blend_tuned.txt
```

默认输出文件：

```text
rec-sys/task1/outputs/predictions_blend_tuned.txt
```

`outputs/` 已经在 `.gitignore` 中，不会误提交。

## 当前推荐模型

当前验证集上最好的配置是融合模型：

```powershell
python rec-sys/task1/run.py evaluate --model blend --epochs 3 --factors 16 --lr 0.004 --reg 0.05 --shrinkage 3 --blend-weight 0.5 --bias-iterations 5
```

含义：

- `baseline` 负责学习全局均值、用户偏置、物品偏置
- `MF` 负责学习用户和物品的隐向量
- `blend` 把 baseline 和 MF 的预测做加权平均
- `--shrinkage 3` 表示偏置正则化较弱，允许用户/物品偏置更充分地表达
- `--bias-iterations 5` 表示 baseline 的用户偏置和物品偏置交替更新 5 轮
- `--blend-weight 0.5` 表示 50% 使用 MF，50% 使用 baseline

目前结果：

```text
Validation RMSE: 16.9196
Train time: about 4.0s
```

相比之前的模型：

```text
baseline shrinkage=5: RMSE 17.4690
MF tuned:             RMSE 17.3591
blend tuned:          RMSE 16.9196
```

所以当前建议优先提交 `predictions_blend_tuned.txt`。

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

## 队友继续调参建议

优先围绕当前最佳配置继续小范围搜索：

```powershell
python rec-sys/task1/run.py evaluate --model blend --epochs 3 --factors 16 --lr 0.004 --reg 0.05 --shrinkage 3 --blend-weight 0.5 --bias-iterations 5
```

建议尝试：

- `--epochs`: 3, 4, 5
- `--factors`: 12, 16, 20, 24
- `--lr`: 0.003, 0.004, 0.005
- `--reg`: 0.03, 0.05, 0.08
- `--shrinkage`: 2, 3, 5
- `--blend-weight`: 0.4, 0.5, 0.6
- `--bias-iterations`: 5, 8, 10

注意：epoch 不是越大越好，之前 `epochs=8` 的 MF 反而明显变差，可能过拟合或者学习率偏大。

## 报告写作建议

报告可以按这个逻辑写：

1. 数据集统计：直接使用 `python rec-sys/task1/run.py stats` 和 `python rec-sys/task1/run.py analysis` 的输出。
2. Baseline：全局均值 + 用户偏置 + 物品偏置。
3. MF：用户和物品映射到隐向量，用 SGD 最小化评分误差。
4. Blend：将 baseline 和 MF 加权融合，兼顾稳定性和个性化。
5. 实验分析：说明单独 MF 有提升，但融合模型 RMSE 最低；原因是测试集大多数 pair 已知，但仍存在冷启动。

## 注意事项

- 不要改 `data/` 下的原始文件。
- 不要覆盖 `data/ResultForm.txt`，它只是格式示例。
- 默认预测会四舍五入成整数分数。
- 如果提交系统允许小数，可以给 `predict` 命令加 `--float-output`。
- 最终提交前务必确认预测文件行数和格式。
