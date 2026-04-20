# PageRank Coursework

本仓库用于完成高性能计算课程 PageRank 作业。项目目标是在给定图数据集 `Data.txt` 上计算 PageRank，并输出 Top-100 节点到 `Res.txt`，同时满足内存与时间约束，并保留完整的工程化协作流程。

## 目标与约束

- 语言：Python，入口文件固定为 `main.py`
- 算法要求：必须实现 `CSR + Block Matrix`
- 收敛参数：`beta = 0.85`，`eps = 1e-8`
- 工程指标：
  - 内存峰值 `<= 80 MB`
  - 运行时间 `<= 60 s`
- 禁止使用现成 PageRank API
- 最终需要支持 PyInstaller 单文件打包

## 目录结构

```text
.
├─ INTERFACE.md
├─ README.md
├─ CONTRIBUTING.md
├─ code_review_checklist.md
├─ requirements.txt
├─ main.py
├─ blocks.py
├─ mock_graph.py
├─ benchmark.py
├─ tests/
├─ report_ch4.md
├─ report_ch6.md
├─ compile-parameter.txt
├─ cross_platform_checklist.md
└─ HANDOVER.md
```

## 团队分工

| 角色 | 负责人 | 范围 |
| --- | --- | --- |
| A | 算法核心 | `load_graph / power_iteration / dump_top_k`，E1/E2/E8，第 2、3 章 |
| B | 队长 + 工程基建 | `INTERFACE.md`、Git 规范、代码审查清单、`build_blocks`、`iterate_by_block`、`benchmark.py`、打包脚本 |
| C | 数据分析 + 报告统稿 | `analyze_dataset.py / sweep.py / plot.py`，E5/E6，第 1、5、7 章，最终 zip 打包 |

## 快速开始

1. 安装开发依赖：

```bash
python -m pip install -r requirements.txt
```

2. 跑单元测试：

```bash
pytest -q
```

3. 跑主提交版本：

```bash
python main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32
```

4. 跑 benchmark：

```bash
python benchmark.py --main main.py --data Data.txt --out bench.csv --interval 0.05 --runs 3 --modes csr,csr_block
```

## 协作入口

- 接口契约：见 `INTERFACE.md`
- Git / 分支 / 提交 / PR 规范：见 `CONTRIBUTING.md`
- 代码审查清单：见 `code_review_checklist.md`
- 跨平台验证步骤：见 `cross_platform_checklist.md`
