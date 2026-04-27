# WORK_LOG.md

## 任务提交记录

| 任务 | 状态 | Commit | 备注 |
| --- | --- | --- | --- |
| P0-1 | 完成 | `a2aa036` | PyInstaller 输出统一为 `main.exe`，文档中的 `pagerank.exe` 口径已替换。 |
| P0-2 | 完成 | `fecfbe1` | `INTERFACE.md` 同步 `iterate_by_block(..., chunk_size=1 << 15)`。 |
| P0-3 | 完成 | `e4b7995` | `HANDOVER.md` stdout JSON 字段表与代码实际输出一致。 |
| P0-4 | 完成 | `53bb5c9` | `requirements.txt` 补齐 matplotlib、pandas、psutil，并改为分组 `>=` 形式。 |
| P0-5 | 完成 | `488d178` | 重新生成并冻结最终 `Res.txt`，记录运行 JSON 和 provenance。 |
| P1-6 | 完成 | `eae019b` | 报告 Markdown 草稿回填 `report_summary.json` 最终数据。 |
| P1-7 | 完成 | `60b8fe3` | 三份 HANDOVER 实验数字同步，`HANDOVER_C.md` 重写为当前状态。 |
| P1-8 | 完成 | `47489a1` | README 目录树更新，`.gitignore` 补充运行/LaTeX 产物规则。 |
| P1-9 | 跳过 | 无 | `tests/test_pagerank_behaviors.py` 存在 `"import json"` probe 脚本字面量；按任务保护条件不删除顶部 import。 |

## P2 验收记录

- `python -m pytest -q`: 11/11 PASS。
- `python main.py --data Data.txt --out .tmp_verify.txt --mode csr_block --K 8 --dtype float32 --beta 0.85 --eps 1e-8`: 与冻结 JSON 的 `top10_signature` 完全一致。
- `python -c "import numpy, scipy, psutil, matplotlib, pandas, pytest"`: import 成功；`micromamba` 环境额外打印 OpenCL vendors 临时文件提示，但命令退出码为 0。
- `.tmp_verify.txt`: 已删除。

## top10_signature 金标准

`75,8686,9678,5104,725,3257,468,7730,7175,5526`

## 已被 ignore 规则覆盖但仍被 git 跟踪的文件

仅列出，不执行 `git rm`，等待人工决定是否清理：

- `000000_C-Cpp示例_000001_李华_第一次作业/实验结果/Res（样例）.txt`
- `000000_Python示例_000001_李华_第一次作业/实验结果/Res.txt`
- `Res.txt`
- `experiments/E1_dense.csv`
- `experiments/E3.csv`
- `experiments/E4.csv`
- `experiments/E7.csv`
- `experiments/E8.csv`
- `experiments/degree_distribution.csv`
- `experiments/sweep_results.csv`
- `report/main.aux`
- `report/main.bbl`
- `report/main.blg`
- `report/main.log`
- `report/main.out`
- `report/main.toc`
- `待填写学号姓名_第一次作业/Res.txt`
- `待填写学号姓名_第一次作业/main.exe`

## 待人工确认

1. 重命名 `待填写学号姓名_第一次作业/` 为真实学号姓名。
2. 确认真实成员学号姓名，并据此生成最终 zip 文件名。
3. 确认 `Report.pdf` 是否需要按最新 Markdown/LaTeX 重新编译。
4. 确认是否清理已被 ignore 但仍被 git 跟踪的文件。
5. 人工最终运行 PyInstaller，并对拍 `main.exe` 与源码版的 `top10_signature`。
