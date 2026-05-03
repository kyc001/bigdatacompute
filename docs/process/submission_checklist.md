# 提交前核验清单

> 状态口径：2026-04-27 收尾检查。未完成项不能在最终提交前跳过。

## 1. 目录与命名

- [ ] 最终 zip 命名严格为 `学号_姓名_学号_姓名_学号_姓名_第一次作业.zip`
- [ ] zip 解压后第一层目录同名，不多套一层无关目录
- [ ] 提交包中包含 `main.py`
- [ ] 提交包中包含 `compile-parameter.txt`
- [ ] 提交包中包含 `Res.txt`
- [ ] 提交包中包含 `Report.pdf`
- [ ] 提交包中包含 Windows 可执行文件 `main.exe` 或约定名称的 exe
- [x] `Data.txt` 已放入提交包，方便老师直接运行
- [ ] `portable_check/` 只作为本地验证包，不放入最终 zip

## 2. 结果文件

- [ ] `Res.txt` 使用 `beta = 0.85`
- [ ] `Res.txt` 共 100 行，若节点不足 100 则为 `min(100, N)` 行
- [ ] 每行格式为 `NodeID Score`
- [ ] Score 固定保留 10 位小数
- [ ] 排序为 Score 降序，同分 NodeID 升序
- [ ] 源码版和 exe 版 Top-10 签名一致
- [ ] `portable_check/run_check.ps1` 输出 PASS

## 3. 程序与性能

- [ ] `python main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32` 可运行
- [ ] stdout 最后一行是合法 JSON
- [ ] 峰值 RSS 低于 80 MB，目标低于 65 MB
- [ ] 运行时间低于 60 s，目标低于 30 s
- [ ] 未调用 `networkx.pagerank`、`igraph.pagerank` 或特征向量 PageRank API
- [ ] main.py 不 import `matplotlib`、`pandas`、`sklearn`
- [ ] PowerShell 中 benchmark 使用 `--modes 'csr,csr_block'`
- [ ] WSL 使用 `/mnt/d/micromamba/micromamba.exe run -n test` 完成源码冒烟

## 4. 报告

- [ ] 报告包含数据集描述
- [ ] 报告包含算法原理与公式
- [ ] 报告包含 CSR 与 Block Matrix 说明
- [ ] 报告包含 dead-end 与 spider-trap 处理
- [ ] 报告包含实验结果与分析
- [ ] 报告包含团队分工与贡献说明
- [ ] 团队分工写明姓名和角色：匡航逸 A、柯云超 B、蒋林瀞 C，并说明 A/B/C 通过抽签确定
- [ ] 报告包含跨平台/可移动验证包说明
- [ ] 报告包含 AI 工具使用说明
- [ ] 图表均来自 CSV 或脚本输出，不手填数据
- [ ] 所有“待补”字样在最终 PDF 前已清除或改成正式结果

## 5. 邮件

- [ ] 收件人：`bigdatacomputing@163.com`
- [ ] 主题含课程作业、队伍成员或 zip 名称
- [ ] 正文说明附件、成员、运行环境
- [ ] 附件为最终 zip
- [ ] 发送前请小组复核 zip 和报告
