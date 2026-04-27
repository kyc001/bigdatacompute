# docs 文档索引

本目录集中存放项目过程文档、交接材料和报告草稿。根目录保留
`README.md`、`PROJECT_STATUS.md`、`INTERFACE.md` 和 `AGENTS.md`，
是为了兼容课程要求与协作工具入口。

## 子目录

| 路径 | 内容 |
| --- | --- |
| `packaging/` | PyInstaller 打包命令和构建说明。 |
| `handover/` | A/B/C 交接文档。 |
| `process/` | 协作规范、代码审查清单、提交检查清单、邮件草稿。 |
| `report_drafts/` | `report_ch*.md` 与 `report_final.md` 草稿归档。 |
| `status/` | 工作日志和阶段记录。 |

## 与根目录的关系

- 根目录 `README.md` 是使用入口和模块说明。
- 根目录 `PROJECT_STATUS.md` 是当前进展入口。
- 根目录 `INTERFACE.md` 是冻结接口契约，不随文档归档移动。
- `report/` 保留 LaTeX 工程、PDF 与图表资源，因为 `plot.py` 和 `report/main.tex`
  当前默认使用该路径。
- `portable_check/` 保留在根目录作为本地换机验证包，包含 `Data.txt`，不默认进入最终提交 zip。
