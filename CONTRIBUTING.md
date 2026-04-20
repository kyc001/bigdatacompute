# CONTRIBUTING.md

## 1. 分支模型

- `main`
  - 只保留可提交、可打包、可复现实验的稳定版本
- `dev`
  - 日常集成分支
  - A / B / C 合并前先走这里
- `feature-*`
  - 每个成员以任务为单位新开分支
  - 示例：
    - `feature-a-csr-power-iteration`
    - `feature-b-block-memmap`
    - `feature-c-benchmark-plot`

## 2. 提交流程

1. 从 `dev` 拉最新代码
2. 新建自己的 `feature-*` 分支
3. 本地自测通过后推送
4. 提交 PR 到 `dev`
5. 至少 1 人 review 通过后再合并
6. 阶段性稳定后，再由队长把 `dev` 合并到 `main`

## 3. Commit Message 规范

统一使用：

```text
<type>(<scope>): <summary>
```

允许的 `type`：

- `feat`：新增功能
- `fix`：修 bug
- `refactor`：重构，不改行为
- `test`：补测试
- `docs`：文档修改
- `perf`：性能优化
- `chore`：杂项维护

示例：

- `feat(block): add memmap-based iterate_by_block`
- `fix(mock): correct dead-end compensation baseline`
- `docs(interface): freeze stdout json contract`

## 4. PR 模板

每个 PR 描述至少包含以下字段：

- 目标：本次改动解决什么问题
- 变更：改了哪些文件、哪些行为
- 风险：可能影响哪些模块
- 验证：跑了哪些测试、命令是什么
- 待办：还有哪些未完成内容

## 5. 冲突处理

- `INTERFACE.md` 冻结后，禁止直接改签名
- 如需改接口，先提 issue，再由队长统一调整
- 若与他人代码冲突，不要直接覆盖；先 rebase，再人工确认契约是否一致

## 6. 合并红线

- 测试不通过，不合并
- `main.py` CLI 或 stdout JSON 被改坏，不合并
- `Res.txt` 格式不符合要求，不合并
- benchmark 输出 CSV schema 变化，不合并
