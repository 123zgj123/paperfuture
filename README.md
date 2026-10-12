# PaperFuture

FarField 远跳与数据处理代码的独立工作副本，创建于 2026-10-10，供后续拆分成完整研究模块和论文项目使用。原 FarField 目录未修改。

目前完成的是代码、依赖、数据快照和测试的迁移；尚未完成架构解耦，也不代表远跳或自进化的科研效果已经得到验证。

## 目录与核心代码

| 位置 | 内容 |
|---|---|
| `src/farfield/extras/openworld/remote.py` | 当前生产远跳检索、概念候选及来源记录 |
| `src/farfield/extras/openworld/discovery.py`、`ideation.py` | 首跳、演化、停滞突破、分支探索与 idea 生成 |
| `src/farfield/extras/openworld/inquiry.py`、`understanding.py`、`sourcecheck.py` | 落点调查、技术脉络与来源核查 |
| `src/farfield/extras/openworld/memory.py`、`response.py`、`dependencies.py` | 分支记忆、结果回应与依赖撤回 |
| `src/farfield/extras/researchspace.py`、`farspace.py` | 研究轨迹表示与概念锚点；保留原有路径，后续统一语义 |
| `src/farfield/extras/arxiv.py`、`concepts.py`、`corpus.py`、`embed.py` | 文献采集、概念／引文构图、时间切分和嵌入 |
| `src/farfield/services/` | 文献源、全文与元数据、索引等依赖 |
| `scripts/build_concepts.py`、`build_corpora.py` | 概念与引文语料构建 |
| `scripts/literature/` | arXiv 索引采集、检索服务及检索评测脚本 |
| `scripts/fetch_lab_data.py` | 原有实验表格数据下载脚本 |
| `concepts/`、`corpora/` | 当前机器已有的概念和引文数据快照及清单 |
| `worlds/`、`contracts/`、`benchmarks/`、`tests/fixtures/` | 运行依赖与回归所需的公开测试材料 |
| `tests/` | 远跳、语料、脉络、分支隔离及相关依赖测试 |
| `reports/` | 迁移时保留的系统设计与论文计划参考 |
| `SOURCE_MANIFEST.json` | 逐文件原始 SHA-256、来源版本、工作区状态与复制分类 |

核心远跳和数据模块共 32 个，另复制其递归导入依赖。当前共 128 个 Python 源模块、12 个测试模块；保留 `farfield` 包名以避免复制时改变导入语义。它们都是普通文件，未使用指向原项目的符号链接，运行不要求原项目目录存在。请使用独立虚拟环境，避免与另一个已安装的 `farfield` 包混用。

## 本地验证

在本目录执行，基础离线代码使用 Python 3.11 及以上：

```bash
python3 tools/check_snapshot.py
PYTHONPATH=src:tests python3 -m unittest discover -s tests -q
```

第一条核对迁移文件哈希、检查导入来源，并使用随副本附带的小概念图做一次离线检索。不访问 API，不评价科学能力。之后主动修改迁移文件会使原始哈希核验报告差异；清单是来源基线，不应为了消除提示覆盖它。

迁移时验证：263 个复制文件哈希一致，离线检索返回 3 个概念候选，未从原 FarField 目录导入模块。189 项测试中 180 项通过、9 项跳过、无失败。跳过项依赖原项目历史全文材料、AgentIdeaBench 运行或 SciFact 实测记录，这些记录没有复制。详情见 [副本核验](verification/snapshot.json) 和 [测试记录](verification/tests.json)。

也可以在单独虚拟环境中用 `python3 -m pip install -e .` 进行本地开发。当前以完整工作目录／editable 方式使用；语料位于项目根目录，尚未设计可携带全部数据的 wheel 发布包。

## 数据和运行边界

- 已有概念图、引文图、manifest 和 heldout 文件一起复制。注册表中部分大型图在原项目就缺失，没有补造；默认检索可能使用 `attn-concepts-s1` 演示语料。
- 没有复制凭据、API 交互记录、`var/` 运行缓存、原项目 Git 历史、Web UI 或外部 benchmark 仓库。需要的外部索引和大型数据应另行配置或生成。
- 采集脚本实际运行时可能访问网络；检索重排、表格数据下载等可选路径需要额外科学计算／模型依赖。此处没有自动下载或调用付费服务。
- `scripts/literature/eval_retrieval.py` 保留原脚本，其中模型评分不能充当独立科研效果证据。
- 数据的原始来源和使用条件仍适用；代码 LICENSE 不自动覆盖所有第三方论文或数据内容。

## 后续独立模块的主线

以“技术脉络 → 机制迁移 → 落点研究 → 实际验证 → 分支记忆与技能 → 下一跳”为中心。先统一远跳入口与数据接口，再将运行依赖缩小；保留分支隔离和证据来源。论文主要验证后续研究效用，而非概念距离或模块分数。

设计参考：[研究轨迹重构与论文计划](reports/RESEARCH_TRAJECTORY_REFACTOR_AND_PAPER_PLAN.zh-CN.md)。复制的历史报告引用了原项目的部分审计目录，这些原始实验轨迹没有迁移；历史相对链接不应当作本项目已包含的实验结果。

远跳机制图（仅展示 idea 如何形成）：[中文预览](figures/farfield_remote_jump_zh.png)、[英文 PDF](figures/farfield_remote_jump_en.pdf)、[中文 SVG](figures/farfield_remote_jump_zh.svg)、[英文 SVG](figures/farfield_remote_jump_en.svg)。依次展示技术瓶颈、外部机制候选、结构映射、落点调查与新方法设计；不包含完整系统模块。使用 `python3 figures/draw_remote_jump.py` 重新生成，[LaTeX 图注](figures/remote_jump_figure.tex) 明确了设计与当前实现的边界。

完整系统框架图：[中文预览](figures/farfield_framework_zh.png)、[英文论文版 PDF](figures/farfield_framework_en.pdf)、[可编辑 SVG](figures/farfield_framework_en.svg)。图示为新架构设计，虚线标明待补齐的技能闭环和固定学习快照；科研效果仍需验证。使用 `python3 figures/draw_framework.py` 可重新生成中英文 PNG、SVG 和 PDF（需 matplotlib 与脚本指定的字体）；[LaTeX 插图及图注](figures/framework_figure.tex) 可直接用于论文草稿。

本目录保留 FarField 的 `LICENSE` 和 `CITATION.cff`；后者是上游引用信息，不是 PaperFuture 的新论文元数据。
