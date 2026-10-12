# PaperFuture

PaperFuture 是一个通过 API 运行的论文研究系统：**文献收集与筛选 → 带权有向发展图 → LLM 选择跃迁方向和幅度 → world model 模拟与讨论 → 动态探索，全局最多四轮 → idea 与价值判断**。

当前主入口是 `src/paperfuture`，支持 Agent security、多模态大模型自进化中的可解释性、AI4EDA。模型推理和向量化均调用 API；本机只执行存储、检索排序、图校验和流程控制，无需 GPU。旧 `farfield` 包作为来源基线保留，新流程复用其聊天 API 适配器，不启用旧的技能演化或分支学习流程。工程测试通过不代表研究效果已经验证。

## 运行系统

需要 Python 3.11 或更高版本，基础运行没有第三方 Python 依赖：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
export PAPERFUTURE_BASE_URL='https://YOUR_PROVIDER/v1'
export PAPERFUTURE_MODEL='YOUR_CHAT_MODEL_ID'
export PAPERFUTURE_EMBEDDING_BASE_URL='https://YOUR_PROVIDER/v1'
export PAPERFUTURE_EMBEDDING_MODEL='YOUR_EMBEDDING_MODEL_ID'
# 在本机设置 PAPERFUTURE_API_KEY；不要把密钥放入配置、源码或 Git。
.venv/bin/paperfuture doctor --config examples/paperfuture/agent_security.json
.venv/bin/paperfuture run --config examples/paperfuture/agent_security.json
```

`doctor` 检查本机配置与凭据可用性，不调用付费接口。首次 `run` 从 arXiv 采集，完成筛选与构图，再生成和评估 idea。API 需要兼容 `/chat/completions` 和 `/embeddings`；不同供应商可分别配置。默认聊天请求使用 `max_tokens`，接入前应确认网关支持该字段。所有模型 ID 均由用户明确指定，不自动猜测服务或静默替换模型。

三个领域的可编辑配置位于 [examples/paperfuture](examples/paperfuture)。配置的 `cutoff` 是带时区的文献截止时间，示例日期为 2026-10-12；`run_id` 标识一个不可变研究设置，并保存主流程源码指纹。改变问题、模型、筛选规则、阈值或实现代码，需要使用新的 `run_id`；重放旧实验请保留对应代码版本。`models` 可以为 `extract/rerank/graph/propose/simulate/critique/assess` 单独指定模型；未指定的角色使用 `default`。凭据只允许 `api_key_env` 或 `api_key_file` 引用，相对文件路径按配置文件所在目录解析。

只构建文献与发展图、恢复运行、查看状态、纯离线重放：

```bash
.venv/bin/paperfuture prepare --config examples/paperfuture/agent_security.json
.venv/bin/paperfuture run --config examples/paperfuture/agent_security.json
.venv/bin/paperfuture status var/paperfuture/runs/agent_security-pilot-01
.venv/bin/paperfuture run --config examples/paperfuture/agent_security.json --replay
.venv/bin/paperfuture schemas --output var/paperfuture/schemas.json
```

使用同一条 `run` 命令恢复中断，已完成步骤直接读取记录；重放缺失的 API 响应不会触发网络。API 总预算包括建库、检索向量化、筛选、构图、探索和失败重试。每轮正常执行四个模型任务：提案、模拟、独立批评、综合判断；格式错误修复有单独的有界重试。网络尝试与 token 消耗写入 SQLite 台账。预算入场检查采用保守估计，实际计费以供应商为准；连接丢失时保留“计费未知”，不会宣称恰好计费一次。

## 文献如何进入系统

配置中的 `sources` 支持多个 arXiv 查询源和本地 JSONL 源。自动采集分页发现论文，再请求明确的 **v1** 元数据，不把最新版摘要贴到旧日期上。默认领域配置保存在 [config.py](src/paperfuture/config.py)，也可覆盖 `queries`。`max_results` 是每条查询上限，排序按首次提交时间；较大领域建议拆成多个时间段或子方向查询，并检查覆盖缺口。也可以用 `ids` 指定种子论文，例如：

```json
{"type":"arxiv","ids":["2302.12173","2402.06363","2403.02691","2406.13352"],"version_policy":"v1","full_text":"pdf","require_full_text":true}
```

`full_text: "pdf"` 下载明确版本的 PDF，使用本机 Poppler 的 `pdftotext` 提取文字，不调用本地模型。默认 `full_text: "none"` 只用摘要，用于接通流程和初步筛选；正式机制图建议提供全文。`require_full_text: true` 时缺失全文就失败；否则明确记录 `abstract_only`，不得假定看过完整实验。PDF 表格、公式和复杂版面解析仍需人工检查。可通过 JSONL 导入已校验的正文分节和真实参考文献：

```json
{"paper_id":"your-source:123","version_id":"v1","title":"Paper title","first_public_at":"2024-01-01T00:00:00Z","version_public_at":"2024-01-01T00:00:00Z","source_url":"https://example.org/paper/v1","sections":{"abstract":"Original abstract text","method":"Original method text"},"references":["your-source:100"],"temporal_provenance":"audited version-specific archive"}
```

每行一篇论文版本。相同 `paper_id@version_id` 的既有文本不可覆盖；摘要记录后续增加正文时追加一个独立内容哈希的完整表示，旧快照不变。同一个快照只选择截止日期内最新的已收集版本。所有搜索、取正文、证据和参考文献查询都只访问这个冻结快照。自动源目前使用 v1，若需后续历史修订版本，应先导入经过日期核验的版本记录。跨来源的 DOI/arXiv 身份合并需要导入时统一 `paper_id`，目前不进行不可靠的自动合并。

检索使用 **Okapi BM25 + 独立 API 向量召回 → RRF → LLM 重排 → 相关性与证据质量过滤 → 覆盖增益与重复惩罚的贪心选择**。向量召回会遍历冻结库的全部论文，不依赖 BM25 候选。覆盖指标包含年份、技术主题和机制标签；这是候选池内的覆盖代理，不能证明领域文献已收全。`retrieval.method: "lexical"` 是显式词法消融，不会在向量 API 出错时自动降级。

arXiv 版本接口和访问策略以[官方文档](https://info.arxiv.org/help/api/user-manual.html)为依据；请求间至少间隔约三秒并缓存结果。查询发现仍依赖当前可用的索引，严格历史评测应另行审计论文发现过程及公开时间，而不只检查内容版本。

## 图、跃迁和动态探索

每个节点代表一个技术阶段，保留问题、机制、假设、能力、瓶颈、论文版本和原文片段。允许分支和汇合。LLM 提议发展关系，程序检查证据归属、选中文献覆盖、严格前向的贡献时间见证和 DAG；不凭相似度制造继承关系，不强行连接没有证据的节点。

边权是固定规则映射的技术变化等级：问题、假设、机制、能力和资源要求各取 0/1/2/未知；关键机制或假设达到 2 时为等级 3，同时能力边界达到 2 时为等级 4；其他情况按局部/实质改变分为 1/2。未知维度使历史边权保留未知。等级不是欧氏距离、成功概率或科研价值。LLM 对变化及证据的语义判断仍需人工抽查，程序校验不能替代专家判断。

跃迁器收到完整阶段图、历史证据和上一轮明确标记的条件反馈，选择方向与目标幅度。世界模型分别模拟成功、部分成立和失败；批评器初次判断看不到模拟结果。综合判断必须处理每条异议且不能重写核心 idea。没有实验、不同于主流和审稿偏好不能作为有证据的否决。效用为三项价值评分的归一化平均减去受限的批评惩罚，重复批评取最大严重程度而不累计投票。原始评分、理由、未知项与可行性分别保留。

达到预先声明的价值与可行性阈值则提前停止；两次连续比较无变化时停止；全局最多四轮。每轮在生成前持久化预留，不存在每条分支各跑四轮。中断后恢复同一轮；模拟和假设不会进入历史图。最终输出选择已评估候选，不再调用模型生成第五个 idea，也不会在第四轮强行宣称成功。

## 输出与代码对应

每次运行位于 `var/paperfuture/runs/<run_id>/`：

| 文件 | 内容 |
|---|---|
| `spec.json`, `snapshot.json`, `selected.json` | 配置、不可变文献版本快照、筛选与覆盖记录 |
| `trajectory.json`, `trajectory.mmd`, `trajectory_review.json` | 带权图、Mermaid 图、待人工检查的关系 |
| `graph_context.json` | 完整阶段图和证据上下文 |
| `rounds/01..04/` | 逐轮提案、模拟、独立批评、综合判断、评分和决策 |
| `state.json`, `events.jsonl` | 可恢复状态及按序事件 |
| `IDEA.md`, `final_idea.json` | 最终 idea、价值、可行性、假设、证据、未决问题与停止原因 |
| `api/`, `usage.json` | 请求响应记录、缓存、每次网络尝试与预算消耗；不保存鉴权头 |

| 代码 | 职责 |
|---|---|
| [literature.py](src/paperfuture/literature.py) | 版本库、arXiv/JSONL 收集、冻结、检索与筛选 |
| [graph.py](src/paperfuture/graph.py) | 技术阶段、证据与时间检查、DAG、权重 |
| [exploration.py](src/paperfuture/exploration.py) | 跃迁、条件模拟、独立批评、综合评分、停止规则 |
| [pipeline.py](src/paperfuture/pipeline.py) | 端到端入口、全局轮次、恢复和最终产物 |
| [api.py](src/paperfuture/api.py) | 复用旧聊天客户端、embedding API、持久预算与缓存 |
| [contracts.py](src/paperfuture/contracts.py), [storage.py](src/paperfuture/storage.py) | JSON Schema、引用检查、哈希、原子写入和进程锁 |
| [bench.py](src/paperfuture/bench.py) | 生成完成后的未来文献时间审计与人工标注指标 |

## 评测与验证

```bash
PYTHONPATH=src:tests python3.11 -m unittest discover -s tests -q
python3.11 tools/check_snapshot.py
```

[PaperFuture 测试](tests/test_paperfuture.py) 使用明确标记的合成文献与 HTTP 响应，验证流程和异常边界，不作为科研结果。正式评测需要独立专家核验发展图、预先固定阈值、等预算基线和消融、多次重复与不确定性报告。

2026-10-12 验证记录见 [verification/paperfuture_v0.2.json](verification/paperfuture_v0.2.json)：216 项测试中 207 项通过、9 项因历史外部材料缺失跳过；wheel 构建与安装通过。真实 API 验收从 arXiv 自动取得 3 篇 v1 全文，生成 3 阶段、3 条边的发展图，完成 3 轮探索后按停滞规则停止；结果为 `needs_validation`。成功运行共 21 次模型/向量 API 网络请求，供应商报告 337,976 tokens；随后安装包离线重放未新增请求。验证记录同时列出开发中失败的两次运行及其开销，避免只报告成功调用。该小语料仅用于工程验收，不作为科研效果结论。

无需密钥的整条流程演示可运行 `python3.11 tools/run_paperfuture_demo.py`；加 `--four-rounds` 验证四轮上限。该工具显式使用合成 HTTP 夹具，输出也标记为 SYNTHETIC；正式入口不会自动退回此模式。

`paperfuture bench --cases cases.json --output audit.json` 只读取已经完成的运行与未来标签，检查截止日期和未来论文首次公开时间，并汇总人工机制匹配评分。标签不会进入生成主流程。`cases.json` 是列表，每项包含 `case_id`、`run_directory`、`future_papers`（`paper_id`、`first_public_at`）以及可选 `human_judgments`（`future_paper_id`、`rater_id`、`mechanism_match` 0–4、`rationale`，可附 `novelty/importance/feasibility` 0–4）。这提供评测基础设施，尚未构成发布并验证过的 PaperFutureBench 数据集。

即使输入论文全部早于截止日期，当前 API 模型仍可能知道未来论文。系统会明确输出“证据受限生成，模型知识泄漏尚未排除”；模型名称、发布日期和服务端声明都不能单独证明训练截止时间或模型权重身份。

## 历史来源与基线

本项目于 2026-10-10 从 FarField 提取代码和数据副本，原 FarField 目录未修改。以下目录和迁移记录保留供基线复核，当前端到端入口见上文。

## 目录与核心代码

| 位置 | 内容 |
|---|---|
| `src/farfield/extras/openworld/remote.py` | 历史远跳检索基线、概念候选及来源记录 |
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

迁移时核心远跳和数据模块共 32 个，另复制其递归导入依赖，共 128 个 Python 源模块、12 个测试模块；保留 `farfield` 包名以避免改变基线导入语义。它们都是普通文件，未使用指向原项目的符号链接，运行不要求原项目目录存在。

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

## PaperFuture 研究主线与系统设计

当前研究主线为：**文献收集与筛选 → 带权有向发展图 → LLM 选择跃迁方向和幅度 → world model 模拟与讨论 → 动态探索，最多四轮 → idea 与价值判断。** 三个研究与评测领域为多模态大模型自进化中的可解释性、Agent security 和 AI4EDA 设计，模型能力通过 API 调用。

- [研究设计与评测方案](reports/PAPERFUTURE_RESEARCH_DESIGN_2026-10-12.zh-CN.md)：研究定位、文献与图、PaperFutureBench、代码审查和投稿建议。
- [系统模块与代码设计](reports/PAPERFUTURE_SYSTEM_DESIGN_2026-10-12.zh-CN.md)：输入输出、数据结构、API 分工、讨论权重、全局四轮控制与逐文件改造。
- 系统框架图：[中文预览](figures/paperfuture_framework_zh.png)、[英文 PDF](figures/paperfuture_framework_en.pdf)、[中文 SVG](figures/paperfuture_framework_zh.svg)、[英文 SVG](figures/paperfuture_framework_en.svg)。

使用 `python3 figures/draw_paperfuture_framework.py` 重新生成中英文 PNG、SVG 和 PDF，需要 matplotlib 及中文字体，不访问模型 API；[LaTeX 插图及图注](figures/paperfuture_figure.tex) 可用于论文草稿。六步主流程已实现，接口以 `src/paperfuture` 和上述运行说明为准；早期设计文档中的旧文件改造建议保留为设计记录，科研效果仍待正式评测。

迁移时的 [研究轨迹重构与论文计划](reports/RESEARCH_TRAJECTORY_REFACTOR_AND_PAPER_PLAN.zh-CN.md)、[FarField 远跳机制图](figures/farfield_remote_jump_zh.png) 和 [FarField 框架图](figures/farfield_framework_zh.png) 保留为历史参考。其中的技能、分支学习等设想不属于当前六步主线；历史报告引用的部分原项目审计目录未迁移，不能当作本仓库已包含的实验结果。

本目录保留 FarField 的 `LICENSE` 和 `CITATION.cff`；后者是上游引用信息，不是 PaperFuture 的新论文元数据。
