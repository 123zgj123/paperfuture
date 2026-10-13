# PaperFuture

PaperFuture v0.4.1 沿用六步主线：**文献收集与筛选 → 带权有向发展图 → LLM 选择跃迁方向和幅度 → world model 模拟与讨论 → 动态探索，全局最多四轮 → idea 与价值判断**。

支持 Agent Security、多模态大模型自进化中的可解释性、AI4EDA。研究推理通过 API；向量化可使用外部 API，也可使用数据盘上的本地开放模型。默认流程无需 GPU；本地向量化使用额外的 CPU 和内存。运行完成表示流程完成，不能据此宣称 idea 已被实验验证。

## 数据盘与启动

所有新论文、PDF、提取文本、数据库、向量缓存和运行记录放在数据盘。当前机器约定：

```bash
export PAPERFUTURE_DATA_ROOT=/mnt/efs/zhangguijia/paperfuture
```

默认入口使用该数据卷位置，不自动回退到系统盘。可用环境变量或 `--workspace` / `--data-root` 显式指定其他数据卷。旧 `var/paperfuture` 已迁至数据盘的 `legacy-v02`，本机原路径保留符号链接；符号链接与真实数据都不提交 Git。克隆代码仓库不会自动获得私有运行数据。

需要 Python 3.11+，PDF 文字提取需要 Poppler 的 `pdftotext`：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
export PAPERFUTURE_BASE_URL='https://YOUR_PROVIDER/v1'
export PAPERFUTURE_MODEL='YOUR_PRIMARY_MODEL_ID'
export PAPERFUTURE_SECONDARY_MODEL='YOUR_SECOND_MODEL_ID'
export PAPERFUTURE_CRITIC_MODEL='YOUR_THIRD_MODEL_ID'
export PAPERFUTURE_EMBEDDING_BASE_URL='https://YOUR_PROVIDER/v1'
export PAPERFUTURE_EMBEDDING_MODEL='YOUR_EMBEDDING_MODEL_ID'
# 首次使用先按下文“文献收集与筛选”建库并发布固定版本。
export PAPERFUTURE_CORPUS_MANIFEST="填写 corpus-publish 输出的 manifest_path"
# 在本机设置 PAPERFUTURE_API_KEY；不要把明文密钥写入配置或 Git。
.venv/bin/paperfuture doctor --config examples/paperfuture/agent_security.json
.venv/bin/paperfuture run --config examples/paperfuture/agent_security.json
```

示例入口读取已发布的固定语料，再顺序执行筛选、构图、跃迁、模拟讨论和输出。默认 `literature_mode=frozen`，生成过程中不采集论文或补全文；采集和增量更新独立执行。API 兼容 `/chat/completions` 与 `/embeddings`；每个角色均可独立设置服务地址、模型和凭据引用。`doctor` 只检查本机配置。模型列表不等于实际可调用性，多模型接入应先做小型调用验证。程序检查配置中模型标识的多样性，不能独立证明网关背后的模型权重身份。

推理模型的参数需要显式配置。例如 GPT-6 Astra 可设置 `"reasoning_effort": "max"`、`"output_token_parameter": "max_completion_tokens"`、`"temperature": null`；`null` 表示不发送温度。其他服务可设置数值温度和 `max_tokens`。程序不偷偷替换模型，实际请求参数参与缓存身份并保存在回执中；修改推理强度后不会复用旧强度的准备结果。输出额度包含供应商计入的推理 tokens，并不保证全部用于可见 JSON。

长回答可显式设置 `"stream": true`，请求流式输出及用量尾帧，减少网关等待完整响应引发的超时。程序同时保存原始流和重组后的 JSON，要求完整结束标记，拒绝截断响应；收到完整响应不等于通过后续科学契约校验。重放使用原回执，不重新请求服务。

不完整流会作为传输失败进入原适配器的最多三次重试，每次均独立预留预算。`api/transport_failures` 记录是否收到结束标记、事件数量和字节数，不保存失败响应正文。准备阶段发生致命错误时取消尚未开始的并发任务，保留已经完成的回执。

本地向量化部署与配置见 [本地向量模型](docs/local_embedding.md)。本地接口只监听 `127.0.0.1`，模型权重、隔离 Python 环境和向量缓存均位于数据盘。`models.embedding.input_prefixes` 分别配置查询和论文前缀；前缀参与实际请求和缓存身份，不能将不同模型或不同前缀的向量混用。

模型调用中的 `E1` 等证据别名只在当次请求中有效。保存产物时，结构化引用和正文中的引用都转换为稳定的证据 ID；下一轮发送时再按本轮证据表编码，避免引用编号漂移。格式修复允许更正证据编号，但不能借机替换候选的研究机制。

新的运行使用 `examples/paperfuture` 中的 v0.4 配置和本机凭据引用。v0.3 运行与快照保留为历史基线，不能用新代码直接续跑它们。示例：

```bash
PYTHONPATH=src python3.11 -m paperfuture run \
  --config examples/paperfuture/agent_security.json \
  --workspace "$PAPERFUTURE_DATA_ROOT"
```

`prepare` 执行到发展图；同一 `run` 命令恢复中断；加 `--replay` 禁止模型和文献网络请求。`run_id` 固定问题、模型、参数与源码指纹，修改后需要新的 ID。重放历史记录须使用对应代码版本。所有重试受次数与总预算约束，网络失败也记录；没有账单价格表时只报告供应商返回的 tokens，不虚构货币成本。

v0.4 的契约新增历史贡献角色、逐维证据角色、假设处理决策、下一跳检索计划、候选重复检查和最终复核。旧配置中的 `policy.stagnation_epsilon` 从未参与有效判断，现已删除；迁移时去掉它并使用新的 `run_id`。停滞依据机制或已解决问题的实质变化，不用 0.02 等数值冒充语义距离。v0.4 不静默转换旧研究产物。

向量按输入内容和模型标识缓存。重排、证据提取和建图还可复用同一个 `replicate_id` 下输入提示、结构、模型及服务地址完全相同、已有成功回执且再次通过当前校验的结果；`usage.json` 记录原始回执和复用来源，不将复用伪装成新请求。新的跃迁、模拟、批评、讨论和综合判断不跨运行复用。需要独立重复实验时使用不同的 `replicate_id`，评测时另行报告准备成本与探索成本。

向语料中插入论文可能改变向量批次，降低整批缓存命中率。维护工具 `tools/repack_embedding_cache.py` 可在运行停止并取得独占锁时，按完全相同的文本、模型与服务地址从成功回执逐行复制向量，形成新批次缓存。它假设服务对各输入独立嵌入，必须显式调用；不重算向量，不覆盖旧记录，发现同文本向量冲突则不复用，逐行保留原始回执和哈希来源。此维护工具不参与论文或 idea 的科学判断。

示例配置按语料规模设置有限预算：Agent Security 最多 800 次调用／800 万 tokens，多模态最多 1,000 次／1,200 万 tokens，AI4EDA 最多 600 次／600 万 tokens。这些是上限，不是实际账单或完成保证；新模型、更多节点、格式重试和独立复现实验会改变用量。首次运行需要为全库向量化及每条候选边的判断、复核留出预算，不能用命中缓存后的少量调用冒充完整成本。

## 1. 文献收集与筛选

可先独立建库，不调用 LLM：

```bash
.venv/bin/paperfuture collect --domain all \
  --data-root "$PAPERFUTURE_DATA_ROOT" \
  --cutoff 2026-10-12T00:00:00Z --fulltext-limit 100
```

建库完成后只需发布一次：

```bash
.venv/bin/paperfuture corpus-publish --domain agent_security \
  --data-root "$PAPERFUTURE_DATA_ROOT" --cutoff 2026-10-12T00:00:00Z \
  --input "$PAPERFUTURE_DATA_ROOT/datasets/agent_security/20261012/papers.jsonl"
```

将输出的 `manifest_path` 作为 `PAPERFUTURE_CORPUS_MANIFEST`。配置固定具体版本，不在实验中跟随 `latest.json`。本机默认配置已切换到固定语料；每周维护只检查新增时间窗口和待补录文献，并仅向量化缓存缺失的文本。更新失败保留旧配置，已开始的实验继续使用原快照。部署和恢复命令见 [固定语料与每周更新](docs/corpus_updates.md)。

采集按领域的明确子方向查询与年份分层。Agent Security 与多模态领域从 2015 年开始，AI4EDA 从 2010 年开始；没有检索结果的年份明确列为缺口。2025—2026 是当前配置的最近两个日历年，2026 仅到截止日。大年份再按月份分页，失败的查询会尝试拆成更小且不重叠的日期窗口；完整记录请求、匹配数量、获取数量、失败与恢复过程和截断情况。查询覆盖不等于全领域覆盖，当前索引也不能被冒充为历史时刻的索引。

发现阶段只取得论文 ID；正文与摘要请求明确的 arXiv **v1**。不把最新修订内容贴到旧年份上。元数据保存标题、作者、类别、DOI／期刊信息（来源有则保存）、首次公开日期、版本日期和来源。全文层保留原始 PDF、提取文本、SHA-256 和来源响应，参考文献中的 arXiv 标识只做机械提取，仍可能不完整。公式、图表、扫描件和复杂排版不能保证被文本提取完整表达。

全文优先覆盖早期年份，并在最近两年之间分配名额、按子方向选择论文；已有全文不会因调整抽样而删除。全文补充在独立建库或周更维护时执行，正常生成直接读取已有证据。未取得的全文记录为 `abstract_only`，不能假定已经看过全文实验。

数据目录如下：

```text
$PAPERFUTURE_DATA_ROOT/
  datasets/<domain>/<cutoff-date>/
    spec.json, discovery.json, manifest.json
    version_records.json, papers.jsonl, literature.sqlite
    snapshot-<content-hash>.json
  cache/arxiv/              原始 Atom、PDF、文本及来源回执
  corpora/<domain>/releases/<release_id>/  不可变 manifest.json 和 papers.jsonl
  corpora/<domain>/latest.json, active.json  最新发布与已激活版本
  cache/embeddings/         原有批次向量与回执来源
  cache/embedding_rows.sqlite  按精确文本、前缀和模型复用的向量
  maintenance/             每周维护审计与日志
  configs/                 本机运行配置，只引用密钥文件
  runs/<run_id>/           完整运行记录
  legacy-v02/              旧数据与旧运行
```

`manifest.json` 是数据清单，包含年份分布、最近两年数量、全文数量及未完成项。`papers.jsonl` 是可以直接输入系统的版本记录；SQLite 中不同内容表示不能重复计作不同论文。自动源使用 v1，其他历史修订版本可通过带日期证明的 JSONL 导入。相同版本已保存的文本不能被修改，补全文时追加内容表示，旧快照保持不变。

2026-10-12 截止快照已经采集并通过文件哈希与日期核验：

| 领域目录 | 论文版本数 | 已归档全文 | 有论文的年份 | 2025—2026 占比 |
|---|---:|---:|---|---:|
| `agent_security/20261012` | 3,582 | 158 | 2016—2026 | 73.2% |
| `multimodal_self_evolution_xai/20261012` | 9,115 | 100 | 2015—2026 | 72.2% |
| `ai4eda/20261012` | 810 | 100 | 2012—2026，2015 无匹配 | 54.6% |

三领域合计 13,507 条领域记录，跨领域去重后为 13,438 个论文版本。当前声明查询内没有未完成年份、缺失版本或失败的全文下载；“已归档全文”是详细证据子集，其余是完整元数据与摘要，不能宣称全部论文都已获取全文。Agent Security 的 2015 年、AI4EDA 的 2010、2011、2015 年无匹配结果，清单保留这些缺口。PDF 单文件下载上限为 128 MB，所有 PDF 和提取文本都位于数据盘。公开的数量与完整性记录见 [数据验收](verification/paperfuture_data_v0.3.json)。

筛选算法为：**当前问题 BM25 + 全库独立向量召回（本地模型或 API）+ 明确分支的多查询召回 + 已有引用回溯 → RRF → 历史／近期分配与分支覆盖 → LLM 双重相关性重排 → 原文提取 → 证据质量筛选 → 覆盖与冗余控制**。`retrieval.facet_queries` 明确当前问题的技术分支；`history_paper_ids` 可声明经过审查、需要进入重排的历史论文 ID，但不能绕过相关性和证据筛选。固定模式只回溯库内已有引用；缺失的历史种子和被引用论文写入 `literature_update_requests.json`，由后续独立更新补录，不能中途改变本次实验的论文或向量。显式 `literature_mode=acquire` 才恢复旧的运行前补全文行为；不要用它作为日常生成入口。

直接相关性和历史贡献分别评分，历史贡献区分使能机制、转折点、反例和评测基础。机制覆盖使用提取的 `mechanism_family`，不再截取描述前八个词。近期占比默认 65%，含义是有覆盖约束、允许候选不足时回填的**预算分配目标**，不是精确比例或硬上限。年份本身不能压过技术分支和有来源的引用桥接；无关旧论文不为凑年份而通过筛选。`selected.json` 的 `funnel` 为每篇论文记录未召回、候选预算排除、重排拒绝、证据不足、提取相关性拒绝、最终预算排除或已入选；`recall_audit` 记录查询、引用来源、未收集的声明文献和分支／年份候选。

证据提取使用有边界的原文片段视图并保留精确偏移。模型侧使用 `E1` 等短证据别名，保存时通过封存的映射准确还原内容哈希 ID；未知引用直接拒绝，不用模糊匹配补造引用。历史清单和查询都需要领域审查，算法并不保证完整召回；GLM 已发现的七篇案例属于开发诊断样本，不能再作为独立测试集宣称覆盖改善。

`retrieval.concurrent_calls` 控制互不依赖的论文提取并发，默认 3、最多 8。请求共享同一个事务预算，结果仍按候选顺序汇总。提取与建图并发遇到致命错误时取消排队工作，已发出的请求完成后保留回执；轮间探索依然按依赖顺序执行。

采集遵循 [arXiv 版本查询接口](https://info.arxiv.org/help/api/user-manual.html)，同一归档目录的请求统一限速、加锁并缓存。`sources` 默认支持不可变发布版 `corpus` 和本地 `jsonl`；网络源 `arxiv_dataset` 与 `arxiv` 必须显式开启 `literature_mode=acquire`。`lexical` 是显式消融，向量调用失败不会静默退化成词法检索。

## 2. 带权有向发展图

每个节点描述问题、机制、假设、已展示能力、局限、瓶颈、论文版本与证据。首次抽取强制一篇论文一个节点，随后所有节点统一接受等价性合并审查，避免批次内提前混合机制。默认保留独立阶段；只有问题、具体机制、假设和能力边界逐项相同，且两篇论文都有对应引文，才允许合并。控制器保留原始技术描述和证据，不能将木马训练、提示注入、过滤和能力授权统称为同一种方法。

分批抽取提出的证据缺口保存在 `batch_evidence_gaps`，附上当批论文范围。例如旧论文批次没有 2026 年证据，不能据此宣称整张发展图缺少 2026 年工作。全局 `coverage_gaps` 使用完整筛选结果的覆盖审计；局部判断仍供探索参考，但不作为全领域不存在某项研究的证据。

默认不超过 24 个节点时审查全部可行时间正向候选对；较大图分别按问题／瓶颈、机制、假设／能力及真实引用召回，并确定性抽查未召回对。`candidate_pair_audit` 保存每对的来源、抽查项与未审查项，未审查不等于不存在关系。实际引用不等于技术继承；所有接受边仍须通过严格时间见证与 DAG 检查。允许分支、汇合、孤立节点，不为连通性强造关系。穷举候选会增加 API 成本，应按节点数和批次设置预算。

每条边由 LLM 对五个维度分别评分：问题、假设、机制、能力边界、资源要求，各 **0—10** 分；另直接给出 **0—100** 的总体技术变化分及不确定区间。每个非零维度必须附自己的依据和证据。0 表示无变化，1—2 表示局部调整，3—4 为实质局部重设，5—6 为结构变化，7—8 为核心机制／假设／能力边界改变，9—10 为极强的范式变化判断。总体分不是五维机械求和，也不是成功概率或科研价值。

每维引文分别标记源机制、目标机制和比较段落。比较段落可以来自一篇论文，只要确实讨论了两端；程序不机械要求每个维度必须引用两篇论文。资源变化须说明增减或权衡，成本增加不等于贡献更大。引文归属检查不能替代语义正确性判断。

独立复核隐藏初评的数值、置信标签、总体评价、逐维理由及变化描述，只提供原始证据、节点及待审关系。先审关系，再审幅度。只有 `supported` 关系进入历史图；最终描述、分数、解释与置信标签来自同一次复核，初评和复核分别留档。两次总体评分差异大于 `graph.max_rating_disagreement`（默认 15）时，保留已支持的关系，但总体分置为 `null` 并标记待裁决；不把两次数值的跨度伪装成统计置信区间。没有任何边通过关系复核时停止并要求改善证据。

`rating_anchors` 可提供同领域、同关系类型的专家审定标尺，必须注明 `split=calibration`、标注者、源／目标论文版本、五维与总体分、理由及原文引文，论文仍须符合截止时间。默认不带专家锚点，评分状态明确为未校准。`bench-graph` 独立读取留出标注，报告关系精确率／召回率、关系类型一致性、五维误差、总体 MAE、Spearman 与成对排序一致性，并拒绝标尺与留出集重叠。它不会生成专家标签，人工分歧和模型不打分均须报告。

默认每次调用只判断或复核一条关系，避免多关系批次中的漏判与引用错配；可在验证模型可靠性后调整 `graph.edge_batch` 和 `graph.review_batch`。`graph.concurrent_calls` 默认 3，范围 1—8；独立关系可以并发处理，结果按输入顺序汇总，每个请求隔离上下文，并在同一事务账本中预留总预算。阶段 ID 与原文引用 ID 分别标识，逐维引用由控制器汇总，但每一个引用和时间见证都必须来自对应端点的真实原文。

## 3. 选择跃迁

输入包含完整阶段／边概览、精确引文 ID、证据清单、本轮补取的段落、条件反馈和所有前轮经验。每篇图中论文至少保留一段证据；导航目标涉及的图引文在本轮预算内从冻结快照展开，尚未展开的 ID 单独列出，不能冒充已经读过的段落。提案说明现有机制、改变的原则、因果链、最近似工作、为什么不只是叠模块，以及能够证伪优势的最小对照实验。`retrieval_use` 逐篇说明本轮检索结果被使用、因不相关而排除或证据不足；允许有理由地不使用新检索结果。

组件组合只有在提出明确的交互机制，并且能通过对照／消融区分交互效应与各组件独立效应时，才可能构成有意义的跃迁。批评和综合判断必须对照全图中最近似的已有方法，而不能只比较提案自行挑选的旧基线；综合判断识别为 `already_known` 后，控制器将有效 novelty 强制为 0。若判为 `integration_only` 或 `unclear`，有效 novelty 最高为 2/4，不能直接给出强推荐。历史边的技术变化幅度不能被当成未来 idea 的新颖性。已有工作识别本身仍依赖模型：真实反例检查中出现过理由承认方法已发表、分类却仍给出机制变化的漏判，验收报告保留失败结果。结构约束不能保证全球新颖性，需要专门的反例集和专家盲评。

## 4. 多模型 world model

`world_model.simulators` 指定多个模型独立产生成功、部分成立和失败情景；批评模型独立检查机制、冲突、未知项和最小实验。初次批评看不到模拟输出。随后各模拟模型阅读相互独立的分析，讨论分歧、修改条件预测、提出区分性实验和下一跳启示；综合判断保留未解决分歧，不按多数投票决定科学事实。

真实证据冲突、缺少验证、审稿偏好分别处理。未验证和非主流不能单独否决 novelty。严重且适用的证据冲突不能因为综合判断漏抄引用而消失；驳回这种冲突必须给出证据。重复异议按最大严重度处理，批评权重默认 0.2、最多 0.35。模型身份、独立响应、讨论及最终处理全部保留。模拟始终是条件预测，不是实际实验。

## 5—6. 累积经验、最多四轮和最终结果

每轮保存带来源 ID 的成功条件、失败预测、争议、下一瓶颈和实际依赖。经验的条件档案与本轮活跃假设分开：`learning_trace` 说明如何使用经验；`assumption_decisions` 对涉及的旧条件明确保留、替换或不适用。引用批评来替换旧前提，不会把该前提重新带回方案。保留时须沿用原 ID、内容和认识状态；`continue_leap` 仍须继承作为条件基础的父方案假设，不能把模拟成功升级为历史事实。

综合判断的下一步计划明确未决问题、改变哪个因果环节、什么结果算通过，以及瓶颈／替代机制／证伪条件三条简短查询。`revise` 优先解决指定问题，`switch` 优先检索替代机制，`continue_leap` 沿条件依赖探索；不会再把整段模拟叙述追加进 BM25。所有查询仍在**同一冻结快照**内执行。

API 输入使用统一的假设表，经验通过假设 ID 引用它；持久化记录仍保留完整依赖。这是可以无损还原的去重，避免同一组假设随每条讨论经验反复复制、在第三或第四轮撑满上下文，不丢弃较早轮次的经验。

每个候选生成后、昂贵模拟前，先检查是否与此前因果机制和干预重复。完全相同的候选由程序识别，改写或语义重复交由评估模型比较。切换必须改变瓶颈、原则或干预，仅换 stage ID 或增加一个测试不足以证明切换。重复候选仍消耗本轮名额，但跳过模拟讨论；不会在同一名额下偷偷再生成候选。JSON 格式修复若改动已给出的科学内容，运行中断并保留回执，不接受替代 idea。

满足机制、价值与可行性条件可提前停止，连续两次无进展则停止，全局最多四个候选轮次。最终对既有早期候选复核后续批评的适用条件、旧前提状态与新颖性反证，重新计算适用证据冲突，再排序。原始逐轮评分不覆盖；后续版本替换了一个前提不自动证明旧方案错误；未知仍不会作为 novelty 惩罚。最终复核不生成第五个 idea，可输出 `needs_validation` 或 `not_recommended`。

## 运行产物与验证

| 产物 | 含义 |
|---|---|
| `snapshot.json`, `selected.json` | 冻结文献、筛选、年份分布、缺口 |
| `trajectory.json`, `trajectory_review.json` | 阶段、通过复核的边、原提案、拒绝原因、五维评分与区间 |
| `graph_context.json` | 全图概览 |
| `rounds/01..04/evidence_context.json` | 本轮检索问题、文献与精确证据 |
| `rounds/01..04/experience_memory.json` | 所有前轮的条件经验与依赖 |
| `proposal/proposal_check/simulation/critique/discussion/assessment/decision.json` | 各轮对应步骤；重复候选没有模拟产物 |
| `final_reviews/01..03.json` | 既有早期候选对后续发现的条件适用性复核 |
| `IDEA.md`, `final_idea.json` | 最终候选、价值、条件、可行性、分歧、证据和停止原因 |
| `api/usage.sqlite`, `usage.json` | 每次网络尝试、未知计费、tokens 与向量复用来源 |

```bash
PYTHONPATH=src:tests python3.11 -m unittest discover -s tests -q
python3.11 tools/check_snapshot.py
python3.11 tools/run_paperfuture_demo.py --workspace "$PAPERFUTURE_DATA_ROOT/offline-demo-v04" --four-rounds
python3.11 tools/audit_paperfuture.py --data-root "$PAPERFUTURE_DATA_ROOT" \
  --run "$PAPERFUTURE_DATA_ROOT/runs/YOUR_COMPLETED_RUN_ID" \
  --output "$PAPERFUTURE_DATA_ROOT/verification/audit.json"
```

`audit_paperfuture.py` 不调用网络；它重新核验导出的论文唯一性、版本日期、PDF／文本哈希、数据盘路径，以及已完成运行中的图约束、每轮证据、累积经验、评分和停止决策。它证明记录与软件契约一致，不代替专家判断论文内容或 idea 质量。

离线演示使用明确标记的合成论文与 HTTP 响应，只检验软件行为。正式入口不会退回合成数据。新增回归检查覆盖证据冲突遗漏、模块堆叠、错误关系剔除、独立评分、多模型调用、全部前轮经验、动态检索、时间平衡、全文范围与原始 PDF 完整性。

v0.4.1 固定语料验收：300 项测试中 291 项通过、9 项跳过；独立安装包的四轮离线运行与禁止网络的结果重放通过。真实 Agent Security 3,587 篇固定语料及其向量在禁止网络下复用成功，新增 API 调用为 0。每周一 02:00 UTC 的持久化定时器已启用，详情见 [固定语料验收](verification/paperfuture_v0.4.1.json)。这证明数据维护与缓存行为，不代表新 idea 的价值已经验证。

v0.4 历史验收：291 项测试中 282 项通过、9 项因历史外部材料缺失跳过；新安装包的四轮合成运行、独立审计和禁止网络的结果重放通过。本地 Nemotron 向量服务已完成 Agent Security 全部 3582 篇标题／摘要编码，三个领域的本机默认配置已接入，见 [本地向量验收](verification/local_embedding.json)。

v0.4 的真实 API 联调仍未完成：冻结的联调版本筛选 12 篇论文、保留 12 个技术阶段，审查全部 66 对候选关系后接受 7 条边；七篇开发诊断文献全部进入重排，其中五篇进入最终选择。第一轮候选、两模型模拟、批评和讨论已保存，价值评估连续三次收到不完整响应而中断。所有预检与失败运行合计 222 次请求，供应商报告 2,862,112 tokens；计入未知用量的保守预留后为 6,145,863 tokens，低于本次 450 次／800 万 tokens 上限。最新源码还修复了流式重试、正文证据别名和批次覆盖范围，但未用它覆盖既有冻结实验。当前不能宣称真实多轮反馈已验证有效，也没有专家评分校准或 idea 实证结果，详情见 [v0.4 验收报告](verification/paperfuture_v0.4.json)。

v0.3 最终验收：240 项测试中 231 项通过、9 项因历史外部材料缺失跳过；四轮离线运行与 263 个上游复制文件的完整性检查通过。Agent Security 的真实运行 `agent-security-v03-20261012-r04` 使用 GPT-4.1、Gemini 2.5 Flash、Qwen3.5 Plus，筛选 24 篇论文（16 篇有全文），构建 23 个技术阶段、54 条通过独立复核的边。76 对候选关系经过检查，另有 177 对可能的阶段关系没有进入候选池，不能据此声称全领域或所有关系已经穷尽。系统完成三轮后因连续缺乏实质进展而停止，输出 `needs_validation`；这次联调没有证明已找到高价值的新研究。独立安装的 wheel 在仓库外、禁止网络的条件下重放成功，最终 JSON 和文档一致，新增 API 调用为 0。

真实运行的文献、图、逐轮引用、全部前轮经验、评分和停止规则通过独立审计。已有方法反例经过当前完整批评和讨论流程后被判为零新颖性；仅更换综合判断、沿用旧讨论的反例检查出现过漏判。通过与失败结果、全部开发／失败尝试的 API 用量均保留在 [v0.3 验收报告](verification/paperfuture_v0.3.json)。最终这次运行新增 25 次调用，但复用了此前的准备结果，不能把 25 次当作从零运行成本。当前只有 Agent Security 做过完整真实 API 联调，另外两个领域完成了真实数据采集、审计和入口配置检查。v0.2 的历史记录见 [旧验收](verification/paperfuture_v0.2.json)。

`paperfuture bench --cases cases.json --output audit.json` 在生成完成后检查未来标签的首次公开时间，并汇总人工机制匹配评分；标签不进入生成。当前不是已发布并通过独立验证的 PaperFutureBench。输入时间正确也不能排除 API 模型自身的未来知识；正式论文仍需专家盲评、等预算基线、消融、重复实验和模型时间证据。

评分审查命令为 `paperfuture bench-graph --run RUN_DIRECTORY --annotations expert-heldout.json --output score-audit.json`。标注文件是 JSON 数组，每条含 `case_id`、`split: "heldout"`、`source_key`、`target_key`、`supported`、`weight`（不确定或关系不成立时为 null）、`annotator`、`rationale`；可附 `relation` 与五维 `delta`。同一论文对可有多名标注者，同一标注者不能重复计数。输入身份只是标注方声明；程序不声称已验证其专家资格。

核心实现位于 [dataset.py](src/paperfuture/dataset.py)、[literature.py](src/paperfuture/literature.py)、[graph.py](src/paperfuture/graph.py)、[exploration.py](src/paperfuture/exploration.py) 与 [pipeline.py](src/paperfuture/pipeline.py)。三个领域的入口配置见 [examples/paperfuture](examples/paperfuture)。

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
