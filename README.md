# PaperFuture

PaperFuture v0.3 沿用六步主线：**文献收集与筛选 → 带权有向发展图 → LLM 选择跃迁方向和幅度 → world model 模拟与讨论 → 动态探索，全局最多四轮 → idea 与价值判断**。

支持 Agent Security、多模态大模型自进化中的可解释性、AI4EDA。模型推理、向量化均通过 API；本机只做数据存储、PDF 文本提取、排序、图校验和控制。无需本地模型或 GPU。运行完成表示流程完成，不能据此宣称 idea 已被实验验证。

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
# 在本机设置 PAPERFUTURE_API_KEY；不要把明文密钥写入配置或 Git。
.venv/bin/paperfuture doctor --config examples/paperfuture/agent_security.json
.venv/bin/paperfuture run --config examples/paperfuture/agent_security.json
```

示例入口会先构建／恢复文献库，再顺序执行筛选、构图、跃迁、模拟讨论和输出。API 兼容 `/chat/completions` 与 `/embeddings`；每个角色均可独立设置服务地址、模型和凭据引用。`doctor` 只检查本机配置。模型列表不等于实际可调用性，多模型接入应先做小型调用验证。程序检查配置中模型标识的多样性，不能独立证明网关背后的模型权重身份。

本机已经配置好的运行可直接使用：

```bash
PYTHONPATH=src python3.11 -m paperfuture run \
  --config "$PAPERFUTURE_DATA_ROOT/configs/agent_security_v03.json" \
  --workspace "$PAPERFUTURE_DATA_ROOT"
```

`prepare` 执行到发展图；同一 `run` 命令恢复中断；加 `--replay` 禁止模型和文献网络请求。`run_id` 固定问题、模型、参数与源码指纹，修改后需要新的 ID。重放历史记录须使用对应代码版本。所有重试受次数与总预算约束，网络失败也记录；没有账单价格表时只报告供应商返回的 tokens，不虚构货币成本。

向量按输入内容和模型标识缓存。重排、证据提取和建图还可复用同一个 `replicate_id` 下输入提示、结构、模型及服务地址完全相同、已有成功回执且再次通过当前校验的结果；`usage.json` 记录原始回执和复用来源，不将复用伪装成新请求。新的跃迁、模拟、批评、讨论和综合判断不跨运行复用。需要独立重复实验时使用不同的 `replicate_id`，评测时另行报告准备成本与探索成本。

示例配置按语料规模设置有限预算：Agent Security 最多 800 次调用／800 万 tokens，多模态最多 1,000 次／1,200 万 tokens，AI4EDA 最多 600 次／600 万 tokens。这些是上限，不是实际账单或完成保证；新模型、更多节点、格式重试和独立复现实验会改变用量。首次运行需要为全库向量化及每条候选边的判断、复核留出预算，不能用命中缓存后的少量调用冒充完整成本。

## 1. 文献收集与筛选

可先独立建库，不调用 LLM：

```bash
.venv/bin/paperfuture collect --domain all \
  --data-root "$PAPERFUTURE_DATA_ROOT" \
  --cutoff 2026-10-12T00:00:00Z --fulltext-limit 100
```

采集按领域的明确子方向查询与年份分层。Agent Security 与多模态领域从 2015 年开始，AI4EDA 从 2010 年开始；没有检索结果的年份明确列为缺口。2025—2026 是当前配置的最近两个日历年，2026 仅到截止日。大年份再按月份分页，失败的查询会尝试拆成更小且不重叠的日期窗口；完整记录请求、匹配数量、获取数量、失败与恢复过程和截断情况。查询覆盖不等于全领域覆盖，当前索引也不能被冒充为历史时刻的索引。

发现阶段只取得论文 ID；正文与摘要请求明确的 arXiv **v1**。不把最新修订内容贴到旧年份上。元数据保存标题、作者、类别、DOI／期刊信息（来源有则保存）、首次公开日期、版本日期和来源。全文层保留原始 PDF、提取文本、SHA-256 和来源响应，参考文献中的 arXiv 标识只做机械提取，仍可能不完整。公式、图表、扫描件和复杂排版不能保证被文本提取完整表达。

全文优先覆盖早期年份，并在最近两年之间分配名额、按子方向选择论文；已有全文不会因调整抽样而删除。正式运行在冻结快照前为相关候选补全文。未取得的全文记录为 `abstract_only`，不能假定已经看过全文实验。

数据目录如下：

```text
$PAPERFUTURE_DATA_ROOT/
  datasets/<domain>/<cutoff-date>/
    spec.json, discovery.json, manifest.json
    version_records.json, papers.jsonl, literature.sqlite
    snapshot-<content-hash>.json
  cache/arxiv/              原始 Atom、PDF、文本及来源回执
  cache/embeddings/         按输入内容与模型标识复用的向量
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

筛选算法为：**BM25 + 全库独立 API 向量召回 → RRF → 时间覆盖约束 → LLM 重排 → 原文证据提取 → 质量／相关性过滤 → 覆盖与冗余控制**。候选截断前保留历史年份，筛选阶段再次平衡历史轨迹与最新两年，默认近期目标占比 65%，并在两个近期年份之间分配名额，避免最新一年挤掉前一年。质量和相关性不达标的历史论文不会为凑年份强行入选；预算或有效候选不足时报告实际比例和缺失年份，不伪造覆盖。证据提取使用有边界的原文片段视图并保留精确偏移；不会把长全文无控制地重复塞入每个模型请求。模型侧使用 `E1` 等短证据别名，保存时通过封存的映射准确还原内容哈希 ID；未知引用直接拒绝，不用模糊匹配补造引用。

采集遵循 [arXiv 版本查询接口](https://info.arxiv.org/help/api/user-manual.html)，同一归档目录的请求统一限速、加锁并缓存。`sources` 支持 `arxiv_dataset` 自动完整建库、本地 `jsonl`，以及兼容旧配置的 `arxiv` 输入。`lexical` 是显式消融，向量调用失败不会静默退化成词法检索。

## 2. 带权有向发展图

每个节点描述问题、机制、假设、已展示能力、局限、瓶颈、论文版本与证据。先分批生成技术阶段，再全局协调重复阶段；默认保留独立阶段，只有与代表阶段的问题、具体机制、假设和能力边界逐项相同才允许合并。控制器保留原始技术描述和论文证据，不能把过滤、授权、重执行等方法统称为一个宽泛的“安全防御”节点。批次边界和年份也不能直接成为技术边界。候选前向关系来自技术相关性和已收集的真实引用，然后由 LLM 判断是否确有技术发展关系。允许分支、汇合和不连通部分。

每条边由 LLM 对五个维度分别评分：问题、假设、机制、能力边界、资源要求，各 **0—10** 分；另直接给出 **0—100** 的总体技术变化分及不确定区间。每个非零维度必须附自己的依据和证据。0 表示无变化，1—2 表示局部调整，3—4 为实质局部重设，5—6 为结构变化，7—8 为核心机制／假设／能力边界改变，9—10 为极强的范式变化判断。总体分不是五维机械求和，也不是成功概率或科研价值。

独立复核模型看原文和关系主张，不看建图模型的数值评分；它重新判断关系和幅度。只有 `supported` 关系进入历史图，`unsupported/uncertain` 的判定与原提案一并保存。引用、主题相似、评测基准提到某个方法，都不能单独证明继承或结合。程序另外检查来源、时间方向、DAG、论文覆盖和评分区间。没有任何边通过复核时停止并要求改善证据，不用伪造关系把图连起来。语义复核仍是模型判断，专家抽查与标注一致性评测仍有必要。

默认每次调用只判断或复核一条关系，避免多关系批次中的漏判与引用错配；可在验证模型可靠性后调整 `graph.edge_batch` 和 `graph.review_batch`。`graph.concurrent_calls` 默认 3，范围 1—8；独立关系可以并发处理，结果按输入顺序汇总，每个请求隔离上下文，并在同一事务账本中预留总预算。阶段 ID 与原文引用 ID 分别标识，逐维引用由控制器汇总，但每一个引用和时间见证都必须来自对应端点的真实原文。

## 3. 选择跃迁

输入包含完整的阶段／边概览、本轮检索到的原文、条件反馈和所有前轮经验。每篇图中论文至少保留一段已提取的原文证据；动态检索补充当前问题的证据，不覆盖历史基线的原文。提案必须说明现有机制、改变的原则、因果链、最近似工作、为什么不只是叠模块，以及能够证伪优势的最小对照实验。

组件组合只有在提出明确的交互机制，并且能通过对照／消融区分交互效应与各组件独立效应时，才可能构成有意义的跃迁。批评和综合判断必须对照全图中最近似的已有方法，而不能只比较提案自行挑选的旧基线；综合判断识别为 `already_known` 后，控制器将有效 novelty 强制为 0。若判为 `integration_only` 或 `unclear`，有效 novelty 最高为 2/4，不能直接给出强推荐。历史边的技术变化幅度不能被当成未来 idea 的新颖性。已有工作识别本身仍依赖模型：真实反例检查中出现过理由承认方法已发表、分类却仍给出机制变化的漏判，验收报告保留失败结果。结构约束不能保证全球新颖性，需要专门的反例集和专家盲评。

## 4. 多模型 world model

`world_model.simulators` 指定多个模型独立产生成功、部分成立和失败情景；批评模型独立检查机制、冲突、未知项和最小实验。初次批评看不到模拟输出。随后各模拟模型阅读相互独立的分析，讨论分歧、修改条件预测、提出区分性实验和下一跳启示；综合判断保留未解决分歧，不按多数投票决定科学事实。

真实证据冲突、缺少验证、审稿偏好分别处理。未验证和非主流不能单独否决 novelty。严重且适用的证据冲突不能因为综合判断漏抄引用而消失；驳回这种冲突必须给出证据。重复异议按最大严重度处理，批评权重默认 0.2、最多 0.35。模型身份、独立响应、讨论及最终处理全部保留。模拟始终是条件预测，不是实际实验。

## 5—6. 累积经验、最多四轮和最终结果

每轮保存带来源 ID 的成功条件、失败预测、争议、下一瓶颈和依赖假设。下一轮在**同一个冻结文献快照**中，用当前问题、导航位置和相关经验重新检索原文，不能偷偷读未来标签。提案的 `learning_trace` 必须引用前轮经验，并说明这些经验怎样改变当前方向／机制。利用某条经验时，其依赖假设也会由控制器带入；假设不能升级成历史事实。

API 输入使用统一的假设表，经验通过假设 ID 引用它；持久化记录仍保留完整依赖。这是可以无损还原的去重，避免同一组假设随每条讨论经验反复复制、在第三或第四轮撑满上下文，不丢弃较早轮次的经验。

控制器检查实质机制变化或已解决的问题，避免仅靠模型生成的短标签判断停滞。发现重复候选后引导换位置，连续无进展则停止。满足预先固定的机制、价值与可行性条件可提前停止，全局最多四个候选轮次，断点恢复不产生额外轮次。最后选择已经评估过的候选，不另生成第五个 idea；可以输出 `needs_validation` 或 `not_recommended`。

## 运行产物与验证

| 产物 | 含义 |
|---|---|
| `snapshot.json`, `selected.json` | 冻结文献、筛选、年份分布、缺口 |
| `trajectory.json`, `trajectory_review.json` | 阶段、通过复核的边、原提案、拒绝原因、五维评分与区间 |
| `graph_context.json` | 全图概览 |
| `rounds/01..04/evidence_context.json` | 本轮检索问题、文献与精确证据 |
| `rounds/01..04/experience_memory.json` | 所有前轮的条件经验与依赖 |
| `proposal/simulation/critique/discussion/assessment/decision.json` | 各轮对应步骤，位于轮次目录 |
| `IDEA.md`, `final_idea.json` | 最终候选、价值、条件、可行性、分歧、证据和停止原因 |
| `api/usage.sqlite`, `usage.json` | 每次网络尝试、未知计费、tokens 与向量复用来源 |

```bash
PYTHONPATH=src:tests python3.11 -m unittest discover -s tests -q
python3.11 tools/check_snapshot.py
python3.11 tools/run_paperfuture_demo.py --workspace "$PAPERFUTURE_DATA_ROOT/offline-demo-v03" --four-rounds
python3.11 tools/audit_paperfuture.py --data-root "$PAPERFUTURE_DATA_ROOT" \
  --run "$PAPERFUTURE_DATA_ROOT/runs/YOUR_COMPLETED_RUN_ID" \
  --output "$PAPERFUTURE_DATA_ROOT/verification/audit.json"
```

`audit_paperfuture.py` 不调用网络；它重新核验导出的论文唯一性、版本日期、PDF／文本哈希、数据盘路径，以及已完成运行中的图约束、每轮证据、累积经验、评分和停止决策。它证明记录与软件契约一致，不代替专家判断论文内容或 idea 质量。

离线演示使用明确标记的合成论文与 HTTP 响应，只检验软件行为。正式入口不会退回合成数据。新增回归检查覆盖证据冲突遗漏、模块堆叠、错误关系剔除、独立评分、多模型调用、全部前轮经验、动态检索、时间平衡、全文范围与原始 PDF 完整性。

v0.3 最终验收：240 项测试中 231 项通过、9 项因历史外部材料缺失跳过；四轮离线运行与 263 个上游复制文件的完整性检查通过。Agent Security 的真实运行 `agent-security-v03-20261012-r04` 使用 GPT-4.1、Gemini 2.5 Flash、Qwen3.5 Plus，筛选 24 篇论文（16 篇有全文），构建 23 个技术阶段、54 条通过独立复核的边。76 对候选关系经过检查，另有 177 对可能的阶段关系没有进入候选池，不能据此声称全领域或所有关系已经穷尽。系统完成三轮后因连续缺乏实质进展而停止，输出 `needs_validation`；这次联调没有证明已找到高价值的新研究。独立安装的 wheel 在仓库外、禁止网络的条件下重放成功，最终 JSON 和文档一致，新增 API 调用为 0。

真实运行的文献、图、逐轮引用、全部前轮经验、评分和停止规则通过独立审计。已有方法反例经过当前完整批评和讨论流程后被判为零新颖性；仅更换综合判断、沿用旧讨论的反例检查出现过漏判。通过与失败结果、全部开发／失败尝试的 API 用量均保留在 [v0.3 验收报告](verification/paperfuture_v0.3.json)。最终这次运行新增 25 次调用，但复用了此前的准备结果，不能把 25 次当作从零运行成本。当前只有 Agent Security 做过完整真实 API 联调，另外两个领域完成了真实数据采集、审计和入口配置检查。v0.2 的历史记录见 [旧验收](verification/paperfuture_v0.2.json)。

`paperfuture bench --cases cases.json --output audit.json` 在生成完成后检查未来标签的首次公开时间，并汇总人工机制匹配评分；标签不进入生成。当前不是已发布并通过独立验证的 PaperFutureBench。输入时间正确也不能排除 API 模型自身的未来知识；正式论文仍需专家盲评、等预算基线、消融、重复实验和模型时间证据。

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
