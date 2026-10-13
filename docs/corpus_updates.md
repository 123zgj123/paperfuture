# 固定语料与每周更新

PaperFuture 从 v0.4.1 开始将文献维护与研究生成分开，v0.5 增加 Alignment 和联合领域入口。正常 `run` 读取配置中指定的不可变语料版本，不访问 arXiv、不下载 PDF、不补录引用论文。筛选和后续各轮的检索都在同一个快照内进行。每周维护产生新版本，下一次新实验才使用它。

这不改变论文的六步研究流程，只明确第一步的数据边界和维护时间。

## 当前数据

数据根目录为 `/mnt/efs/zhangguijia/paperfuture`。2026-10-13 发布／核验的固定版本如下；前三个库截止 2026-10-12 00:00 UTC，Alignment 截止 2026-10-13 00:00 UTC：

| 领域 | 论文版本 | 已归档全文 | 2025—2026 论文 | 有论文的年份 |
|---|---:|---:|---:|---|
| Agent Security | 3,587 | 176 | 2,622（73.1%） | 2016—2026 |
| 多模态自进化可解释性 | 9,115 | 100 | 6,584（72.2%） | 2015—2026 |
| AI4EDA | 810 | 100 | 442（54.6%） | 2012—2026，2015 无匹配 |
| LLM Alignment | 12,596 | 217 | 8,630（68.5%） | 2015—2026 |

这些数量是明确查询得到的文献库规模，可以支持方法原型；不是领域完整性证明。Agent Security 的 2015 年、AI4EDA 的 2010、2011、2015 年仍无匹配。全文占比有限，摘要不能替代实验细节；arXiv 之外的会议、产业资料，以及没有命中查询词的关键转折仍可能遗漏。需要独立专家标注的子方向、关键论文和引用链评测覆盖率。

Agent Security 版本来自本次真实实验已经冻结的 3,587 篇快照，包含在原 3,582 篇基础上补取的 5 篇引用论文和额外全文；旧采集导出、旧实验和旧版本均保留。

Alignment 的初始分支查询发现 12,559 篇，经 Constitutional AI／RLAIF 历史补查增加 37 篇。八篇明确列出的发展脉络文献均有全文。四库跨领域去重为 25,059 个版本，其中 568 篇有全文；本次共核验 1,136 个 PDF／文本文件。Agent Security＋Alignment 在联合截止日下去重为 15,766 篇。

## 检索和向量缓存

当前检索依次执行问题 BM25、独立全库向量召回、技术分支多查询、库内引用回溯；通过 RRF 合并，分配历史与近期候选，进行 LLM 相关性重排、原文证据提取、覆盖与冗余筛选。动态探索改变查询和候选，不改变实验语料版本。

本地向量模型编码“标题 + 摘要”，正文供证据提取使用。缓存身份由服务地址、固定模型 ID 和带 `query:`／`passage:` 前缀的精确输入文本共同决定。

- 新增论文：只计算其缓存缺失的标题／摘要。
- 重排、插入或删除论文：其他论文的已有向量不变。
- 补充 PDF、全文和引用：标题／摘要不变时复用原向量。
- 标题／摘要、前缀、模型或推理配置变化：生成对应的新向量，旧缓存保留。
- 新的研究查询可能需要一个新的查询向量，不等于重新计算全库。

本地配置使用 `cache_scope: "text"` 和 `cache/embedding_rows.sqlite`。同一身份出现不同向量时直接报错，不任意覆盖。远程模型默认仍使用原批次缓存，只有确认输出与批次同伴无关后才能开启逐条复用。历史向量已通过原始成功回执迁移，保留回执与向量哈希，不调用模型重新生成。

本次前三个库已完成全文献标题／摘要索引。新增 Alignment 索引由用户级 `paperfuture-alignment-index.service` 持续预计算，首次准备未完成时不能宣称全库向量已就绪。进度见 `runs/local-corpus-index-llm_alignment-*/progress.json` 和 `reviews/alignment-local-vector-precompute.log`；完成后写入 `reviews/llm_alignment-full-vector-index-20261013.json`，并用禁止网络的联合检索检查写入 `reviews/v05-joint-hybrid-cache-replay.json`。已有向量不会因这个任务重新编码。

## 每周维护规则

本机计划每周一 **02:00 UTC（北京时间 10:00）**执行。使用用户级 systemd 持久化定时器；用户已启用 linger，退出 SSH 不影响定时任务。机器关机或数据盘不可用时不能执行；`Persistent=true` 会在定时器恢复后补一次错过的执行。

维护参数在数据盘 `configs/weekly_corpora.json`：

1. 分别读取四个领域默认配置固定的版本。
2. 检索上次截止日到本次截止日的新增窗口，向前重叠 7 天处理索引延迟；已存在的论文版本不重复请求元数据。
3. 每领域最多补录 20 个缺失的历史种子或引用 ID；这些 ID 来自未来运行的 `literature_update_requests.json` 和显式历史清单。补录仍要求 v1 日期不晚于本次截止日。
4. 每领域最多补 24 篇全文。元数据窗口不完整时不发布新版本；PDF 获取失败保留摘要并写明失败。
5. 校验新版本并预热本地向量缓存。只有缺失的文本会调用本地模型；不调用远程聊天模型，不自动生成 idea。
6. 向量预热成功后，将默认配置切换到新版本和新 `run_id`，并将旧配置归档到 `configs/history/`。采集或向量化失败时原默认配置继续可用。

每个领域独立成功或失败，维护日志保留部分失败状态。`latest.json` 表示最新发布的语料；`active.json` 表示已完成向量预热并激活的语料，二者可能暂时不同。旧实验只读取自己的 `snapshot.json`；恢复它应使用原配置或归档配置，不使用更新后的默认配置。

`combined_profiles` 可指定联合研究的默认配置路径。联合配置声明的每个领域都须出现在单领域 `profiles` 中；只有各库都更新成功、对应向量与联合问题的查询向量准备好后，才原子替换联合配置中的整组 manifest。任一依赖失败则保留原来的版本组合，旧配置同样归档。联合截止日取各语料截止日的较早值。

7 天重叠不能保证发现更长索引延迟或后来改正的旧论文分类。它也不会自动扩展研究范围；没有规范论文 ID 的覆盖缺口仍需要人工检查查询和历史文献清单。

## 命令和检查

首次建库与发布：

```bash
export PAPERFUTURE_DATA_ROOT=/mnt/efs/zhangguijia/paperfuture
.venv/bin/paperfuture collect --domain agent_security \
  --cutoff 2026-10-12T00:00:00Z --fulltext-limit 100
.venv/bin/paperfuture corpus-publish --domain agent_security \
  --cutoff 2026-10-12T00:00:00Z \
  --input "$PAPERFUTURE_DATA_ROOT/datasets/agent_security/20261012/papers.jsonl"
```

将 `manifest_path` 填入配置的 `sources: [{"type":"corpus","manifest":"..."}]`，并设置 `literature_mode: "frozen"`。本机 v0.5 的入口使用 `configs/*_v05.json`，联合入口为 `configs/agent_security_alignment_v05.json`；旧配置与安装包保留供历史运行恢复。示例配置见 [联合语料与 skill](research_skills_and_memory.md)。

检查周更配置（不联网、不改数据）：

```bash
.venv/bin/paperfuture corpus-maintain \
  --config "$PAPERFUTURE_DATA_ROOT/configs/weekly_corpora.json" --check-only
systemctl --user list-timers paperfuture-corpus-weekly.timer
systemctl --user status paperfuture-corpus-weekly.service
```

手动触发同一维护任务：

```bash
systemctl --user start paperfuture-corpus-weekly.service
```

手动发布一次增量版本而不切换默认配置：

```bash
.venv/bin/paperfuture corpus-update --manifest /absolute/release/manifest.json \
  --cutoff 2026-10-19T02:00:00Z --overlap-days 7 --fulltext-limit 24
```

检查结果：`maintenance/latest.json`、`maintenance/reports/`、`maintenance/weekly.log`、`corpora/<domain>/updates/*/audit.json`。向量预热记录位于 `runs/local-corpus-index-*/`，与研究实验明确区分。

发现历史查询遗漏时，不必移动原来的截止日：

```bash
.venv/bin/paperfuture corpus-supplement --manifest /absolute/release/manifest.json \
  --paper-id arxiv:2212.08073 \
  --query '(all:"constitutional AI" OR all:RLAIF)' --fulltext-limit 24
```

这会独立补查声明年份范围、校验 v1 日期、优先补充指定论文全文，然后发布新的不可变版本。查询未完成时不会发布；旧版本不变。该命令不是生成过程里的工具，明确查询的覆盖也不等于完整领域召回。

每周任务安装包固定在数据盘 `software/v0.5.2/installed/`。升级源码不会悄悄改变定时任务或已冻结实验的实现；后续升级应安装新版本并显式切换服务配置。
