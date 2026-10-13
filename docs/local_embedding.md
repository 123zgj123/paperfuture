# 本地论文向量化

PaperFuture 可以把向量化改为本机推理，同时保留原有的文献筛选、发展图、跃迁和多模型模拟流程。当前部署选择 NVIDIA 的 `Nemotron-3-Embed-1B-BF16`，固定 Hugging Face 修订为 `c0c9fea93ea424587517f2c59e20db9f1d6bf615`。

这个模型于 2026 年 7 月发布，面向文本检索，输出 2048 维向量，支持中英文等语言。选择依据是检索用途、较新的公开模型和本机资源，不代表它在所有检索任务上排名第一。官方说明见 [模型卡](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)；权重与配置遵循模型仓库的 OpenMDW-1.1 许可，原始许可文件随下载保留。PaperFuture 的软件许可不替代模型许可。

## 数据盘位置

根目录为 `/mnt/efs/zhangguijia/paperfuture`：

| 内容 | 根目录下的位置 |
|---|---|
| 权重、分词器、模型卡、原始许可及下载哈希 | `models/nvidia/Nemotron-3-Embed-1B-BF16/c0c9fea93ea424587517f2c59e20db9f1d6bf615/` |
| 隔离的 CPU 推理环境与固定依赖清单 | `software/local-embedding-cpu/venv/`、`requirements.lock` |
| 实际推理参数和服务标识 | `software/local-embedding-cpu/profile.json` |
| 安装日志和服务日志 | `software/local-embedding-cpu/` |
| 向量缓存及来源 | `cache/embeddings/` |
| 各领域的默认配置及同内容的显式本地配置 | `configs/*_v04.json`、`configs/*_v04_local_embedding.json` |
| 切换前的远程向量配置备份 | `configs/*_v04_api_embedding.json` |

下载使用 `tools/download_embedding_model.py`，固定不可变修订，校验 LFS SHA256 或 Git blob 哈希，临时文件与最终文件都在数据盘。启动时再次核验全部文件。加载只读取本地文件，禁用远程 Python 代码和 Hugging Face 网络访问。

## 接入方式

`tools/local_embedding_server.py` 提供本机 `/v1/embeddings` 接口。当前 CPU 配置使用 BF16、SDPA、4 个线程、每个推理批次 1 条文本、2048 维归一化向量。服务模型 ID 包含检查点修订及推理配置的哈希，包括运行库版本、精度与长度限制；更换这些条件会产生不同缓存身份。

模型原生长度上限是 32768 tokens。本机为控制内存与响应时间，将运行配置限制为 4096 tokens；超长输入返回错误，不会静默截断。PaperFuture 当前向量召回编码论文标题与摘要，全文证据仍由后续检索与提取使用。若需要整篇论文向量化，应另行明确分块策略，不能把截断后的摘要向量当成全文向量。

服务只监听 `127.0.0.1:8769`，通过本地凭据文件鉴权。不要把远程 LLM API 密钥用于此服务。使用部署时生成的配置，其中 embedding 部分结构如下：

```json
{
  "base_url": "http://127.0.0.1:8769/v1",
  "model": "填写 profile.json 中的 model_id",
  "api_key_file": "/mnt/efs/zhangguijia/paperfuture/configs/local_embedding.key",
  "input_prefixes": {"query": "query: ", "document": "passage: "},
  "cache_scope": "text"
}
```

论文与查询必须使用相同模型和相应前缀。前缀是模型训练约定的一部分，不是给 LLM 的研究提示词。模型不会在服务端猜测单条输入是否为查询。输入的精确字符串参与向量缓存哈希；旧 Qwen API 向量不会复用为 Nemotron 向量。

新配置必须使用新的 `run_id`，不能修改正在进行或已完成运行的冻结配置。本地请求不产生外部 embedding API 费用，仍会记录请求、tokens 和缓存来源。LLM 筛选、构图与探索仍按其 API 配置产生费用。

## 安装和运行

部署使用 Python 3.11 的独立虚拟环境，CPU 版 PyTorch 2.14.1、Transformers 5.19.0、Sentence Transformers 6.1.0；所有传递依赖记录在 `requirements.lock`。安装时设置 `PIP_CACHE_DIR`、`TMPDIR`、`HF_HOME` 和 `XDG_CACHE_HOME` 到数据盘子目录。PyTorch 从官方 CPU 软件源安装，避免下载 CUDA 依赖。

服务启动命令为：

```bash
/mnt/efs/zhangguijia/paperfuture/software/local-embedding-cpu/venv/bin/python \
  tools/local_embedding_server.py \
  --model-directory /mnt/efs/zhangguijia/paperfuture/models/nvidia/Nemotron-3-Embed-1B-BF16/c0c9fea93ea424587517f2c59e20db9f1d6bf615 \
  --key-file /mnt/efs/zhangguijia/paperfuture/configs/local_embedding.key \
  --profile-file /mnt/efs/zhangguijia/paperfuture/software/local-embedding-cpu/profile.json
```

该命令用于手动前台启动。本机已将同一份服务代码固定在数据盘，并启用用户级 `paperfuture-embedding.service`；它随用户服务管理器启动，异常退出自动重启，退出 SSH 不影响运行。检查使用 `systemctl --user status paperfuture-embedding.service`，不要同时手动启动第二个实例。`GET /health` 和 `GET /v1/models` 都需要本地鉴权；前者提供实际推理配置。三个领域的 `*_v04.json` 默认配置已切换到本地服务，也可使用同内容的 `*_v04_local_embedding.json`。正在进行的旧实验继续使用其冻结配置。

只预计算论文向量、不调用远程研究模型时，可以运行：

```bash
python3.11 tools/warm_local_embeddings.py \
  --config /mnt/efs/zhangguijia/paperfuture/configs/agent_security_v04_local_embedding.json \
  --data-root /mnt/efs/zhangguijia/paperfuture
```

此工具只允许 `127.0.0.1` 的向量服务，记录冻结的标题／摘要输入、逐次回执、进度和最终用量。v0.4.1 默认本地配置使用 `cache_scope=text`：按精确标题／摘要、用途前缀、模型及服务地址逐条复用。新增、排序变化、补全文不会触发未变化的标题／摘要重新编码。更换模型或更改标题／摘要才生成对应的新向量。远程模型默认仍使用批次缓存，只有确认输出与批次同伴无关后才能开启逐条复用。历史回执的一次性迁移使用 `tools/repack_embedding_cache.py`，保留来源哈希且不调用模型。进度位于 `runs/local-embedding-index-*/progress.json`，完成后生成 `completed.json`。

检索小样本仅验证加载、输入约定、向量质量的基本合理性和接口连通性；它不证明全领域召回完整，也不证明优于原来的 API 向量模型。正式评测需要独立标注的查询、相关论文和历史轨迹覆盖标准。

回溯式未来预测评测同样需要登记向量模型的发布时间与训练时间证据；2026 年发布的向量模型不能被直接当成较早年份已经可用的组件。

2026-10-13 本机验收：24 篇论文与 3 条查询的小样本测试通过；随后完成 Agent Security 的 3582 篇标题／摘要向量预计算，首次计算约 22.2 分钟。禁止网络的全库缓存重放约 22.6 秒，新增请求为 0。推理进程实测常驻内存约 3.4 GiB。其他两个领域已生成本地向量配置，小样本接入已验证，全库预计算尚未执行。详细记录见 [本地向量验收](../verification/local_embedding.json)。
