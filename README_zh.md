# 中文文本向量化 API (Chinese Text2Vec API)

[English](./README.md) | 简体中文

本项目提供了一个基于 FastAPI 的简单服务，用于将中文文本转换为高质量的向量嵌入（Embeddings）。它采用了强大的 `shibing624/text2vec-base-chinese` Sentence Transformer 模型，为各种 NLP 任务提供了一个易于集成的解决方案。

## 核心功能

*   **中文文本嵌入**：为中文句子生成 768 维的向量表示。
*   **简单的 RESTful API**：通过标准的 HTTP POST 请求，轻松将文本嵌入功能集成到任何应用中。
*   **性能优化**：利用高效的 Sentence Transformers 库，并针对 Apple Silicon 等平台进行了优化。

## 快速上手（API 使用者）

### 1. 配置（可选）

您可以使用环境变量来配置默认模型。

1.  复制示例环境变量文件：
    ```bash
    cp .env.example .env
    ```
2.  根据需要编辑 `.env`：
    *   `MODEL_NAME`: 默认模型名称（例如：`bge-small-zh-v1.5`）。

### 2. 运行 API 服务器

```bash
uv run python main.py
```

### 3. 查看可用模型

获取所有支持的模型及其详细信息：

```bash
curl http://0.0.0.0:8015/v1/models
```

### 4. 发送嵌入请求 (标准 API)

使用标准的 `/v1/embeddings` 接口。您可以为不同的语言选择不同的模型。

```bash
curl -X POST "http://0.0.0.0:8015/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{
       "input": ["今天天气很好", "Hello world"],
       "model": "multilingual-e5-small"
     }'
```

#### 预期响应

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "embedding": [...], "index": 0 },
    { "object": "embedding", "embedding": [...], "index": 1 }
  ],
  "model": "multilingual-e5-small",
  "usage": { "prompt_tokens": 17, "total_tokens": 17 }
}
```

## 支持的模型

| 模型名称 | 推荐语言 | 维度 | 描述 |
| :--- | :--- | :--- | :--- |
| `text2vec-base-chinese` | 中文 | 768 | 默认的平衡中文模型。 |
| `bge-small-zh-v1.5` | 中文 | 512 | BAAI 出品的高效率中文模型。 |
| `bge-large-zh-v1.5` | 中文 | 1024 | SOTA 级别的中文大模型。 |
| `all-MiniLM-L6-v2` | 英文 | 384 | 流行且快速的英文模型。 |
| `multilingual-e5-small` | 多语言 | 384 | 优秀的英/中/多语言支持。 |

## 嵌入详情

*   **输出格式**：每个输入句子被转换为 768 维的浮点向量。
*   **应用场景**：这些嵌入可用于语义搜索、文档相似度对比、聚类等任务。
*   **相似度**：两个嵌入向量之间的余弦相似度越高，表示对应的句子在语义上越相关（值通常在 0 到 1 之间）。

## 相关资源

*   **底层模型**：Hugging Face 上的 `shibing624/text2vec-base-chinese` ([模型卡片](https://huggingface.co/shibing624/text2vec-base-chinese))
*   **Sentence Transformers 库**：官方文档 ([sbert.net](https://www.sbert.net/))
