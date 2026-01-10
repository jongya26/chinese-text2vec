# 中文文本向量化 API (Chinese Text2Vec API)

[English](./README.md) | 简体中文

本项目提供了一个基于 FastAPI 的简单服务，用于将中文文本转换为高质量的向量嵌入（Embeddings）。它采用了强大的 `shibing624/text2vec-base-chinese` Sentence Transformer 模型，为各种 NLP 任务提供了一个易于集成的解决方案。

## 核心功能

*   **中文文本嵌入**：为中文句子生成 768 维的向量表示。
*   **简单的 RESTful API**：通过标准的 HTTP POST 请求，轻松将文本嵌入功能集成到任何应用中。
*   **性能优化**：利用高效的 Sentence Transformers 库，并针对 Apple Silicon 等平台进行了优化。

## 快速上手（API 使用者）

### 1. 配置（可选）

您可以使用环境变量来配置服务器的主机、端口和模型。

1.  复制示例环境变量文件：
    ```bash
    cp .env.example .env
    ```
2.  根据需要编辑 `.env`：
    *   `HOST`: 绑定接口（默认：`0.0.0.0`）。
    *   `PORT`: 运行端口（默认：`8015`）。
    *   `MODEL_NAME`: 加载的 Sentence Transformer 模型（默认：`shibing624/text2vec-base-chinese`）。

### 2. 运行 API 服务器

```bash
uv run python main.py
```

API 将在 `http://0.0.0.0:8015`（或您配置的主机/端口）启动。

### 3. 发送嵌入请求

使用 `curl` 等工具向 `/embed` 接口发送 POST 请求。

```bash
curl -X POST "http://0.0.0.0:8015/embed" \
     -H "Content-Type: application/json" \
     -d '{ "sentences": ["今天天气很好", "上海的天气怎么样？"] }'
```

#### 预期响应

API 将返回包含嵌入向量的 JSON 对象：

```json
{
    "embeddings": [
        [0.123, 0.456, ..., 0.789],
        [0.789, 0.321, ..., 0.654]
    ]
}
```

每个内部列表代表对应输入句子的 768 维嵌入向量。

## 嵌入详情

*   **输出格式**：每个输入句子被转换为 768 维的浮点向量。
*   **应用场景**：这些嵌入可用于语义搜索、文档相似度对比、聚类等任务。
*   **相似度**：两个嵌入向量之间的余弦相似度越高，表示对应的句子在语义上越相关（值通常在 0 到 1 之间）。

## 相关资源

*   **底层模型**：Hugging Face 上的 `shibing624/text2vec-base-chinese` ([模型卡片](https://huggingface.co/shibing624/text2vec-base-chinese))
*   **Sentence Transformers 库**：官方文档 ([sbert.net](https://www.sbert.net/))
