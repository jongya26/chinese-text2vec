# Chinese Text2Vec API

English | [简体中文](./README_zh.md)

This project provides a straightforward FastAPI service for converting Chinese text into high-quality vector embeddings. It leverages the powerful `shibing624/text2vec-base-chinese` Sentence Transformer model, offering an easy-to-integrate solution for various NLP tasks.

## Key Features

*   **Chinese Text Embeddings**: Generate 768-dimensional vector representations for Chinese sentences.
*   **Simple RESTful API**: Easily integrate text embedding capabilities into any application via a standard HTTP POST request.
*   **Performance Optimized**: Utilizes the efficient Sentence Transformers library and is optimized for platforms like Apple Silicon.

## Quick Start (for API Consumers)

### 1. Configuration (Optional)

You can configure the default model using environment variables.

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Edit `.env` to suit your needs:
    *   `MODEL_NAME`: The default model key (e.g., `bge-small-zh-v1.5`).

### 2. Run the API Server

```bash
uv run python main.py
```

### 3. Discover Models

Get a list of all supported models and their specifications:

```bash
curl http://0.0.0.0:8015/v1/models
```

### 4. Send an Embedding Request (Standard API)

Use the standard `/v1/embeddings` endpoint. You can specify different models for different languages.

```bash
curl -X POST "http://0.0.0.0:8015/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{
       "input": ["今天天气很好", "Hello world"],
       "model": "multilingual-e5-small"
     }'
```

#### Expected Response

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

## Supported Models

| Model Key | Language | Dims | Description |
| :--- | :--- | :--- | :--- |
| `text2vec-base-chinese` | ZH | 768 | Default balanced Chinese model. |
| `bge-small-zh-v1.5` | ZH | 512 | High efficiency Chinese model. |
| `bge-large-zh-v1.5` | ZH | 1024 | SOTA large Chinese model. |
| `all-MiniLM-L6-v2` | EN | 384 | Popular fast English model. |
| `multilingual-e5-small` | Multilingual | 384 | Excellent EN/ZH/Multilingual support. |

## Embedding Details

*   **Output Format**: Each input sentence is transformed into a 768-dimensional float vector.
*   **Applications**: These embeddings can be used for tasks such as semantic search, document similarity, clustering, and more.
*   **Similarity**: Higher cosine similarity between two embedding vectors indicates that their corresponding sentences are more semantically related (values typically range from 0 to 1).

## Resources

*   **Underlying Model**: `shibing624/text2vec-base-chinese` on Hugging Face ([Model Card](https://huggingface.co/shibing624/text2vec-base-chinese))
*   **Sentence Transformers Library**: Official documentation ([sbert.net](https://www.sbert.net/))
