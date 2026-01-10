# Chinese Text2Vec API

English | [简体中文](./README_zh.md)

This project provides a straightforward FastAPI service for converting Chinese text into high-quality vector embeddings. It leverages the powerful `shibing624/text2vec-base-chinese` Sentence Transformer model, offering an easy-to-integrate solution for various NLP tasks.

## Key Features

*   **Chinese Text Embeddings**: Generate 768-dimensional vector representations for Chinese sentences.
*   **Simple RESTful API**: Easily integrate text embedding capabilities into any application via a standard HTTP POST request.
*   **Performance Optimized**: Utilizes the efficient Sentence Transformers library and is optimized for platforms like Apple Silicon.

## Quick Start (for API Consumers)

### 1. Configuration (Optional)

You can configure the server host, port, and model using environment variables.

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Edit `.env` to suit your needs:
    *   `HOST`: The interface to bind to (default: `0.0.0.0`).
    *   `PORT`: The port to run on (default: `8015`).
    *   `MODEL_NAME`: The Sentence Transformer model to load (default: `shibing624/text2vec-base-chinese`).

### 2. Run the API Server

```bash
uv run python main.py
```

The API will start on `http://0.0.0.0:8015` (or your configured host/port).

### 3. Send an Embedding Request

Use a tool like `curl` to send a POST request to the `/embed` endpoint.

```bash
curl -X POST "http://0.0.0.0:8015/embed" \
     -H "Content-Type: application/json" \
     -d '{ "sentences": ["今天天气很好", "上海的天气怎么样？"] }'
```

#### Expected Response

The API will return a JSON object containing the embeddings:

```json
{
    "embeddings": [
        [0.123, 0.456, ..., 0.789],
        [0.789, 0.321, ..., 0.654]
    ]
}
```

Each inner list represents the 768-dimensional embedding vector for the corresponding input sentence.

## Embedding Details

*   **Output Format**: Each input sentence is transformed into a 768-dimensional float vector.
*   **Applications**: These embeddings can be used for tasks such as semantic search, document similarity, clustering, and more.
*   **Similarity**: Higher cosine similarity between two embedding vectors indicates that their corresponding sentences are more semantically related (values typically range from 0 to 1).

## Resources

*   **Underlying Model**: `shibing624/text2vec-base-chinese` on Hugging Face ([Model Card](https://huggingface.co/shibing624/text2vec-base-chinese))
*   **Sentence Transformers Library**: Official documentation ([sbert.net](https://www.sbert.net/))
