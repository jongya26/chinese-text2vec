# Chinese Text2Vec API

A FastAPI application that provides a simple API for generating high-quality Chinese text embeddings using the `shibing624/text2vec-base-chinese` Sentence Transformer model. This API is optimized for performance and ease of use.

## Features

*   **Fast Chinese Text Embeddings**: Utilizes the `shibing624/text2vec-base-chinese` model for efficient text vectorization.
*   **Simple RESTful API**: Easily integrate text embedding capabilities into your applications.
*   **Optimized for Apple Silicon**: Leverages hardware acceleration for faster processing on compatible devices.

## API Usage

### Running the API Server

To start the API server, ensure you have `uv` installed (https://astral.sh/uv/install.sh), then run:

```bash
uv run python main.py
```

The API will be available at `http://0.0.0.0:8015`.

### Endpoint: `/embed`

This endpoint generates embeddings for a list of Chinese sentences.

*   **Method**: `POST`
*   **URL**: `/embed`
*   **Content-Type**: `application/json`

#### Request Body

A JSON object with a single key `sentences` which is a list of strings (Chinese sentences).

```json
{
    "sentences": ["今天天气很好", "上海的天气怎么样？", "如何更换花呗绑定银行卡"]
}
```

#### Response Body

A JSON object containing the `embeddings` key, which is a list of lists of floats. Each inner list represents the 768-dimensional embedding vector for the corresponding input sentence.

```json
{
    "embeddings": [
        [0.123, 0.456, ..., 0.789],
        [0.789, 0.321, ..., 0.654],
        [0.456, 0.789, ..., 0.123]
    ]
}
```

## Embedding Details

*   **Embedding Dimension**: Each sentence is transformed into a 768-dimensional vector.
*   **Similarity Scores**: While not directly exposed by the API, embeddings can be used to calculate semantic similarity (e.g., cosine similarity). A higher score (closer to 1) indicates greater similarity.

## Resources

*   **Model Card**: [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)
*   **Sentence Transformers Documentation**: [sbert.net](https://www.sbert.net/)