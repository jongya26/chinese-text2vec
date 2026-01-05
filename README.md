# Chinese Text2Vec with Sentence Transformers

A quick setup guide for using `shibing624/text2vec-base-chinese` model with sentence-transformers on macOS (M1/M2/M3).

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Terminal access

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### 2. Create Project

```bash
mkdir chinese-text2vec
cd chinese-text2vec
uv init
```

### 3. Install Dependencies

```bash
uv add sentence-transformers
```

## Quick Start

### Basic Usage

Create `example.py`:

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('shibing624/text2vec-base-chinese')

# Encode sentences
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡', '今天天气很好']
embeddings = model.encode(sentences)

# Calculate similarity
similarity = model.similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity.item():.4f}")
```

Run:

```bash
uv run python example.py
```

### One-liner Test

```bash
uv run --with sentence-transformers python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('shibing624/text2vec-base-chinese')
embeddings = model.encode(['你好世界', '再见世界'])
print(f'Shape: {embeddings.shape}')
"
```

## Understanding Results

- **Similarity score**: Range from 0 to 1
  - `> 0.8`: Very similar meaning
  - `0.5 - 0.8`: Somewhat related
  - `< 0.5`: Different topics

- **Embedding shape**: `(num_sentences, 768)`
  - Each sentence becomes a 768-dimensional vector

## Common Use Cases

- Semantic search
- Document similarity comparison
- Question-answer matching
- Text clustering
- Duplicate detection

## API Usage

### Running the API

To start the API server, run:

```bash
uv run python api.py
```

The API will be available at `http://0.0.0.0:8000`.

### Embedding Sentences

Send a POST request to the `/embed` endpoint with a JSON body containing a list of sentences.

**Endpoint:** `POST /embed`
**Content-Type:** `application/json`

**Example Request:**

```json
{
    "sentences": ["今天天气很好", "上海的天气怎么样？"]
}
```

**Example Response:**

```json
{
    "embeddings": [
        [0.123, 0.456, ...],
        [0.789, 0.321, ...]
    ]
}
```

## Notes

- First run downloads ~400MB model
- Model cached in `~/.cache/huggingface/`
- Optimized for Apple Silicon

## Resources

- [Model Card](https://huggingface.co/shibing624/text2vec-base-chinese)
- [Sentence Transformers Docs](https://www.sbert.net/)

