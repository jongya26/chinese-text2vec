# Chinese Text2Vec API Context

## Project Overview

**Chinese Text2Vec** is a FastAPI-based service designed to generate high-quality vector embeddings for Chinese text. It utilizes the `shibing624/text2vec-base-chinese` model via the `sentence-transformers` library. This project provides a simple, RESTful interface for integrating Chinese text embedding capabilities into other applications for tasks like semantic search, clustering, and similarity analysis.

## Key Technologies

*   **Language:** Python 3.11+
*   **Web Framework:** FastAPI
*   **Server:** Uvicorn
*   **ML Library:** Sentence Transformers
*   **Model:** `shibing624/text2vec-base-chinese` (Hugging Face)
*   **Package Manager:** uv
*   **Testing:** pytest

## Architecture

*   **Entry Point (`main.py`):** Initializes and runs the Uvicorn server, hosting the FastAPI app defined in `api.py`.
*   **Application Logic (`api.py`):**
    *   Defines the FastAPI application `app`.
    *   Loads the `shibing624/text2vec-base-chinese` model upon startup.
    *   Exposes a POST endpoint `/embed` that accepts a list of sentences and returns their 768-dimensional vector embeddings.
*   **Configuration (`pyproject.toml`, `.env`):**
    *   `pyproject.toml`: Manages project metadata and dependencies.
    *   `.env`: (Optional) Environment variables for server settings (`HOST`, `PORT`) and model selection (`MODEL_NAME`). See `.env.example`.
*   **Documentation:**
    *   `README.md`: English documentation and quick start guide.
    *   `README_zh.md`: Chinese documentation.
    *   `GEMINI.md`: Detailed project context and architecture for LLM agents.

## Building and Running

### Prerequisites
*   Python 3.11 or higher
*   `uv` (Universal Python Package Installer/Manager)

### Installation
Install dependencies using `uv`:
```bash
uv sync
```

### Configuration
Copy `.env.example` to `.env` to customize settings:
```bash
cp .env.example .env
```

### Running the API
Start the development server:
```bash
uv run python main.py
```
The server will start on `http://0.0.0.0:8015` (default).

### API Usage
**Endpoint:** `POST /embed`

**Request Body:**
```json
{
  "sentences": ["今天天气很好", "上海的天气怎么样？"]
}
```

**Response:**
```json
{
  "embeddings": [
    [0.123, ...],
    [0.789, ...]
  ]
}
```

## Development and Testing

### Running Tests
Execute the test suite using `pytest`:
```bash
uv run pytest
```

### Test Structure
*   **`tests/test_api.py`:** Integration tests for the API endpoints, ensuring correct status codes and response formats (including handling of invalid inputs).
*   **`tests/test_model_unit.py`:** Unit tests for the underlying model, verifying loading, embedding shapes, and semantic similarity scores.

## Code Conventions
*   **Formatting/Linting:** The presence of `.ruff_cache` suggests `ruff` is used for linting and formatting.
*   **Type Hinting:** Python type hints are used (e.g., in `api.py` with `pydantic` models).
*   **Dependency Management:** `pyproject.toml` and `uv.lock` are the sources of truth for dependencies.
