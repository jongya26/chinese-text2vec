import pytest
from fastapi.testclient import TestClient
from api import app  # Assuming api.py is in the project root
import numpy as np

client = TestClient(app)


def test_embed_single_sentence():
    response = client.post("/embed", json={"sentences": ["今天天气很好"]})
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)
    assert len(data["embeddings"]) == 1
    assert len(data["embeddings"][0]) == 768  # Assuming 768-dimensional embeddings


def test_embed_multiple_sentences():
    sentences = ["你好世界", "再见世界", "今天上海天气怎么样？"]
    response = client.post("/embed", json={"sentences": sentences})
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)
    assert len(data["embeddings"]) == len(sentences)
    assert all(len(emb) == 768 for emb in data["embeddings"])


def test_embed_empty_sentences_list():
    response = client.post("/embed", json={"sentences": []})
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)
    assert len(data["embeddings"]) == 0


def test_embed_invalid_input():
    # Test with missing "sentences" key
    response = client.post("/embed", json={"text": ["some text"]})
    assert response.status_code == 422  # Unprocessable Entity

    # Test with non-list input for sentences
    response = client.post("/embed", json={"sentences": "single sentence"})
    assert response.status_code == 422  # Unprocessable Entity

    # Test with list containing non-string elements
    response = client.post("/embed", json={"sentences": ["hello", 123]})
    assert response.status_code == 422  # Unprocessable Entity
