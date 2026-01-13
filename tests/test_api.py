import pytest
from fastapi.testclient import TestClient
from api import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) >= 5
    assert any(m["id"] == "shibing624/text2vec-base-chinese" for m in data)

def test_v1_embeddings_no_model(client):
    """When no model is provided, it should return the list of models."""
    response = client.post(
        "/v1/embeddings",
        json={"input": ["你好", "世界"]}
    )
    assert response.status_code == 200
    res_json = response.json()
    assert "data" in res_json
    assert any(m["id"] == "shibing624/text2vec-base-chinese" for m in res_json["data"])

def test_v1_embeddings_with_explicit_model(client):
    """Providing a model explicitly should return embeddings."""
    response = client.post(
        "/v1/embeddings",
        json={
            "input": ["你好", "世界"],
            "model": "text2vec-base-chinese"
        }
    )
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["object"] == "list"
    assert len(res_json["data"]) == 2
    assert "embedding" in res_json["data"][0]
    assert len(res_json["data"][0]["embedding"]) == 768

def test_v1_embeddings_alternative_model(client):
    # Test with a smaller model to verify switch
    response = client.post(
        "/v1/embeddings",
        json={
            "input": ["Hello world"],
            "model": "all-MiniLM-L6-v2"
        }
    )
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["model"] == "all-MiniLM-L6-v2"
    assert len(res_json["data"][0]["embedding"]) == 384

def test_v1_embeddings_invalid_model(client):
    response = client.post(
        "/v1/embeddings",
        json={
            "input": ["test"],
            "model": "non-existent-model"
        }
    )
    assert response.status_code == 400

def test_legacy_embed(client):
    response = client.post(
        "/embed",
        json={"sentences": ["测试一下"]}
    )
    assert response.status_code == 200
    assert "embeddings" in response.json()

def test_v1_embeddings_validation(client):
    # Too many sentences
    response = client.post(
        "/v1/embeddings",
        json={"input": ["test"] * 33}
    )
    assert response.status_code == 422