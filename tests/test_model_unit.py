import pytest
from sentence_transformers import SentenceTransformer
import numpy as np


@pytest.fixture(scope="module")
def model():
    """Fixture to load the model once for all tests in this module."""
    print("Loading model for unit tests...")
    return SentenceTransformer("shibing624/text2vec-base-chinese")


def test_model_loading(model):
    """Test that the model loads successfully."""
    assert isinstance(model, SentenceTransformer)
    assert model.max_seq_length > 0


def test_embeddings_shape(model):
    """Test that the embeddings have the correct shape."""
    sentences = ["你好世界", "再见世界"]
    embeddings = model.encode(sentences)
    assert embeddings.shape == (len(sentences), 768)
    assert embeddings.dtype == np.float32


def test_similarity_score(model):
    """Test similarity scores for known similar and dissimilar sentences."""
    # Similar sentences
    sentence1 = "如何更换花呗绑定银行卡"
    sentence2 = "花呗更改绑定银行卡"
    # Dissimilar sentences
    sentence3 = "今天天气很好"

    embeddings = model.encode([sentence1, sentence2, sentence3])
    embedding1, embedding2, embedding3 = embeddings[0], embeddings[1], embeddings[2]

    # Calculate similarity
    similarity_1_2 = model.similarity(embedding1, embedding2).item()
    similarity_1_3 = model.similarity(embedding1, embedding3).item()

    # Print results for observation
    print(f"\nSimilarity between sentence 1 and 2: {similarity_1_2:.4f}")
    print(f"Similarity between sentence 1 and 3: {similarity_1_3:.4f}")

    # Assertions
    assert similarity_1_2 > 0.8  # Very similar
    assert similarity_1_3 < 0.5  # Dissimilar
    assert 0 <= similarity_1_2 <= 1
    assert 0 <= similarity_1_3 <= 1
