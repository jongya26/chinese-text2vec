from sentence_transformers import SentenceTransformer

# Load the Chinese text2vec model
print("Loading model...")
model = SentenceTransformer("shibing624/text2vec-base-chinese")

# Test sentences
sentences = ["如何更换花呗绑定银行卡", "花呗更改绑定银行卡", "今天天气很好"]

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(sentences)

# Print results
print(f"\nEmbedding shape: {embeddings.shape}")
print(
    f"\nSimilarity between sentence 1 and 2: {model.similarity(embeddings[0], embeddings[1]).item():.4f}"
)
print(
    f"Similarity between sentence 1 and 3: {model.similarity(embeddings[0], embeddings[2]).item():.4f}"
)
