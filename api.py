from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI()

# Load the Chinese text2vec model
print("Loading model...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("Model loaded.")

class SentenceRequest(BaseModel):
    sentences: List[str]

@app.post("/embed")
async def embed_sentences(request: SentenceRequest):
    embeddings = model.encode(request.sentences).tolist()
    return {"embeddings": embeddings}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
