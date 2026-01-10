import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "shibing624/text2vec-base-chinese")
MAX_BATCH_SIZE = 32
MAX_SENTENCE_LENGTH = 512


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    print(f"Loading model: {MODEL_NAME}...")
    try:
        app.state.model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise RuntimeError(f"Could not load model {MODEL_NAME}") from e
    yield
    # Clean up resources if needed
    print("Shutting down...")


app = FastAPI(
    title="Chinese Text2Vec API",
    description="API for generating Chinese text embeddings using Sentence Transformers.",
    version="0.1.0",
    lifespan=lifespan,
)


class SentenceRequest(BaseModel):
    sentences: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"List of sentences to embed (max {MAX_BATCH_SIZE})",
    )

    @field_validator("sentences")
    @classmethod
    def validate_sentence_length(cls, v: List[str]) -> List[str]:
        for i, sentence in enumerate(v):
            if len(sentence) > MAX_SENTENCE_LENGTH:
                raise ValueError(
                    f"Sentence at index {i} exceeds {MAX_SENTENCE_LENGTH} characters."
                )
            if not sentence.strip():
                raise ValueError(
                    f"Sentence at index {i} cannot be empty or whitespace only."
                )
        return v


@app.post("/embed")
def embed_sentences(request: Request, body: SentenceRequest):
    """
    Generate embeddings for a list of sentences.
    This endpoint is synchronous (def) to run in a thread pool and avoid blocking the event loop during heavy CPU operations.
    """
    model: SentenceTransformer = request.app.state.model
    embeddings = model.encode(body.sentences).tolist()
    return {"embeddings": embeddings}
