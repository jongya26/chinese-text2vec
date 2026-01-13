import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

# --- Model Configuration & Registry ---

SUPPORTED_MODELS = {
    "text2vec-base-chinese": {
        "id": "shibing624/text2vec-base-chinese",
        "name": "Text2Vec Base Chinese",
        "description": "Standard Chinese text embedding model, well-balanced for most tasks.",
        "languages": ["zh"],
        "dimensions": 768,
    },
    "bge-small-zh-v1.5": {
        "id": "BAAI/bge-small-zh-v1.5",
        "name": "BGE Small Chinese v1.5",
        "description": "High-efficiency Chinese model from BAAI with excellent performance-to-size ratio.",
        "languages": ["zh"],
        "dimensions": 512,
    },
    "bge-large-zh-v1.5": {
        "id": "BAAI/bge-large-zh-v1.5",
        "name": "BGE Large Chinese v1.5",
        "description": "State-of-the-art large Chinese model for maximum semantic accuracy.",
        "languages": ["zh"],
        "dimensions": 1024,
    },
    "all-MiniLM-L6-v2": {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "name": "All-MiniLM-L6-v2",
        "description": "Fast and lightweight model optimized for English text.",
        "languages": ["en"],
        "dimensions": 384,
    },
    "multilingual-e5-small": {
        "id": "intfloat/multilingual-e5-small",
        "name": "Multilingual E5 Small",
        "description": "High-performance multilingual model supporting 94 languages including EN and ZH.",
        "languages": ["en", "zh", "multilingual"],
        "dimensions": 384,
    }
}

DEFAULT_MODEL_KEY = os.getenv("MODEL_NAME", "text2vec-base-chinese")
MAX_BATCH_SIZE = 32
MAX_SENTENCE_LENGTH = 512

class ModelManager:
    def __init__(self):
        self._loaded_models: Dict[str, SentenceTransformer] = {}

    def get_model(self, model_key: str) -> SentenceTransformer:
        # 1. Try direct lookup by key
        if model_key in SUPPORTED_MODELS:
            target_key = model_key
        else:
            # 2. Try lookup by model ID
            target_key = next((k for k, v in SUPPORTED_MODELS.items() if v["id"] == model_key), None)
            
        if not target_key:
            raise ValueError(f"Model '{model_key}' is not supported.")
        
        if target_key not in self._loaded_models:
            model_info = SUPPORTED_MODELS[target_key]
            print(f"Loading model: {model_info['id']} ({target_key})...")
            self._loaded_models[target_key] = SentenceTransformer(model_info['id'])
            print(f"Model {target_key} loaded successfully.")
            
        return self._loaded_models[target_key]

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the default model
    try:
        model_manager.get_model(DEFAULT_MODEL_KEY)
    except Exception as e:
        print(f"Failed to load default model: {e}")
    yield
    # Clean up
    model_manager._loaded_models.clear()
    print("Shutting down...")

app = FastAPI(
    title="Chinese Text2Vec API",
    description="Multi-model API for generating text embeddings.",
    version="0.2.0",
    lifespan=lifespan,
)

# --- Pydantic Models ---

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    languages: List[str]
    dimensions: int

class EmbeddingRequest(BaseModel):
    input: List[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    model: Optional[str] = None

    @field_validator("input")
    @classmethod
    def validate_input_length(cls, v: List[str]) -> List[str]:
        for i, text in enumerate(v):
            if len(text) > MAX_SENTENCE_LENGTH:
                raise ValueError(f"Text at index {i} exceeds {MAX_SENTENCE_LENGTH} characters.")
            if not text.strip():
                 raise ValueError(f"Text at index {i} cannot be empty.")
        return v

class LegacySentenceRequest(BaseModel):
    sentences: List[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)

# --- Endpoints ---

@app.get("/v1/models")
def list_models():
    """List all supported embedding models and their metadata."""
    return {"data": [info for info in SUPPORTED_MODELS.values()]}

@app.post("/v1/embeddings")
def create_embeddings(body: EmbeddingRequest):
    """
    Standardized embedding endpoint (OpenAI style).
    If no model is provided, returns the list of supported models.
    """
    if not body.model:
        return list_models()

    try:
        model = model_manager.get_model(body.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    embeddings = model.encode(body.input).tolist()
    
    # Standard format: list of objects with index and embedding
    data = [
        {"object": "embedding", "embedding": emb, "index": i}
        for i, emb in enumerate(embeddings)
    ]
    
    return {
        "object": "list",
        "data": data,
        "model": body.model,
        "usage": {
            "prompt_tokens": sum(len(t) for t in body.input), # Simple char-based estimation
            "total_tokens": sum(len(t) for t in body.input)
        }
    }

@app.post("/embed")
def legacy_embed(body: LegacySentenceRequest):
    """
    Backward compatible endpoint for the original /embed route.
    Uses the default model.
    """
    model = model_manager.get_model(DEFAULT_MODEL_KEY)
    embeddings = model.encode(body.sentences).tolist()
    return {"embeddings": embeddings}