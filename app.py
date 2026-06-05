# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from src.models import StackedLSTMWithAttention

app = FastAPI(
    title="MovieLens-1M Sequential Recommendation API",
    description="Production inference server using a 2-Layer Stacked LSTM + Bahdanau Attention.",
    version="1.0.0"
)

# Global configuration boundaries
MODEL_PATH = "best_lstm_weights.pth"
VOCAB_SIZE = 3417  # Your verified MovieLens dataset vocabulary size

# 1. Initialize the architecture shape and load frozen weights onto CPU
try:
    model = StackedLSTMWithAttention(vocab_size=VOCAB_SIZE, embedding_dim=64, hidden_dim=128, dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()  # Lock down layers (freezes dropout and batchnorm)
    print(f"[✓] Inference engine successfully configured from checkpoint: {MODEL_PATH}")
except FileNotFoundError:
    print(f"[!] Warning: {MODEL_PATH} not found. Ensure training has run before starting server.")
    model = None

# 2. Define the exact input validation schema using Pydantic
class RecommendationRequest(BaseModel):
    movie_history_tokens: list[int]

@app.post("/predict")
async def predict_next_movies(payload: RecommendationRequest):
    """
    Exposes an HTTP POST endpoint that takes a list of movie history tokens,
    runs an isolated inference pass, and returns the top 5 recommendations.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Inference model weights asset is not initialized on server.")
        
    # Strip padding out if accidentally passed from frontend to get true history depth
    tokens = [t for t in payload.movie_history_tokens if t != 0]
    
    if len(tokens) == 0:
        raise HTTPException(status_code=400, detail="Incoming movie watch history sequence cannot be empty.")
        
    # 3. Apply Left-Padding or Truncation to match our rigid 50-step sequence length
    max_len = 50
    if len(tokens) >= max_len:
        padded_tokens = tokens[-max_len:]
    else:
        padded_tokens = [0] * (max_len - len(tokens)) + tokens
        
    # 4. Wrap array into a PyTorch evaluation tensor batch: Shape [1, 50]
    input_tensor = torch.tensor(padded_tokens, dtype=torch.long).unsqueeze(0)
    
    # 5. Execute forward pass through frozen network weights
    with torch.no_grad():
        logits, attention_weights = model(input_tensor)
        
    # 6. Extract top 5 recommended movie tokens
    top_k_results = torch.topk(logits, k=5, dim=1)
    recommended_ids = top_k_results.indices.squeeze(0).tolist()
    
    # Extract attention weights, squeezing batch dims to map directly over the 50 steps
    attention_scores = attention_weights.squeeze(0).squeeze(-1).tolist()
    
    # 7. Construct and return clean JSON response payload
    return {
        "input_history_depth": len(tokens),
        "recommended_tokens": recommended_ids,
        "attention_weights": attention_scores
    }

@app.get("/health")
async def server_health_check():
    """Lightweight endpoint used by cloud orchestrators or Docker to verify availability."""
    return {"status": "healthy", "model_loaded": model is not None}