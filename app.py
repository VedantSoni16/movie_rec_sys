# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
from src.models import StackedLSTMWithAttention

app = FastAPI(
    title="MovieLens-1M Sequential Recommendation API",
    description="Production inference server using a 2-Layer Stacked LSTM + Bahdanau Attention.",
    version="1.0.0"
)

# Global configuration boundaries
MODEL_PATH = "best_lstm_weights.pth"
VOCAB_SIZE = 3417  # Your verified MovieLens dataset vocabulary size

# Load the EXACT synchronized mapping file into backend memory
try:
    df_map = pd.read_csv('data/raw/movie_to_tokens.csv')
    # Create a direct lookup: {token_id: "Movie Title"}
    TOKEN_TO_TITLE = pd.Series(df_map.title.values, index=df_map.token_id).to_dict()
    print(f"[✓] Backend token translation map loaded successfully: {len(TOKEN_TO_TITLE)} entries.")
except Exception as e:
    print(f"[!] Warning: Could not load movie_to_tokens.csv in backend: {e}")
    TOKEN_TO_TITLE = {}

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
    runs an isolated inference pass, translates them to clean strings, and returns them.
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
    
    # Translate predicted tokens directly to titles using your exact data loader indexing rules
    translated_recommendations = []
    for token in recommended_ids:
        title = TOKEN_TO_TITLE.get(token, f"Unknown Token ID: {token}")
        translated_recommendations.append(title)
        
    # 7. Construct and return clean JSON response payload
    return {
        "input_history_depth": len(tokens),
        "recommended_tokens": recommended_ids,
        "translated_recommendations": translated_recommendations,
        "attention_weights": attention_scores
    }

@app.get("/health")
async def server_health_check():
    """Lightweight endpoint used by cloud orchestrators or Docker to verify availability."""
    return {"status": "healthy", "model_loaded": model is not None}