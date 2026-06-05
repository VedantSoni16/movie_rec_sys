# run_pipeline.py
"""
Entry point for the MovieLens-1M Sequential Recommendation Pipeline.
Run:  python run_pipeline.py
All outputs (weights, plots, scorecard) are saved automatically.
"""
import os
import sys
import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import MovieLensDataLoader
from word2vec    import Word2VecBaseline
from models      import GRU4Rec, StackedLSTMWithAttention
from train       import RecommendationTrainer
from evaluate    import MovieLensEvaluator
from explain     import RecommendationExplainer
from make_plots  import (plot_training_curves, plot_benchmark_bar,
                          plot_recall_at_k_curve, plot_attention_heatmap)

# ── Config ────────────────────────────────────────────────────────────────────
RATINGS_PATH  = "data/raw/ratings.dat"
MOVIES_PATH   = "data/raw/movies.dat"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 64
HIDDEN_DIM    = 128
DROPOUT       = 0.3
BATCH_SIZE    = 256
EPOCHS        = 50
GRU_LR        = 1e-3
LSTM_LR       = 3e-4
NUM_NEG       = 100
MIN_INTER     = 5
MAX_SEQ_LEN   = 50
XAI_MIN_HIST  = 25   # minimum non-padding tokens for a good case-study user

print("\n" + "=" * 72)
print("  MOVIELENS-1M SEQUENTIAL RECOMMENDATION PIPELINE")
print("=" * 72)

# ── Phase 1: Data ─────────────────────────────────────────────────────────────
print("\n[PHASE 1] Data ingestion & leave-one-out split")
print("-" * 72)
loader = MovieLensDataLoader(min_interactions=MIN_INTER, max_seq_len=MAX_SEQ_LEN)
train, val, test, vocab_size = loader.pipeline(RATINGS_PATH, MOVIES_PATH)
X_train, y_train = train
X_val,   y_val   = val
X_test,  y_test  = test
print(f"[✓] Train users: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
print(f"[✓] Vocab size : {vocab_size}")

# ── Phase 2: Word2Vec ─────────────────────────────────────────────────────────
print("\n[PHASE 2] Word2Vec Skip-Gram pretraining")
print("-" * 72)
w2v = Word2VecBaseline(vector_size=EMBEDDING_DIM, window=5)
w2v.train(X_train)
embed_matrix = w2v.get_pytorch_embedding_matrix(vocab_size)

# ── Phase 3: Model init ───────────────────────────────────────────────────────
print("\n[PHASE 3] Initialising model architectures")
print("-" * 72)
gru_model  = GRU4Rec(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
lstm_model = StackedLSTMWithAttention(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT)
lstm_model.load_pretrained_embeddings(embed_matrix)
print(f"[✓] GRU4Rec params : "
      f"{sum(p.numel() for p in gru_model.parameters()):,}")
print(f"[✓] LSTM+Attn params: "
      f"{sum(p.numel() for p in lstm_model.parameters()):,}")

# ── Phase 4: Training ─────────────────────────────────────────────────────────
print("\n[PHASE 4] Training models")
print("-" * 72)
trainer = RecommendationTrainer(num_negatives=NUM_NEG, device=DEVICE)

print("\n[*] Training Baseline GRU4Rec Model (lr=1e-3)...")
gru_history = trainer.fit_single_config(
    model=gru_model, train_data=train, val_data=val,
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=GRU_LR,
    is_lstm=False, save_name="best_gru_weights.pth",
)

print("\n[*] Training Primary Stacked LSTM + Attention Model (lr=3e-4)...")
lstm_history = trainer.fit_single_config(
    model=lstm_model, train_data=train, val_data=val,
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LSTM_LR,
    is_lstm=True, save_name="best_lstm_weights.pth",
)

# ── Phase 5: Evaluation ───────────────────────────────────────────────────────
print("\n[PHASE 5] Benchmark evaluation (100-item pool)")
print("-" * 72)

# Reload best checkpointed weights before evaluating
gru_model.load_state_dict(torch.load("best_gru_weights.pth",
                                      map_location=DEVICE))
lstm_model.load_state_dict(torch.load("best_lstm_weights.pth",
                                       map_location=DEVICE))

evaluator = MovieLensEvaluator(vocab_size=vocab_size, device=DEVICE)
table, scorecard = evaluator.run_benchmark_suite(
    train, val, test, w2v, gru_model, lstm_model)

# ── Phase 6: XAI case study ───────────────────────────────────────────────────
print("\n[PHASE 6] XAI Explainability Case Study")
print("-" * 72)
explainer = RecommendationExplainer(data_loader=loader, device=DEVICE)

# Pick first test user with enough non-padding history for a meaningful heatmap
case_idx = None
for idx in range(len(y_test)):
    if np.count_nonzero(X_test[idx]) >= XAI_MIN_HIST:
        case_idx = idx
        break

if case_idx is not None:
    explainer.explain_single_user_session(
        model=lstm_model,
        raw_history_tokens=X_test[case_idx],
        true_target=int(y_test[case_idx]),
    )

    # Also save the attention heatmap as an image
    lstm_model.eval()
    with torch.no_grad():
        seq   = torch.tensor(X_test[case_idx], dtype=torch.long,
                              device=DEVICE).unsqueeze(0)
        _, attn = lstm_model(seq)
    weights = attn.squeeze(0).squeeze(-1).cpu().numpy()

    # Build title list (non-padding positions, most recent first)
    history   = X_test[case_idx].tolist()
    non_pad   = [(t, w) for t, w in zip(reversed(history), reversed(weights))
                 if t != 0]
    titles    = [loader.item_titles.get(t, f"item_{t}") for t, _ in non_pad]
    w_vals    = np.array([w for _, w in non_pad])
    plot_attention_heatmap(w_vals, titles, "attention_heatmap_case_study.png")
else:
    print("[!] No suitable test user found for XAI case study.")

# ── Phase 7: Plots ────────────────────────────────────────────────────────────
print("\n[PHASE 7] Generating output plots")
print("-" * 72)
plot_training_curves(gru_history, lstm_history)
plot_benchmark_bar(scorecard)

# Recall@K curve for LSTM
user_histories = evaluator.build_global_user_histories(train, val, test)
plot_recall_at_k_curve(
    lstm_model, test, user_histories, vocab_size,
    device=DEVICE, is_lstm=True, label="LSTM + Attention",
)

print("\n" + "=" * 72)
print("  PIPELINE COMPLETE — check outputs/ folder for all plots")
print("=" * 72 + "\n")