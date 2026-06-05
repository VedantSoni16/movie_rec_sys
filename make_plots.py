# make_plots.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)


def plot_training_curves(gru_history: dict, lstm_history: dict):
    """Plot train/val loss curves for both models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training & Validation Loss Curves", fontsize=14, fontweight="bold")

    for ax, history, title in zip(
        axes,
        [gru_history, lstm_history],
        ["GRU4Rec", "Stacked LSTM + Attention"],
    ):
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
        ax.plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=2,
                linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = "outputs/training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {path}")


def plot_benchmark_bar(scorecard: dict):
    """Bar chart comparing all models across all metrics."""
    metrics = ["Recall@10", "NDCG@10", "MRR@10", "Recall@50"]
    models  = list(scorecard.keys())
    x       = np.arange(len(metrics))
    width   = 0.18
    colors  = ["#9E9E9E", "#4FC3F7", "#FFB74D", "#81C784"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [scorecard[model].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model,
                      color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — 100-Item Pool Evaluation", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = "outputs/benchmark_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {path}")


def plot_recall_at_k_curve(model, test_tuple, user_histories, vocab_size,
                            device="cpu", is_lstm=True, label="LSTM"):
    """Recall@K curve from K=1 to K=50 for a given model."""
    import torch
    model.eval()
    dev       = torch.device(device)
    model.to(dev)
    X_test, y_test = test_tuple
    all_items = np.arange(1, vocab_size)
    K_values  = list(range(1, 51))
    hits      = {k: 0.0 for k in K_values}

    with torch.no_grad():
        for i in range(len(y_test)):
            true_target = int(y_test[i])
            negatives   = np.setdiff1d(all_items, list(user_histories[i]))
            sampled_neg = np.random.choice(negatives, size=99, replace=False)
            eval_pool   = np.concatenate([[true_target], sampled_neg])

            seq = torch.tensor(X_test[i], dtype=torch.long, device=dev).unsqueeze(0)
            logits = model(seq)[0] if is_lstm else model(seq)
            logits = logits.squeeze(0).cpu().numpy()

            pool_scores = logits[eval_pool]
            true_score  = pool_scores[0]
            rank        = np.sum(pool_scores > true_score) + 1

            for k in K_values:
                if rank <= k:
                    hits[k] += 1.0

    recalls = [hits[k] / len(y_test) for k in K_values]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(K_values, recalls, linewidth=2.5, color="#81C784", marker="o",
            markersize=3, label=label)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title(f"Recall@K Curve — {label}", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = f"outputs/recall_at_k_{label.replace(' ', '_').lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {path}")


def plot_attention_heatmap(weights: np.ndarray, titles: list, save_name: str):
    """Horizontal attention heatmap for a single user's history."""
    fig, ax = plt.subplots(figsize=(12, max(4, len(titles) * 0.35)))
    im = ax.imshow(weights.reshape(-1, 1), aspect="auto",
                   cmap="YlOrRd", vmin=0, vmax=weights.max())
    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles, fontsize=9)
    ax.set_xticks([])
    ax.set_title("Attention Weight Heatmap", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    plt.tight_layout()
    path = f"outputs/{save_name}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {path}")