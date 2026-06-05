# src/explain.py
import torch
import numpy as np


class RecommendationExplainer:
    def __init__(self, data_loader, device: str = "cpu"):
        """Post-hoc attention weight visualiser for the LSTM model."""
        self.data_loader = data_loader
        self.device      = torch.device(device if torch.cuda.is_available() else "cpu")

    def explain_single_user_session(self, model: torch.nn.Module,
                                    raw_history_tokens: np.ndarray,
                                    true_target: int):
        """
        Passes a user sequence through the trained LSTM, intercepts attention
        weights, and prints a ranked timeline heatmap with real movie titles.
        """
        model.eval()
        model.to(self.device)

        b_x = torch.tensor(raw_history_tokens, dtype=torch.long,
                            device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, attention_weights = model(b_x)
            predicted_idx = torch.argmax(logits, dim=1).item()

        # Flatten to [SeqLen]
        weights      = attention_weights.squeeze(0).squeeze(-1).cpu().numpy()
        history_list = raw_history_tokens.tolist()

        print("\n=========================================================================")
        print("             LIVE XAI DIAGNOSTIC INSIGHT CASE STUDY                      ")
        print("=========================================================================")
        print(f"User Active Watch History Length       : "
              f"{len([t for t in history_list if t != 0])} movies")
        print(f"Ground-Truth True Next Movie Watched   : "
              f"{self.data_loader.item_titles.get(true_target, 'Unknown')}")
        print(f"Model Primary Next Movie Recommendation: "
              f"{self.data_loader.item_titles.get(predicted_idx, 'Unknown')}")
        print(f"Inference Match Success Status         : {predicted_idx == true_target}")
        print("-------------------------------------------------------------------------")
        print("  TIMELINE WATCH HISTORY HEATMAP ATTENTION WEIGHTS:                      ")
        print("-------------------------------------------------------------------------")

        step = 1
        for token, weight in zip(reversed(history_list), reversed(weights)):
            if token == 0:
                continue
            title       = self.data_loader.item_titles.get(token, "Unknown Movie")
            clean_title = title if len(title) < 65 else title[:62] + "..."
            print(f" [-] Step -{step:02d} | "
                  f"Attention Weight: {weight * 100:05.2f}% | "
                  f"Name: {clean_title}")
            step += 1

        print("=========================================================================\n")