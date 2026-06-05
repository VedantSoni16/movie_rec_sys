# src/evaluate.py
import torch
import numpy as np
import pandas as pd
from typing import Dict, Set


class MovieLensEvaluator:
    def __init__(self, vocab_size: int, device: str = "cpu"):
        """
        100-item pool evaluator. Ranks 1 true target against 99 random negatives.
        """
        self.vocab_size = vocab_size
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    def build_global_user_histories(self, train_tuple, val_tuple,
                                    test_tuple) -> Dict[int, Set[int]]:
        """Collects every item each user has ever seen to avoid false negatives."""
        X_tr, y_tr = train_tuple
        X_va, y_va = val_tuple
        X_te, y_te = test_tuple
        num_users   = len(y_te)
        user_histories = {}

        for i in range(num_users):
            seen = set()
            seen.update(int(t) for t in X_tr[i] if t != 0)
            seen.add(int(y_tr[i]))
            seen.update(int(t) for t in X_va[i] if t != 0)
            seen.add(int(y_va[i]))
            seen.update(int(t) for t in X_te[i] if t != 0)
            seen.add(int(y_te[i]))
            user_histories[i] = seen

        return user_histories

    # ------------------------------------------------------------------
    def _rank_in_pool(self, pool_scores: np.ndarray) -> float:
        """Returns 1-based rank of index-0 item (the true target)."""
        true_score          = pool_scores[0]
        higher              = np.sum(pool_scores > true_score)
        ties                = np.sum(pool_scores == true_score) - 1
        return higher + 1 + ties / 2.0

    # ------------------------------------------------------------------
    def calculate_pool_metrics(self, model: torch.nn.Module, test_tuple: tuple,
                               user_histories: Dict[int, Set[int]],
                               is_lstm: bool = True) -> Dict[str, float]:
        model.eval()
        model.to(self.device)

        X_test, y_test = test_tuple
        num_users      = len(y_test)
        hits_10 = ndcg_10 = mrr_10 = hits_50 = 0.0
        all_items = np.arange(1, self.vocab_size)

        print(f"[*] Processing 100-item pool ranking metrics across {num_users} users...")

        with torch.no_grad():
            for i in range(num_users):
                true_target = int(y_test[i])
                negatives   = np.setdiff1d(all_items, list(user_histories[i]))
                sampled_neg = np.random.choice(negatives, size=99, replace=False)
                eval_pool   = np.concatenate([[true_target], sampled_neg])

                seq_tensor  = torch.tensor(X_test[i], dtype=torch.long,
                                           device=self.device).unsqueeze(0)
                logits      = (model(seq_tensor)[0] if is_lstm
                               else model(seq_tensor))
                logits      = logits.squeeze(0).cpu().numpy()

                pool_scores = logits[eval_pool]
                rank        = self._rank_in_pool(pool_scores)

                if rank <= 10:
                    hits_10 += 1.0
                    mrr_10  += 1.0 / rank
                    ndcg_10 += 1.0 / np.log2(rank + 1)
                if rank <= 50:
                    hits_50 += 1.0

        return {
            "Recall@10": hits_10 / num_users,
            "NDCG@10":   ndcg_10 / num_users,
            "MRR@10":    mrr_10  / num_users,
            "Recall@50": hits_50 / num_users,
        }

    # ------------------------------------------------------------------
    def evaluate_word2vec_baseline(self, w2v_baseline, test_tuple: tuple,
                                   user_histories: Dict[int, Set[int]]) -> Dict[str, float]:
        X_test, y_test = test_tuple
        num_users      = len(y_test)
        hits_10 = ndcg_10 = mrr_10 = hits_50 = 0.0
        all_items = np.arange(1, self.vocab_size)

        for i in range(num_users):
            true_target = int(y_test[i])
            negatives   = np.setdiff1d(all_items, list(user_histories[i]))
            sampled_neg = np.random.choice(negatives, size=99, replace=False)
            eval_pool   = np.concatenate([[true_target], sampled_neg])

            active      = [int(t) for t in X_test[i] if t != 0]
            seed_item   = active[-1] if active else 1
            seed_str    = f"item_{seed_item}"

            w2v_scores = np.zeros(self.vocab_size)
            if w2v_baseline.model and seed_str in w2v_baseline.model.wv:
                for idx in eval_pool:
                    tgt = f"item_{idx}"
                    if tgt in w2v_baseline.model.wv:
                        w2v_scores[idx] = w2v_baseline.model.wv.similarity(
                            seed_str, tgt)

            pool_scores = w2v_scores[eval_pool]
            rank        = self._rank_in_pool(pool_scores)

            if rank <= 10:
                hits_10 += 1.0
                mrr_10  += 1.0 / rank
                ndcg_10 += 1.0 / np.log2(rank + 1)
            if rank <= 50:
                hits_50 += 1.0

        return {
            "Recall@10": hits_10 / num_users,
            "NDCG@10":   ndcg_10 / num_users,
            "MRR@10":    mrr_10  / num_users,
            "Recall@50": hits_50 / num_users,
        }

    # ------------------------------------------------------------------
    def run_benchmark_suite(self, train_tuple, val_tuple, test_tuple,
                            w2v_baseline, gru_model, lstm_model) -> str:
        user_histories = self.build_global_user_histories(
            train_tuple, val_tuple, test_tuple)

        # FIX 9: Honest random-chance baseline (theoretical, clearly labelled)
        pop_metrics = {
            "Recall@10": 10.0 / 100.0,
            "NDCG@10":   float(np.mean([1.0 / np.log2(r + 1) if r <= 10 else 0
                                        for r in np.random.randint(1, 101, 1000)])),
            "MRR@10":    float(np.mean([1.0 / r if r <= 10 else 0
                                        for r in np.random.randint(1, 101, 1000)])),
            "Recall@50": 50.0 / 100.0,
        }

        w2v_metrics  = self.evaluate_word2vec_baseline(
            w2v_baseline, test_tuple, user_histories)
        gru_metrics  = self.calculate_pool_metrics(
            gru_model,  test_tuple, user_histories, is_lstm=False)
        lstm_metrics = self.calculate_pool_metrics(
            lstm_model, test_tuple, user_histories, is_lstm=True)

        scorecard = {
            "Random Chance (theoretical)": pop_metrics,
            "Word2Vec Baseline":           w2v_metrics,
            "GRU4Rec":                     gru_metrics,
            "LSTM + Attention (ours)":     lstm_metrics,
        }

        df    = pd.DataFrame(scorecard).T
        table = df.to_markdown(floatfmt=".4f")
        print("\nFINAL BENCHMARK SCORECARD:")
        print(table)
        return table, scorecard