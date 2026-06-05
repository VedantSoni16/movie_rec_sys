# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class MovieLensDataset(Dataset):
    def __init__(self, X_tokens, y_targets):
        self.X_tokens  = torch.tensor(X_tokens,  dtype=torch.long)
        self.y_targets = torch.tensor(y_targets, dtype=torch.long)

    def __len__(self):
        return len(self.y_targets)

    def __getitem__(self, idx):
        return self.X_tokens[idx], self.y_targets[idx]


class RecommendationTrainer:
    def __init__(self, num_negatives: int = 100, device: str = "cpu"):
        self.num_negatives = num_negatives
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        os.makedirs("data/processed", exist_ok=True)

    # ------------------------------------------------------------------
    def compute_bpr_loss(self, logits: torch.Tensor,
                         positive_targets: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_size = logits.size()
        pos_logits  = logits.gather(dim=1, index=positive_targets.unsqueeze(-1))
        neg_indices = torch.randint(1, vocab_size,
                                    (batch_size, self.num_negatives),
                                    device=self.device)
        neg_logits  = logits.gather(dim=1, index=neg_indices)
        return -torch.mean(torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-8))

    # ------------------------------------------------------------------
    def train_epoch(self, model: nn.Module, dataloader: DataLoader,
                    optimizer: optim.Optimizer, is_lstm: bool) -> float:
        model.train()
        total_loss   = 0.0
        ce_criterion = nn.CrossEntropyLoss(ignore_index=0)

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()

            logits = model(batch_x)[0] if is_lstm else model(batch_x)

            ce_loss  = ce_criterion(logits, batch_y)
            bpr_loss = self.compute_bpr_loss(logits, batch_y)
            loss     = ce_loss + 0.5 * bpr_loss

            loss.backward()
            # FIX 3: gradient clipping prevents LSTM explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    # ------------------------------------------------------------------
    def evaluate_validation(self, model: nn.Module, dataloader: DataLoader,
                             is_lstm: bool) -> float:
        """FIX 2: Validation uses CE-only — no BPR stochastic noise."""
        model.eval()
        total_loss   = 0.0
        ce_criterion = nn.CrossEntropyLoss(ignore_index=0)

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = model(batch_x)[0] if is_lstm else model(batch_x)
                total_loss += ce_criterion(logits, batch_y).item()

        return total_loss / len(dataloader)

    # ------------------------------------------------------------------
    def fit_single_config(self, model: nn.Module, train_data: tuple,
                          val_data: tuple, epochs: int, batch_size: int,
                          lr: float, is_lstm: bool, save_name: str) -> dict:

        X_tr, y_tr = train_data
        X_va, y_va = val_data
        train_loader = DataLoader(MovieLensDataset(X_tr, y_tr),
                                  batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(MovieLensDataset(X_va, y_va),
                                  batch_size=batch_size, shuffle=False)
        model = model.to(self.device)

        # ------ Optimizer setup ----------------------------------------
        if is_lstm:
            # Embedding group starts frozen (lr=0); LSTM body trains at lr
            optimizer = optim.Adam([
                {"params": model.embedding.parameters(), "lr": 0.0},
                {"params": [p for n, p in model.named_parameters()
                            if "embedding" not in n], "lr": lr}
            ], weight_decay=1e-5)
            # FIX 1: Scheduler only controls param_group[1] (LSTM body).
            # Embedding lr is managed manually at epoch 6 — NOT by scheduler.
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-5,
                last_epoch=-1
            )
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-5
            )

        history         = {"train_loss": [], "val_loss": []}
        best_val_loss   = float("inf")
        patience        = 15
        patience_counter = 0

        print(f"\n[*] Commencing training run loop for {save_name}...")

        for epoch in range(epochs):
            current_epoch = epoch + 1

            # FIX 1 (continued): Unfreeze embedding at epoch 6 manually.
            # We then immediately re-pin param_group[1] lr so the scheduler
            # step below does not accidentally reset it.
            if is_lstm and current_epoch == 6:
                print("     [!] Triggering gradual unfreezing: "
                      "Activating embedding weights at lr=1e-4...")
                optimizer.param_groups[0]["lr"] = 1e-4
                # Snapshot body lr before scheduler touches it
                body_lr_before = optimizer.param_groups[1]["lr"]

            train_loss = self.train_epoch(model, train_loader, optimizer,
                                          is_lstm=is_lstm)
            val_loss   = self.evaluate_validation(model, val_loader,
                                                  is_lstm=is_lstm)

            # Step scheduler (affects param_group[1] for LSTM, all for GRU)
            scheduler.step()

            # FIX 1: After scheduler.step(), lock embedding lr to 1e-4
            # so cosine decay never erodes the just-unfrozen embedding group.
            if is_lstm and current_epoch >= 6:
                optimizer.param_groups[0]["lr"] = 1e-4

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            active_lr = (optimizer.param_groups[1]["lr"] if is_lstm
                         else optimizer.param_groups[0]["lr"])
            print(f"      -> Epoch {current_epoch:02d}/{epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Core LR: {active_lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_name)
                print(f"         [+] Validation loss improved. Checkpoint saved.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"         [!] Early stopping at epoch {current_epoch}.")
                    break

        # Persist training curves for plotting
        torch.save({"history": history, "model": save_name},
                   f"data/processed/history_{save_name.replace('.pth','')}.pth")
        return history