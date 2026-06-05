# src/word2vec.py
import numpy as np
from gensim.models import Word2Vec
from typing import List


class Word2VecBaseline:
    def __init__(self, vector_size: int = 64, window: int = 5):
        self.vector_size = vector_size
        self.window      = window
        self.model       = None

    def train(self, X_tokens: np.ndarray):
        print("[*] Reconstructing movie co-occurrence paths for Word2Vec training...")
        sessions: List[List[str]] = []
        for row in X_tokens:
            session = [f"item_{idx}" for idx in row if idx != 0]
            if session:
                sessions.append(session)

        print(f"[*] Extracted {len(sessions)} valid trajectories for Word2Vec.")
        print(f"[*] Training Skip-Gram embeddings "
              f"(dim={self.vector_size}, window={self.window})...")
        self.model = Word2Vec(
            sentences=sessions,
            vector_size=self.vector_size,
            window=self.window,
            sg=1,
            min_count=1,
            workers=4,
            epochs=15,
        )
        print("[+] Word2Vec embedding training complete.")

    def get_pytorch_embedding_matrix(self, vocab_size: int) -> np.ndarray:
        print("[*] Compiling pre-trained weight matrix for PyTorch embedding layer...")
        weight_matrix = np.zeros((vocab_size, self.vector_size), dtype=np.float32)
        for idx in range(1, vocab_size):
            key = f"item_{idx}"
            if key in self.model.wv:
                weight_matrix[idx] = self.model.wv[key]
            else:
                weight_matrix[idx] = np.random.normal(
                    scale=0.5, size=(self.vector_size,))
        return weight_matrix