# src/data_loader.py
import os
import pandas as pd
import numpy as np


class MovieLensDataLoader:
    def __init__(self, min_interactions: int = 5, max_seq_len: int = 50):
        self.min_interactions = min_interactions
        self.max_seq_len      = max_seq_len
        self.item_titles      = {}
        self.vocab_size       = 0

    def load_raw_data(self, ratings_path: str, movies_path: str):
        print("[*] Parsing raw MovieLens data streams using python engine...")
        ratings_df = pd.read_csv(
            ratings_path, sep="::", engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
            dtype={"user_id": int, "movie_id": int, "rating": int, "timestamp": int},
        )
        movies_df = pd.read_csv(
            movies_path, sep="::", engine="python",
            names=["movie_id", "title", "genres"],
            encoding="ISO-8859-1",
        )
        return ratings_df, movies_df

    def apply_core_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"[*] Applying iterative core filter (min_interactions >= {self.min_interactions})...")
        starting_rows = len(df)
        while True:
            keep_users  = df["user_id"].value_counts()
            keep_movies = df["movie_id"].value_counts()
            keep_users  = keep_users[keep_users  >= self.min_interactions].index
            keep_movies = keep_movies[keep_movies >= self.min_interactions].index
            filtered    = df[df["user_id"].isin(keep_users) &
                             df["movie_id"].isin(keep_movies)]
            if len(filtered) == len(df):
                break
            df = filtered
        print(f"[✓] Core stabilized: Retained {len(df)}/{starting_rows} rows.")
        return df

    def compile_token_mappings(self, filtered_df: pd.DataFrame,
                               movies_df: pd.DataFrame) -> dict:
        print("[*] Generating high-signal sequential token mappings...")
        unique_movies  = sorted(filtered_df["movie_id"].unique())
        movie_to_token = {raw_id: idx + 1 for idx, raw_id in enumerate(unique_movies)}
        self.vocab_size = len(unique_movies) + 1

        meta_dict      = dict(zip(movies_df["movie_id"], movies_df["title"]))
        self.item_titles = {0: "[PAD]"}
        for raw_id, token_id in movie_to_token.items():
            self.item_titles[token_id] = meta_dict.get(raw_id, f"Unknown:{raw_id}")

        print(f"[✓] Vocabulary size finalized at: {self.vocab_size} unique tokens.")
        return movie_to_token

    def _pad(self, tokens: list, max_len: int) -> list:
        if len(tokens) >= max_len:
            return tokens[-max_len:]
        return [0] * (max_len - len(tokens)) + tokens

    def generate_leave_one_out_matrices(self, filtered_df: pd.DataFrame,
                                        movie_to_token: dict) -> tuple:
        print("[*] Grouping user viewing timelines and applying leave-one-out splits...")
        tr_X, tr_y = [], []
        va_X, va_y = [], []
        te_X, te_y = [], []

        grouped = filtered_df.sort_values("timestamp").groupby("user_id")
        for _, group in grouped:
            tokens = [movie_to_token[mid] for mid in group["movie_id"].values]
            if len(tokens) < 3:
                continue

            test_target  = tokens[-1]
            val_target   = tokens[-2]

            tr_X.append(self._pad(tokens[:-2], self.max_seq_len))
            tr_y.append(tokens[-2])

            va_X.append(self._pad(tokens[:-1], self.max_seq_len))
            va_y.append(val_target)

            te_X.append(self._pad(tokens[:-1], self.max_seq_len))
            te_y.append(test_target)

        train = (np.array(tr_X), np.array(tr_y))
        val   = (np.array(va_X), np.array(va_y))
        test  = (np.array(te_X), np.array(te_y))
        return train, val, test

    def pipeline(self, ratings_path: str, movies_path: str) -> tuple:
        ratings_df, movies_df = self.load_raw_data(ratings_path, movies_path)
        filtered_df           = self.apply_core_filtering(ratings_df)
        movie_to_token        = self.compile_token_mappings(filtered_df, movies_df)
        train, val, test      = self.generate_leave_one_out_matrices(
            filtered_df, movie_to_token)
        return train, val, test, self.vocab_size