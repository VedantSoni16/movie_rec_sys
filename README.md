# 🎬 Explainable AI Sequential Recommendation Engine

An end-to-end sequential recommendation system built on the MovieLens-1M dataset that models user preferences as evolving behavioral sequences rather than static profiles. The project combines deep learning, explainable AI, and production deployment through a FastAPI backend and Streamlit frontend.

---

## 🚀 Overview

Traditional recommendation systems often ignore the order in which user interactions occur. This project treats a user's viewing history as a chronological sequence and predicts the next likely interaction using recurrent neural networks and attention mechanisms.

Key objectives:

- Capture evolving user preferences over time
- Improve next-item recommendation accuracy
- Provide interpretable recommendations through attention visualization
- Deploy the model through a production-ready API and web interface

---

## ✨ Features

- Sequential recommendation using a 2-layer LSTM
- Custom Bahdanau (Additive) Attention mechanism
- Explainable AI visualization of attention weights
- FastAPI inference service
- Streamlit interactive dashboard
- Dockerized deployment
- Word2Vec and GRU4Rec baselines for comparison
- MovieLens-1M benchmark evaluation

---

## 🏗️ System Architecture

User History → Embedding Layer → Stacked LSTM → Additive Attention → Recommendation Head → Top-N Predictions

The attention mechanism highlights which historical interactions contributed most to the recommendation, improving interpretability.

---

## 📂 Repository Structure

```text
SEQ_REC/
│
├── data/
│   └── raw/
│       ├── movies.dat
│       └── movie_to_tokens.csv
│
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── models.py
│   ├── train.py
│   └── word2vec.py
│
├── app.py
├── frontend.py
├── generate_map.py
├── Dockerfile
├── requirements.txt
├── best_lstm_weights.pth
└── best_gru_weights.pth
```

---

## 🧠 Model Architecture

### Primary Model: LSTM + Attention

- 64-dimensional item embeddings
- 2-layer stacked LSTM
- Dropout regularization (0.3)
- Layer normalization
- Custom Bahdanau Attention
- Fully connected prediction head

### Attention Mechanism

The model computes attention scores using:

```math
Score(h_t, q) = V^T tanh(W_h h_t + W_q q)
```

Benefits:

- Focuses on relevant past interactions
- Improves interpretability
- Handles long-range dependencies better than standard recurrent models

### Baselines

#### GRU4Rec
- Single-layer GRU architecture
- Next-item prediction benchmark

#### Word2Vec
- Skip-Gram item embeddings
- Non-sequential baseline

---

## 📊 Data Pipeline

### K-Core Filtering

To reduce sparsity and noise, the dataset undergoes iterative K-core filtering:

- Remove users with fewer than 5 interactions
- Remove items with fewer than 5 interactions
- Repeat until convergence

### Token Mapping

A dedicated token mapping layer ensures alignment between:

- Internal model indices
- MovieLens item IDs
- Human-readable movie titles

This prevents prediction-to-title mismatches during inference.

---

## ⚙️ Training Strategy

### Loss Functions

The model combines:

1. Cross-Entropy Loss
2. Bayesian Personalized Ranking (BPR) Loss

This enables both:

- Accurate classification
- Better ranking quality

### Optimization Techniques

- Gradient clipping (max norm = 1.0)
- Cosine annealing learning rate scheduling
- Gradual embedding unfreezing
- Early stopping based on validation performance

---

## 📈 Benchmark Results

| Model | Recall@10 | NDCG@10 | MRR@10 | Recall@50 |
|---------|---------|---------|---------|---------|
| Random Baseline | 0.100 | 0.049 | 0.033 | 0.500 |
| Word2Vec | 0.520 | 0.316 | 0.253 | 0.866 |
| GRU4Rec | 0.363 | 0.207 | 0.160 | 0.818 |
| **LSTM + Attention** | **0.525** | **0.324** | **0.262** | **0.845** |

---

## 🌐 Deployment

### Backend

FastAPI serves model predictions through REST endpoints.

### Frontend

Streamlit provides:

- Movie timeline creation
- Recommendation generation
- Attention visualization
- Interactive model exploration

### Docker

Run the application in a containerized environment:

```bash
docker build -t seq-rec .
docker run -p 10000:10000 seq-rec
```

---

## 🧪 Example Use Cases

### Psychological Thriller Sequence

```text
Se7en
→ The Usual Suspects
→ The Silence of the Lambs
→ Pulp Fiction
→ Fargo
→ L.A. Confidential
```

Expected recommendations:

- Fight Club
- Reservoir Dogs
- GoodFellas

### Animation Sequence

```text
Toy Story
→ Aladdin
→ The Lion King
→ Beauty and the Beast
→ A Bug's Life
```

Expected recommendations:

- Toy Story 2
- Mulan
- Tarzan

---

## 📊 Explainable AI Layer

The attention module exposes the contribution of each movie in the user's viewing history.

Benefits:

- Transparent recommendations
- Easier debugging
- Improved stakeholder trust
- Better model interpretability

Users can directly inspect which previous movies influenced each recommendation.

---

## 🛠️ Tech Stack

- Python
- PyTorch
- FastAPI
- Streamlit
- Docker
- Pandas
- NumPy
- Gensim

---




