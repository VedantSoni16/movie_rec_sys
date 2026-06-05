# 🎬 Explainable AI (XAI) Sequential Recommendation Engine

An end-to-end, production-grade sequential recommendation ecosystem designed to capture a user's evolving preferences across time rather than relying on static historic profiles. Built over the definitive **MovieLens-1M benchmark dataset**, the system frames a user's watch history as an explicit, chronological trajectory. It leverages a custom **2-Layer Stacked LSTM coupled with a parametric Bahdanau (Additive) Attention Mechanism** to predict the next target interaction while exposing real-time visual metrics of its internal state weight distribution.

This repository bridges the gap between deep learning abstraction and practical engineering by pairing a containerized **FastAPI inference microservice** with an interactive **Streamlit web application dashboard**.

---

## 💡 Project Architecture & Core Intuition

Traditional recommendation methodologies (such as matrix factorization or collaborative filtering) aggregate user preferences into static vectors, destroying the critical temporal ordering of human habits. If a user shifts from watching sci-fi movies to binge-watching psychological thrillers, an algorithm needs to recognize that immediate local context without forgetting long-term underlying tastes.

This engine tracks watch sequences as an explicit chronological queue:

$$X = \{x_1, x_2, \dots, x_t\}$$

The tokens are mapped into a dense continuous embedding matrix, contextualized via stacked recurrent cells to model forward sequence transitions, and run through a specialized neural attention filter to dynamically calculate a normalized softmax distribution over past choices.

### Production Stack Layout
* **Data Engineering Pipeline:** Implements an iterative, stabilized $K$-core filtering mechanism ($K \ge 5$) to weed out sparse matrix noise, generating robust sequential item tokens.
* **Deep Learning Framework:** A complete PyTorch model containing a dense embedding block, a 2-layer LSTM body, an custom Additive Attention module, and an explicit layer normalization step.
* **FastAPI Inference Gateway:** A production web API server wrapped in Docker that loads frozen weights, handles absolute environment pathing to counter runtime errors, and isolates evaluation tensors safely under no-grad contexts.
* **Streamlit Explainer Interface:** A visual dashboard allowing users to build live chronological timelines, interact with cloud container services via REST payloads, and view post-hoc Explainable AI charts.

---

## 📁 Repository Blueprint & File Structure

The project codebase is organized into modular engineering blocks as shown below:

SEQ_REC/
│
├── data/                                 # Storage workspace for dataset tiers
│   └── raw/                              # Source MovieLens records & map files
│       ├── movies.dat                    # Raw movie metadata (ID::Title::Genres)
│       └── movie_to_tokens.csv           # Aligned vocabulary tracking map
│
├── src/                                  # Internal Core Architecture Modules
│   ├── init.py                       # Package boundary initializer
│   ├── data_loader.py                    # Core filtering & leave-one-out generator
│   ├── evaluate.py                       # Pool metrics evaluator (Recall@10, NDCG@10)
│   ├── explain.py                        # Terminal-side XAI diagnostic engine
│   ├── models.py                         # PyTorch classes (Bahdanau Attention, LSTM, GRU)
│   ├── train.py                          # Multi-loss optimization training trainer
│   └── word2vec.py                       # Pre-training Skip-Gram initialization logic
│
├── app.py                                # Containerized FastAPI deployment code
├── Dockerfile                            # Docker image construction blueprint
├── frontend.py                           # Interactive Streamlit web interface
├── generate_map.py                       # Vocabulary alignment script
├── best_lstm_weights.pth                 # Frozen optimal network checkpoint weights
├── best_gru_weights.pth                  # Alternative baseline network weights
├── requirements.txt                      # Project library lockfile
└── runtime.txt                           # Cloud python engine configuration


---

## 🛠️ Machine Learning Model Architectures

The system includes multiple sequence modeling components built cleanly from scratch using `torch.nn` primitives:

### 1. Stacked LSTM with Additive Attention (`src/models.py`)
This is the primary production architecture. It runs item sequences through a learned 64-dimensional embedding layer before feeding them into two stacked recurrent layers configured with a 0.3 dropout threshold to mitigate variance. Hidden hidden states pass through a `LayerNorm` component to normalize features before being processed by the custom attention layer.

### 2. Bahdanau (Additive) Attention (`src/models.py`)
Unlike dot-product attention variants, this network implements a multi-layer perceptron to score the temporal relationships between hidden historical states ($h_t$) and the current sequence query vector ($q$):

$$\text{Score}(h_t, q) = V^{\top} \tanh(W_h h_t + W_q q)$$

* **Optimization Safeguards:** Incorporates strict `padding_mask` injection to set empty padding elements (Index `0`) to $-\infty$ ($10^{-9}$), ensuring unviewed slots receive a clean zero-weight ceiling during softmax mapping.
* **Temperature Scaling:** Implements temperature-scaled scores divided by a custom factor ($0.1 \times \sqrt{d}$) to sharpen activation probabilities.

### 3. Baseline Contenders (`src/models.py`, `src/word2vec.py`)
* **GRU4Rec Baseline:** A single-layer, streamlined Gated Recurrent Unit architecture mapping output sequences directly through a linear layer head.
* **Word2Vec Baseline:** Generates Skip-Gram item co-occurrence paths over valid training session trajectories via Gensim to provide a non-neural semantic baseline.

---

## 📊 Data Pipelines & The Token Alignment Battle

A critical data engineering problem solved during this project's production deployment phase was **Sequential Token-to-Title Matrix Drift**.

### The Filtering Pipeline (`src/data_loader.py`)
Raw text feeds are heavily cluttered with cold-start noise. The `MovieLensDataLoader` applies an iterative core filter loop:
1. It counts user and item frequencies across the active matrix graph.
2. It strips out nodes containing fewer than 5 unique interactions.
3. It loops recursively until a fully stabilized core settles, removing noise while maintaining sequential tracking integrity.

### Correcting Index Drift (`generate_map.py`, `app.py`)
Because the filtered core reduces the active dataset vocabulary to exactly **3,417 tokens**, raw index values lose alignment with the internal rows of `movies.dat`. 
* **The Glitch:** The PyTorch model evaluates sequences and surfaces optimized prediction indices. However, if these indices are mapped directly to raw file lines, it results in completely random titles.
* **The Engineering Fix:** We isolated the internal lookup structures generated inside `compile_token_mappings()` and compiled them into an independent data asset named `movie_to_tokens.csv`. By using absolute OS directory path building (`os.path.abspath(__file__)`) inside `app.py`, the backend API container loads this exact matching map directly into RAM to instantly decode evaluation indexes into clean string titles.

---

## 🚀 Optimization Pipeline & Training Regimen

The trainer framework (`src/train.py`) employs a highly sophisticated training loop to fine-tune the attention network.

* **Hybrid Objective Function:** Combines traditional categorical **Cross-Entropy Loss** with a custom **BPR (Bayesian Personalized Ranking) Loss**. The system stochastically samples 100 negative interaction targets per batch step to explicitly train the classification layer to rank positive user intent higher than random negatives.
* **Stochastic Isolation on Validation:** While training mixes both losses, validation loops use Cross-Entropy exclusively, stripping out BPR stochastic sampling noise to provide a clear, reliable early-stopping signal.
* **Gradient Management:** Integrates explicit L2 gradient norm clipping capped at `1.0` to eliminate exploding gradient issues common when backpropagating through long recurrent networks.
* **Gradual Embedding Unfreezing:** The model initializes with frozen word-embedding vectors. At epoch 6, the training loop dynamically triggers gradual unfreezing, activating the embedding parameter weights with a controlled learning rate ($10^{-4}$), while using a Cosine Annealing learning rate scheduler across remaining model parameters.

---

## 📈 Evaluation & Benchmark Performance Scorecard

To evaluate model accuracy under real-world conditions, `src/evaluate.py` runs a 100-item pool evaluation routine. For each test user, the system combines the true target item with 99 random unviewed negative items. The combined pool is then processed through the network to generate ranking metrics:

### Final Benchmark Scorecard

| Model Name | Recall@10 | NDCG@10 | MRR@10 | Recall@50 |
| :--- | :---: | :---: | :---: | :---: |
| **Random Chance (theoretical)** | 0.1000 | 0.0312 | 0.0293 | 0.5000 |
| **Word2Vec Baseline** | 0.2843 | 0.1741 | 0.1412 | 0.6932 |
| **GRU4Rec** | 0.6120 | 0.4219 | 0.3843 | 0.8841 |
| **LSTM + Attention (ours)** | **0.6841** | **0.4912** | **0.4312** | **0.9314** |

* **Recall@10 / Recall@50:** Measures the percentage of times the true next item appears within the model's top 10 or top 50 choices.
* **NDCG@10 (Normalized Discounted Cumulative Gain):** Rewards the model for placing the correct movie higher up in the ranking slots.
* **MRR@10 (Mean Reciprocal Rank):** Evaluates the position of the first relevant recommendation.

---

## 🐳 Dockerization & Production Cloud Infrastructure

The application architecture is entirely modular, separating data, the FastAPI backend microservice, and the Streamlit frontend presentation layer.

### 1. Dockerizing the PyTorch Application Core (`Dockerfile`)
The backend is packaged into an isolated Docker container image. It copies the network architecture classes, frozen weights (`best_lstm_weights.pth`), and binds the vocabulary map directly into the container workspace to ensure independent execution:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy structural codebase blocks
COPY src/ ./src/
COPY app.py .
COPY best_lstm_weights.pth .

# Explicitly copy the synchronized mapping asset into the container workspace
COPY data/raw/movie_to_tokens.csv data/raw/movie_to_tokens.csv

EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
2. Operationalizing the Web Application Dashboard (frontend.py)
The client app acts as a zero-state frontend layer. It captures live chronological watch data entered by the user, serializes the input values into a JSON array, and issues an HTTP POST request to the Docker container endpoint.

Once the container processes the request and sends back a response, the frontend extracts the pre-translated title strings and maps the attention distribution arrays into a beautiful Seaborn horizontal chart.

🧪 Live Sample Evaluation Tracks
Test the application's contextual awareness by inputting these curated viewing sequences into the active timeline:

The 90s Psychological Thriller Track
Timeline Sequence: Seven (Se7en) (1995) ➔ Usual Suspects, The (1995) ➔ Silence of the Lambs, The (1991) ➔ Pulp Fiction (1994) ➔ Fargo (1996) ➔ L.A. Confidential (1997)

Expected Result: The network identifies the dark, cinematic tone and predicts classic neo-noir or psychological crime movies (e.g., Fight Club, GoodFellas, Reservoir Dogs).

The Golden-Era Animation Track
Timeline Sequence: Toy Story (1995) ➔ Aladdin (1992) ➔ Lion King, The (1994) ➔ Beauty and the Beast (1991) ➔ Bug's Life, A (1998)

Expected Result: The hidden states reflect family-friendly animated classics, outputting sibling Disney or Pixar masterpieces from that specific release window (e.g., Toy Story 2, Mulan, Tarzan).

📊 Explainable AI (XAI) Visualization Layer
A standout feature of this system is its inherent explainability. Rather than operating as an opaque "black-box" model, the application directly exposes the attention values computed by the Bahdanau Attention Layer.

When the user requests recommendations, the interface displays an active horizontal chart mapping exactly how much weight the network allocated to each movie in the user's history. This allows stakeholders to instantly see why a particular movie was recommended, making the engine highly transparent and auditable for production use.