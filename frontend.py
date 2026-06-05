# frontend.py
import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="XAI Sequential Recommender", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS to style borders, margins, and presentation frames
st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: 700; color: #F63366; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #A0A2A6; margin-bottom: 30px; }
    .section-header { font-size: 22px; font-weight: 600; margin-top: 20px; margin-bottom: 15px; }
    .rec-card { background-color: #1E1E24; padding: 15px; border-radius: 8px; border-left: 4px solid #F63366; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Explainable AI (XAI) Sequential Recommendation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A production-grade inference engine displaying sharp historical attention tracks over the MovieLens-1M benchmark.</div>', unsafe_allow_html=True)
st.markdown("---")


# --- DYNAMIC VOCABULARY DATA LOADER LAYER ---
@st.cache_data
def load_movie_vocabulary():
    try:
        # Read the raw MovieLens .dat file from its exact repository folder pathway
        movies_df = pd.read_csv(
            'data/raw/movies.dat', 
            sep='::', 
            engine='python', 
            names=['movie_id', 'title', 'genres'], 
            encoding='latin-1'
        )
        # Convert directly to a clean dictionary mapping: {ID: "Title (Year)"}
        return pd.Series(movies_df.title.values, index=movies_df.movie_id).to_dict()
    except Exception as e:
        # Robust safety fallback network containing our primary animation cluster
        return {
            1008: "Aladdin (1992)", 
            2926: "Bug's Life, A (1998)", 
            1520: "Antz (1998)",
            1979: "Hunchback of Notre Dame, The (1996)", 
            862: "Hercules (1997)", 
            1069: "Mulan (1998)",
            3412: "Pocahontas (1995)",
            3232: "Tarzan (1999)",
            3078: "Toy Story (1995)",
            341: "Beauty and the Beast (1991)",
            105: "Cinderella (1950)"
        }

# Populate vocabulary spaces dynamically
MOVIE_LOOKUP = load_movie_vocabulary()
title_to_id = {v: k for k, v in MOVIE_LOOKUP.items()}
movie_options = sorted(list(MOVIE_LOOKUP.values()))


# --- SIDEBAR CONFIGURATION: ORDERED CHRONOLOGICAL TIMELINE ---
st.sidebar.markdown("### 🕒 User Viewing History Timeline")
st.sidebar.markdown("Define a strict chronological watch history sequence (oldest to newest):")

# Maintain an explicit list of selections cleanly via Session State variables
if "timeline_history" not in st.session_state:
    # Use animation titles as clean defaults since they exist in both full and fallback modes
    st.session_state.timeline_history = [
        "Aladdin (1992)", 
        "Bug's Life, A (1998)", 
        "Antz (1998)", 
        "Hunchback of Notre Dame, The (1996)", 
        "Hercules (1997)", 
        "Mulan (1998)"
    ]

# Render an explicit, numbered sequential track list for user clarity
updated_history = []
for idx, default_val in enumerate(st.session_state.timeline_history):
    # Fallback to item 0 if a custom entry isn't found in current dictionary space
    start_idx = movie_options.index(default_val) if default_val in movie_options else 0
    
    val_choice = st.sidebar.selectbox(
        f"Step {idx+1:02d} (Watched #{idx+1})",
        options=movie_options,
        index=start_idx,
        key=f"step_select_{idx}"
    )
    updated_history.append(val_choice)

st.session_state.timeline_history = updated_history

# Action controller buttons inside the sidebar layout
st.sidebar.markdown("#### Controls")
col_add, col_rem = st.sidebar.columns(2)
with col_add:
    if st.button("➕ Add Step"):
        st.session_state.timeline_history.append(movie_options[0])
        st.rerun()
with col_rem:
    if st.button("➖ Remove Step") and len(st.session_state.timeline_history) > 1:
        st.session_state.timeline_history.pop()
        st.rerun()

st.sidebar.markdown("---")
trigger_inference = st.sidebar.button("🚀 Generate Recommendations", type="primary", use_container_width=True)


# --- MAIN PAGE INFERENCE CONSOLE INTERACTION LAYER ---
if trigger_inference:
    input_tokens = [title_to_id[title] for title in st.session_state.timeline_history if title in title_to_id]
    
    # Live production URL of your container backend service
    backend_url = "https://movie-rec-sys-3yj4.onrender.com/predict"
    payload = {"movie_history_tokens": input_tokens}
    
    try:
        with st.spinner("Passing user history array down to Dockerized FastAPI module..."):
            response = requests.post(backend_url, json=payload)
            
        if response.status_code == 200:
            data = response.json()
            rec_tokens = data["recommended_tokens"]
            raw_weights = data["attention_weights"]
            
            # Display Clean, Modern Recommendation Cards Section
            st.markdown('<div class="section-header">🔮 Model Next-Item Predictions (Top-5 Rank)</div>', unsafe_allow_html=True)
            cols = st.columns(5)
            for i, token in enumerate(rec_tokens[:5]):
                with cols[i]:
                    title_string = MOVIE_LOOKUP.get(token, f"Movie ID: {token}")
                    st.markdown(f"""
                        <div class="rec-card">
                            <span style="font-size:12px; color:#F63366; font-weight:bold; text-transform:uppercase;">Rank {i+1}</span>
                            <p style="margin:5px 0 0 0; font-size:15px; font-weight:600; color:#FFFFFF;">{title_string}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("---")
            
            # Display High-Contrast Explainable AI Heatmap Block
            st.markdown('<div class="section-header">💡 Post-Hoc Explainable AI (XAI) Attention Metrics</div>', unsafe_allow_html=True)
            st.markdown("The chart below maps exactly how much weight the Bahdanau layer allocated to past items to evaluate the current transition:")
            
            history_depth = len(input_tokens)
            active_weights = raw_weights[-history_depth:]
            
            plot_df = pd.DataFrame({
                "User Viewing History Step (Chronological)": [f"#{i+1}: {t}" for i, t in enumerate(st.session_state.timeline_history)],
                "Attention Allocation Weight (%)": [w * 100 for w in active_weights]
            })
            
            # Set styled dark background palette context for Seaborn
            sns.set_theme(style="darkgrid", rc={
                "axes.facecolor": "#121216", 
                "figure.facecolor": "#0E1117", 
                "text.color": "#FFFFFF", 
                "axes.labelcolor": "#FFFFFF", 
                "xtick.color": "#FFFFFF", 
                "ytick.color": "#FFFFFF"
            })
            
            fig, ax = plt.subplots(figsize=(11, 4))
            sns.barplot(
                x="Attention Allocation Weight (%)", 
                y="User Viewing History Step (Chronological)", 
                data=plot_df, 
                palette="YlOrRd_r", 
                ax=ax
            )
            ax.set_xlim(0, 100)
            ax.set_xlabel("Attention Allocation Weight (%)", fontsize=11, fontweight='bold', labelpad=10)
            ax.set_ylabel("", fontsize=11)
            
            for p in ax.patches:
                width = p.get_width()
                if width > 0.1:
                    ax.text(width + 1.5, p.get_y() + p.get_height()/2, f"{width:.1f}%", va="center", color="#FFFFFF", fontweight="bold", fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
    except requests.exceptions.ConnectionError:
        st.error("Connection Refused. Cloud infrastructure server endpoint failed to respond.")
else:
    st.info("👈 Use the chronological sidebar panel on the left to set up a movie sequence track, then click Generate Recommendations to inspect inference results.")