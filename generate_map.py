# generate_map.py
import os
import pandas as pd
from src.data_loader import MovieLensDataLoader

print("🔄 Initializing system alignment map extractor...")

# Define paths pointing to your repository data files
RATINGS_PATH = "data/raw/ratings.dat"
MOVIES_PATH = "data/raw/movies.dat"
OUTPUT_PATH = "data/raw/movie_to_tokens.csv"

if not os.path.exists(RATINGS_PATH) or not os.path.exists(MOVIES_PATH):
    print("❌ Critical Error: Could not locate files in data/raw/. Please check paths!")
    exit()

try:
    # 1. Initialize your actual production data loader pipeline
    loader = MovieLensDataLoader(min_interactions=5, max_seq_len=50)
    
    # 2. Extract and filter data using your data loader logic
    ratings_df, movies_df = loader.load_raw_data(RATINGS_PATH, MOVIES_PATH)
    filtered_df = loader.apply_core_filtering(ratings_df)
    
    # 3. Generate your model's exact dynamic internal token map
    movie_to_token = loader.compile_token_mappings(filtered_df, movies_df)
    
    # 4. Save your loader's internal titles list directly as your frontend map!
    aligned_list = []
    for token_id, title in loader.item_titles.items():
        aligned_list.append({"token_id": token_id, "title": title})
        
    df_out = pd.DataFrame(aligned_list)
    df_out.to_csv(OUTPUT_PATH, index=False)
    
    print("\n" + "="*50)
    print(f"🎯 SUCCESS! Aligned vocabulary written to: {OUTPUT_PATH}")
    print(f"Total tracking tokens mapped: {len(df_out)}")
    print("="*50)

except Exception as e:
    print(f"❌ Map translation failed: {str(e)}")