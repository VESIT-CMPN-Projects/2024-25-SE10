import os
import pickle
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from minisom import MiniSom
import os
import re
import time
import io
import numpy as np
import torch
from moviepy import *
from sentence_transformers import SentenceTransformer, util
import spacy
import faiss
from minisom import MiniSom
from rapidfuzz import fuzz


# --- Configuration ---
LOCAL_VIDEO_FOLDER = "./tp/videos"
EMBEDDINGS_FILE = "./tp/video_embeddings.pkl"
FAISS_INDEX_FILE = "./tp/faiss_index.bin"
META_FILE = "./tp/meta.pkl"
SOM_FILE = "./tp/som.pkl"

nlp = spacy.load("en_core_web_sm")
# Initialize SentenceTransformer
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def save_pickle(obj, filename):
    """Utility to save objects as pickle files."""
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    """Utility to load pickle objects if they exist."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None


def build_or_load_video_embeddings(video_folder):
    """Builds video embeddings or loads from saved files."""
    if os.path.exists(EMBEDDINGS_FILE):
        print("Loading cached video embeddings...")
        return load_pickle(EMBEDDINGS_FILE)

    video_embeddings = {}
    flat_data = []
    for filename in os.listdir(video_folder):
        if filename.endswith(".mp4"):
            file_path = os.path.join(video_folder, filename)
            base_name = os.path.splitext(filename)[0].lower()
            keywords = base_name.replace('_', ' ').replace('-', ' ').split()
            embeddings = sbert_model.encode(keywords, convert_to_tensor=True)
            video_embeddings[file_path] = list(zip(keywords, embeddings))
            for kw, emb in video_embeddings[file_path]:
                flat_data.append((file_path, kw, emb))

    save_pickle((video_embeddings, flat_data), EMBEDDINGS_FILE)
    return video_embeddings, flat_data


def build_or_load_faiss_index(flat_data):
    """Builds FAISS index or loads from saved files."""
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(META_FILE):
        print("Loading cached FAISS index...")
        index = faiss.read_index(FAISS_INDEX_FILE)
        meta = load_pickle(META_FILE)
        return index, meta

    if not flat_data:
        raise ValueError("No video keyword data found.")

    dim = flat_data[0][2].shape[0]
    embeddings_np = np.vstack([emb.cpu().numpy() for (_, _, emb) in flat_data]).astype('float32')
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)

    meta = [(video_file, kw) for (video_file, kw, _) in flat_data]

    faiss.write_index(index, FAISS_INDEX_FILE)
    save_pickle(meta, META_FILE)
    return index, meta


def train_or_load_som(flat_data, som_x=10, som_y=10):
    """Trains a SOM or loads from a saved file."""
    if os.path.exists(SOM_FILE):
        print("Loading cached SOM...")
        return load_pickle(SOM_FILE)

    dim = flat_data[0][2].shape[0]
    embeddings_np = np.vstack([emb.cpu().numpy() for (_, _, emb) in flat_data])
    som = MiniSom(x=som_x, y=som_y, input_len=dim, sigma=1.0, learning_rate=0.5)
    som.random_weights_init(embeddings_np)
    print("Training SOM on video keyword embeddings...")
    som.train_random(embeddings_np, num_iteration=1000)

    save_pickle(som, SOM_FILE)
    return som


def generate_ngrams(words, n):
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]


def process_sentence(sentence, video_embeddings, faiss_index, meta, som,
                     similarity_threshold=0.7, k=5, fuzzy_threshold=90):
    doc = nlp(sentence)
    tokens = [token.text.lower().strip() for token in doc if not token.is_punct]
    n_tokens = len(tokens)
    best_matches = {}  # Store the best matching video for each start index

    print(f"Processing sentence: {sentence}")

    for n in range(1, n_tokens + 1):
        print(f"\nTrying {n}-word phrases...")
        ngrams = generate_ngrams(tokens, n)
        for i, phrase in enumerate(ngrams):
            start_index = i
            print(f"  Checking phrase: '{phrase}' (start index: {start_index})")
            best_score = -1
            best_match_video = None

            for video_file, keywords in video_embeddings.items():
                for kw, _ in keywords:
                    score = fuzz.ratio(phrase, kw)
                    if score >= fuzzy_threshold and score > best_score:
                        best_score = score
                        best_match_video = video_file

            if best_match_video:
                print(f"    Best match for '{phrase}': '{os.path.basename(best_match_video)}' (Score: {best_score})")
                best_matches[start_index] = best_match_video

    # Order the best matches based on their starting index
    ordered_matches = sorted(best_matches.items(), key=lambda item: item[0])

    # Extract the ordered list of unique video paths
    ordered_videos = []
    seen_videos = set()
    for index, video_path in ordered_matches:
        if video_path not in seen_videos:
            ordered_videos.append(video_path)
            seen_videos.add(video_path)

    print("Final selected videos (in order):", [os.path.basename(v) for v in ordered_videos])
    return ordered_videos

def update_embeddings_if_needed(video_embeddings, video_folder):
    """
    Check for new .mp4 files in video_folder not present in video_embeddings.
    If new files exist, update video_embeddings and rebuild flat_data.
    Returns (updated_video_embeddings, updated_flat_data) if new files found,
    otherwise (video_embeddings, None).
    """
    # Get full paths of current video files
    current_files = set(os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".mp4"))
    cached_files = set(video_embeddings.keys())
    new_files = current_files - cached_files

    if new_files:
        print("New video files detected. Updating embeddings...")
        for file_path in new_files:
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0].lower()
            keywords = base_name.replace('_', ' ').replace('-', ' ').split()
            embeddings = sbert_model.encode(keywords, convert_to_tensor=True)
            video_embeddings[file_path] = list(zip(keywords, embeddings))
        # Rebuild flat_data from updated video_embeddings
        flat_data = []
        for video_file, keywords in video_embeddings.items():
            for kw, emb in keywords:
                flat_data.append((video_file, kw, emb))
        # Save updated embeddings to cache
        save_pickle((video_embeddings, flat_data), EMBEDDINGS_FILE)
        return video_embeddings, flat_data
    else:
        return video_embeddings, None



def stitch_videos(video_list, output_filename="output.mp4", speedup=1.5):
    """
    Concatenates a list of video files into one final video and speeds it up by the given factor.
    """
    if not video_list:
        print("No videos to stitch together.")
        return
    clips = []
    for video_path in video_list:
        print(f"Loading video clip: {video_path}")
        clip = VideoFileClip(video_path)
        clips.append(clip)
    print("Concatenating videos...")
    final_clip = concatenate_videoclips(clips)
    final_clip = final_clip.with_speed_scaled(speedup)
    print(f"Writing final video to {output_filename} ...")
    final_clip.write_videofile(output_filename, codec="libx264")
    for clip in clips:
        clip.close()
    final_clip.close()

def getInit(sentence):
    # Load or build the video embeddings
    video_embeddings, flat_data = build_or_load_video_embeddings(LOCAL_VIDEO_FOLDER)

    # Check for new videos and update embeddings if necessary.
    video_embeddings, updated_flat = update_embeddings_if_needed(video_embeddings, LOCAL_VIDEO_FOLDER)
    # Use the updated flat_data if new files were found.
    if updated_flat is not None:
        flat_data = updated_flat

    # Rebuild FAISS index and meta from the (possibly updated) flat_data.
    faiss_index, meta = build_or_load_faiss_index(flat_data)
    # Retrain or load SOM on the updated flat_data.
    som = train_or_load_som(flat_data)

    # Process the sentence to get selected video files.
    selected_videos = process_sentence(sentence, video_embeddings, faiss_index, meta, som, similarity_threshold=0.7, k=5, fuzzy_threshold=90)

    if not selected_videos:
        print("No matching sign language videos found for the given sentence.")
        return
    else:
        stitch_videos(selected_videos, output_filename="output.mp4", speedup=1.5)
        return 1

# --- Initialization (Run Once) ---
if __name__ == "__main__":

    __all__ = ["getInit"]

    print("Initialization complete.")