# import os
# import numpy as np
# import torch
# from moviepy import *
# from sentence_transformers import SentenceTransformer, util
# import spacy
# import faiss
# from minisom import MiniSom
# from rapidfuzz import fuzz

# # Initialize spaCy and SentenceTransformer.
# nlp = spacy.load("en_core_web_sm")
# sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# def build_video_embeddings(video_folder):
#     """
#     Scans the video folder and builds a dictionary mapping each video file
#     to a list of tuples: (keyword, embedding). Also flattens all data into a list.
#     """
#     video_embeddings = {}  # Per-video dictionary.
#     flat_data = []         # List of tuples: (video_file, keyword, embedding)
#     for filename in os.listdir(video_folder):
#         if filename.lower().endswith(".mp4"):
#             file_path = os.path.join(video_folder, filename)
#             base_name = os.path.splitext(filename)[0].lower()
#             # Extract keywords by replacing underscores/dashes with spaces.
#             keywords = base_name.replace('_', ' ').replace('-', ' ').split()
#             print(f"File: {filename}, Keywords: {keywords}")  # Debug line
#             embeddings = sbert_model.encode(keywords, convert_to_tensor=True)
#             video_embeddings[file_path] = list(zip(keywords, embeddings))
#             for kw, emb in video_embeddings[file_path]:
#                 flat_data.append((file_path, kw, emb))
#     return video_embeddings, flat_data

# def build_faiss_index(flat_data):
#     """
#     Builds a FAISS index from the flattened video keyword data.
#     Also returns a metadata list with (video_file, keyword) for each embedding.
#     """
#     if not flat_data:
#         raise ValueError("No video keyword data found.")
#     dim = flat_data[0][2].shape[0]
#     embeddings_list = [emb.cpu().numpy() for (_, _, emb) in flat_data]
#     embeddings_np = np.vstack(embeddings_list).astype('float32')
    
#     # Create a FAISS index (using L2 distance).
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings_np)
    
#     # Create metadata list for each embedding.
#     meta = [(video_file, kw) for (video_file, kw, _) in flat_data]
#     return index, meta

# def train_som(flat_data, som_x=10, som_y=10):
#     """
#     Trains a SOM on the flattened video keyword embeddings.
#     Returns the trained SOM.
#     """
#     dim = flat_data[0][2].shape[0]
#     embeddings_list = [emb.cpu().numpy() for (_, _, emb) in flat_data]
#     embeddings_np = np.vstack(embeddings_list)
#     som = MiniSom(x=som_x, y=som_y, input_len=dim, sigma=1.0, learning_rate=0.5)
#     som.random_weights_init(embeddings_np)
#     print("Training SOM on video keyword embeddings...")
#     som.train_random(embeddings_np, num_iteration=1000)
#     return som

# def convert_num_to_word(token_text):
#     """Converts numeric strings to their word equivalent."""
#     mapping = {
#         "1": "one",
#         "2": "two",
#         "3": "three",
#         "4": "four",
#         "5": "five",
#         "6": "six",
#         "7": "seven",
#         "8": "eight",
#         "9": "nine",
#         "10": "ten"
#     }
#     return mapping.get(token_text, token_text)

# def process_sentence(sentence, video_embeddings, faiss_index, meta, som, similarity_threshold=0.7, k=5, fuzzy_threshold=90):
#     """
#     Tokenizes the sentence with spaCy, applies number conversion, and for each token:
#       1. Checks for an exact match (using fuzzy matching) among video keywords.
#       2. Otherwise, uses SentenceTransformer embeddings, FAISS search, and SOM filtering.
#     Returns a list of video file paths corresponding to the best match for each token.
#     """
#     doc = nlp(sentence)
#     selected_videos = []

#     for token in doc:
#         # Skip punctuation.
#         if token.is_punct:
#             continue
#         # Skip stop words unless they are numeric tokens.
#         if token.is_stop and not token.like_num:
#             continue

#         token_text = token.text.lower().strip()
#         print(f"Processing token: '{token_text}'")  # Debug
#         if token.like_num and token_text.isdigit():
#             token_text = convert_num_to_word(token_text)

#         # First, try to find an exact match using fuzzy string matching.
#         exact_match = None
#         for video_file, keywords in video_embeddings.items():
#             for kw, emb in keywords:
#                 # If the fuzzy match ratio is above the threshold, treat as a match.
#                 if fuzz.ratio(token_text, kw) >= fuzzy_threshold:
#                     exact_match = video_file
#                     break
#             if exact_match:
#                 break

#         if exact_match:
#             print(f"Exact match: Selected video '{os.path.basename(exact_match)}' for token '{token.text}'")
#             selected_videos.append(exact_match)
#             continue

#         # Compute embedding for the token.
#         token_embedding = sbert_model.encode(token_text, convert_to_tensor=True)
#         best_sim = 0
#         best_video = None

#         # Use FAISS to search for top-k nearest keyword embeddings.
#         token_np = token_embedding.cpu().numpy().astype('float32').reshape(1, -1)
#         D, I = faiss_index.search(token_np, k)
#         token_bmu = som.winner(token_embedding.cpu().numpy())
#         # Iterate over candidates.
#         for dist, idx in zip(D[0], I[0]):
#             # Convert L2 distance to a similarity score.
#             sim = 1 / (1 + dist)
#             video_file, keyword = meta[idx]
#             # For debugging, get candidate BMU.
#             candidate_emb = faiss_index.reconstruct(idx)
#             candidate_bmu = som.winner(candidate_emb)
#             print(f"Candidate '{keyword}' in video '{os.path.basename(video_file)}' has similarity {sim:.2f}, token BMU: {token_bmu}, candidate BMU: {candidate_bmu}")
#             if sim > best_sim:
#                 best_sim = sim
#                 best_video = video_file

#         if best_video and best_sim >= similarity_threshold:
#             print(f"Selected video '{os.path.basename(best_video)}' for token '{token.text}' (similarity: {best_sim:.2f})")
#             selected_videos.append(best_video)
#         else:
#             print(f"No matching video found for token '{token.text}' (best similarity: {best_sim:.2f})")
    
#     return selected_videos

# def stitch_videos(video_list, output_filename="output.mp4", speedup=1.75):
#     """
#     Concatenates a list of video files into one final video and speeds it up by the given factor.
#     """
#     if not video_list:
#         print("No videos to stitch together.")
#         return
#     clips = []
#     for video_path in video_list:
#         print(f"Loading video clip: {video_path}")
#         clip = VideoFileClip(video_path)
#         clips.append(clip)
#     print("Concatenating videos...")
#     final_clip = concatenate_videoclips(clips)
#     # Apply speed-up factor using the imported speedx.
#     final_clip = final_clip.with_speed_scaled(speedup)
#     print(f"Writing final video to {output_filename} ...")
#     final_clip.write_videofile(output_filename, codec="libx264")
#     for clip in clips:
#         clip.close()
#     final_clip.close()

# if __name__ == "__main__":
#     sentence = input("Enter a sentence: ")
#     video_folder = "videos"  # Change to your actual video folder path.
    
#     # Precompute embeddings for all video keywords.
#     video_embeddings, flat_data = build_video_embeddings(video_folder)
#     # Build FAISS index and metadata.
#     faiss_index, meta = build_faiss_index(flat_data)
#     # Train a SOM on the keyword embeddings.
#     som = train_som(flat_data, som_x=10, som_y=10)
    
#     # Process the sentence to find the best matching video files.
#     selected_videos = process_sentence(sentence, video_embeddings, faiss_index, meta, som, similarity_threshold=0.7, k=5, fuzzy_threshold=90)
    
#     if not selected_videos:
#         print("No matching sign language videos found for the given sentence.")
#     else:
#         stitch_videos(selected_videos, output_filename="output2.mp4", speedup=1.5)
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
from video_indexing import *


# --- Configuration for Public Drive Folder ---
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1U-Pr4r1-cupgNOOq9NH_uTsQnPSVEKco"
LOCAL_VIDEO_FOLDER = "videos"

# --- Initialize spaCy and SentenceTransformer ---
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Functions for Local Video Processing ---

def build_video_embeddings(video_folder):
    """
    Scans the video folder and builds a dictionary mapping each video file
    to a list of tuples: (keyword, embedding). Also flattens all data into a list.
    """
    video_embeddings = {}  # Per-video dictionary.
    flat_data = []         # List of tuples: (video_file, keyword, embedding)
    for filename in os.listdir(video_folder):
        if filename.lower().endswith(".mp4"):
            file_path = os.path.join(video_folder, filename)
            base_name = os.path.splitext(filename)[0].lower()
            # Extract keywords by replacing underscores/dashes with spaces.
            keywords = base_name.replace('_', ' ').replace('-', ' ').split()
            print(f"Local file: {filename}, Keywords: {keywords}")
            embeddings = sbert_model.encode(keywords, convert_to_tensor=True)
            video_embeddings[file_path] = list(zip(keywords, embeddings))
            for kw, emb in video_embeddings[file_path]:
                flat_data.append((file_path, kw, emb))
    return video_embeddings, flat_data

def build_faiss_index(flat_data):
    """
    Builds a FAISS index from the flattened video keyword data.
    Also returns a metadata list with (video_file, keyword) for each embedding.
    """
    if not flat_data:
        raise ValueError("No video keyword data found.")
    dim = flat_data[0][2].shape[0]
    embeddings_list = [emb.cpu().numpy() for (_, _, emb) in flat_data]
    embeddings_np = np.vstack(embeddings_list).astype('float32')
    
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)
    
    meta = [(video_file, kw) for (video_file, kw, _) in flat_data]
    return index, meta

def train_som(flat_data, som_x=10, som_y=10):
    """
    Trains a SOM on the flattened video keyword embeddings.
    Returns the trained SOM.
    """
    dim = flat_data[0][2].shape[0]
    embeddings_list = [emb.cpu().numpy() for (_, _, emb) in flat_data]
    embeddings_np = np.vstack(embeddings_list)
    som = MiniSom(x=som_x, y=som_y, input_len=dim, sigma=1.0, learning_rate=0.5)
    som.random_weights_init(embeddings_np)
    print("Training SOM on video keyword embeddings...")
    som.train_random(embeddings_np, num_iteration=1000)
    return som

def convert_num_to_word(token_text):
    """Converts numeric strings to their word equivalent."""
    mapping = {
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten"
    }
    return mapping.get(token_text, token_text)

def process_sentence(sentence, video_embeddings, faiss_index, meta, som, similarity_threshold=0.7, k=5, fuzzy_threshold=90):
    """
    Tokenizes the sentence with spaCy, applies number conversion, and for each token:
      1. Checks for an exact match (using fuzzy matching) among video keywords.
      2. Otherwise, uses SentenceTransformer embeddings, FAISS search, and SOM filtering.
    Returns a list of video file paths corresponding to the best match for each token.
    """
    doc = nlp(sentence)
    selected_videos = []

    for token in doc:
        if token.is_punct:
            continue
        if token.is_stop and not token.like_num:
            continue

        token_text = token.text.lower().strip()
        print(f"Processing token: '{token_text}'")
        if token.like_num and token_text.isdigit():
            token_text = convert_num_to_word(token_text)

        # First, try to find an exact match using fuzzy matching in local files.
        exact_match = None
        for video_file, keywords in video_embeddings.items():
            for kw, emb in keywords:
                if fuzz.ratio(token_text, kw) >= fuzzy_threshold:
                    exact_match = video_file
                    break
            if exact_match:
                break

        # If no local match, attempt to download the specific file from Drive.
        if not exact_match:
            exact_match = ensure_video_exists(token_text, LOCAL_VIDEO_FOLDER, DRIVE_FOLDER_URL)
            if exact_match:
                print(f"Updating embeddings with downloaded file: {exact_match}")
                new_embeds, new_flat = build_video_embeddings(LOCAL_VIDEO_FOLDER)
                video_embeddings.update(new_embeds)
                meta.extend([(f, kw) for (f, kw, _) in new_flat if f == exact_match])
                _, flat_data = build_video_embeddings(LOCAL_VIDEO_FOLDER)
                faiss_index, meta = build_faiss_index(flat_data)

        if exact_match:
            print(f"Exact match: Selected video '{os.path.basename(exact_match)}' for token '{token.text}'")
            selected_videos.append(exact_match)
            continue

        # Otherwise, use FAISS search.
        token_embedding = sbert_model.encode(token_text, convert_to_tensor=True)
        best_sim = 0
        best_video = None
        token_np = token_embedding.cpu().numpy().astype('float32').reshape(1, -1)
        D, I = faiss_index.search(token_np, k)
        token_bmu = som.winner(token_embedding.cpu().numpy())
        for dist, idx in zip(D[0], I[0]):
            sim = 1 / (1 + dist)
            video_file, keyword = meta[idx]
            # Cast idx to int to satisfy FAISS API.
            candidate_emb = faiss_index.reconstruct(int(idx))
            candidate_bmu = som.winner(candidate_emb)
            print(f"Candidate '{keyword}' in video '{os.path.basename(video_file)}' has similarity {sim:.2f}, token BMU: {token_bmu}, candidate BMU: {candidate_bmu}")
            if sim > best_sim:
                best_sim = sim
                best_video = video_file

        if best_video and best_sim >= similarity_threshold:
            print(f"Selected video '{os.path.basename(best_video)}' for token '{token.text}' (similarity: {best_sim:.2f})")
            selected_videos.append(best_video)
        else:
            print(f"No matching video found for token '{token.text}' (best similarity: {best_sim:.2f})")
    
    return selected_videos

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

# --- Main Pipeline ---

if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    
    # Build local embeddings, index, and SOM.
    video_embeddings, flat_data = build_video_embeddings(LOCAL_VIDEO_FOLDER)
    faiss_index, meta = build_faiss_index(flat_data)
    som = train_som(flat_data, som_x=10, som_y=10)
    
    # Process the sentence.
    selected_videos = process_sentence(sentence, video_embeddings, faiss_index, meta, som, similarity_threshold=0.7, k=5, fuzzy_threshold=90)
    
    if not selected_videos:
        print("No matching sign language videos found for the given sentence.")
    else:
        stitch_videos(selected_videos, output_filename="output.mp4", speedup=1.5)
