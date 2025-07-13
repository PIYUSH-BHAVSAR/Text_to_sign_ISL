import os
import json
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# === Load .env configuration ===
load_dotenv()
MAPPING_JSON = os.getenv("MAPPING_JSON", "sentence_keypoint_mapping_upper_full.json")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OUTPUT_PATH = os.getenv("OUTPUT_NPZ", "sentence_keypoint_dataset_upper_full.npz")
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0")) or None  # 0 disables limit

# === LOAD MODEL ===
print("🔤 Loading sentence transformer model...")
model = SentenceTransformer(EMBEDDING_MODEL)

# === LOAD MAPPINGS ===
print(f"📄 Reading mapping from {MAPPING_JSON}")
with open(MAPPING_JSON, "r", encoding="utf-8") as f:
    mappings = json.load(f)

if MAX_SAMPLES:
    mappings = mappings[:MAX_SAMPLES]

embeddings = []
keypoints = []
skipped = 0

print("🔁 Processing each sentence and keypoint file...")

for item in tqdm(mappings):
    sentence = item["text"]
    npy_path = item["npy"]

    try:
        # Generate sentence embedding (384-dim for MiniLM)
        embedding = model.encode(sentence)

        # Load keypoints: Expect shape (N, 529, 2)
        keypoint_seq = np.load(npy_path)

        # Shape check
        if keypoint_seq.ndim != 3 or keypoint_seq.shape[1:] != (529, 2):
            print(f"⚠️ Skipped due to shape mismatch: {npy_path} | Found: {keypoint_seq.shape}")
            skipped += 1
            continue

        embeddings.append(embedding)
        keypoints.append(keypoint_seq)

    except Exception as e:
        print(f"⚠️ Skipped: {sentence} - {e}")
        skipped += 1

# === SAVE FINAL DATASET ===
embeddings = np.array(embeddings)
keypoints = np.array(keypoints)

np.savez_compressed(OUTPUT_PATH, X=embeddings, Y=keypoints)

# === SUMMARY ===
print(f"\n✅ Dataset saved to {OUTPUT_PATH}")
print(f"🧠 Embeddings shape: {embeddings.shape}")
print(f"🎯 Keypoints shape: {keypoints.shape}")
print(f"⚠️ Skipped samples: {skipped}")
