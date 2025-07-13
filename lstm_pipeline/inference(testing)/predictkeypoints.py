import os
import argparse
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# === Load Environment Variables ===
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "sentence2keypoints_upper_full_final.h5")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
DEFAULT_OUTPUT_PATH = os.getenv("DEFAULT_OUTPUT_PATH", "predicted_keypoints_upper_full.npy")
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 32))
KEYPOINTS = int(os.getenv("KEYPOINTS", 529))
DIM_PER_POINT = int(os.getenv("DIM_PER_POINT", 2))

# === Load Models ===
print("🧠 Loading sentence embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

print(f"📦 Loading trained keypoint prediction model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# === Predict Function ===
def predict_keypoints(sentence: str):
    print(f"📝 Input sentence: {sentence}")
    
    embedding = embedder.encode([sentence])  # (1, 384)
    prediction = model.predict(embedding, verbose=0)  # (1, 32, 1058)

    keypoint_seq = prediction.reshape(SEQUENCE_LENGTH, KEYPOINTS, DIM_PER_POINT)  # (32, 529, 2)
    print(f"📐 Output keypoints shape: {keypoint_seq.shape}")
    return keypoint_seq

# === CLI Entrypoint ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sign language keypoints from a sentence using a trained LSTM model.")
    parser.add_argument("--text", type=str, required=True, help="Sentence to generate keypoints for")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save output .npy file")
    args = parser.parse_args()

    # Predict and save
    keypoints = predict_keypoints(args.text)
    np.save(args.output, keypoints)
    print(f"✅ Keypoints saved to {args.output}")
