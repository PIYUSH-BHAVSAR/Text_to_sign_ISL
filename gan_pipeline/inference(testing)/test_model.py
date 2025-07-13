# generate_gif.py - Generate sign language video/gif for a sentence using trained GAN

import os
import numpy as np
import tensorflow as tf
import imageio
import pickle
from dotenv import load_dotenv

# === Load env variables ===
load_dotenv()

GENERATOR_PATH = os.getenv("GENERATOR_PATH", "sign_language_word_generator.h5")
ENCODER_PATH = os.getenv("ENCODER_PATH", "word_label_encoder.pkl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "generated_videos")
FPS = int(os.getenv("GIF_FPS", 5))

IMG_SHAPE = (16, 64, 64, 3)

# === Load models and encoder ===
print("📦 Loading generator model and label encoder...")
generator = tf.keras.models.load_model(GENERATOR_PATH, compile=False)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Generate frames for one word ===
def generate_frames(word):
    word = word.lower().strip()
    if word not in label_encoder.classes_:
        print(f"❌ '{word}' not in vocabulary. Skipping.")
        return None
    
    label = label_encoder.transform([word])
    frames = generator.predict(np.array(label).reshape(1, 1), verbose=0)[0]
    frames = (frames * 255).astype(np.uint8)
    return list(frames)

# === Generate animated GIF from sentence ===
def generate_sentence_gif(text, output_name="output"):
    words = text.strip().lower().split()
    combined_frames = []

    for word in words:
        frames = generate_frames(word)
        if frames:
            combined_frames.extend(frames)
            pause = [np.zeros_like(frames[0])] * 3  # 3-frame pause between words
            combined_frames.extend(pause)

    if not combined_frames:
        print("❌ No valid words found.")
        return

    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in output_name)
    gif_path = os.path.join(OUTPUT_DIR, f"{safe_name}.gif")
    imageio.mimsave(gif_path, combined_frames, fps=FPS)
    print(f"✅ Saved GIF: {gif_path}")

# === Example Usage ===
if __name__ == "__main__":
    sentence = "hi hello"  # Replace with your test sentence
    generate_sentence_gif(sentence, output_name=sentence.replace(" ", "_"))
