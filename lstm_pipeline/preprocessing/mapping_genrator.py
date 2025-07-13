import os
import json
from dotenv import load_dotenv

# === Load from .env ===
load_dotenv()
KEYPOINT_ROOT = os.getenv("KEYPOINT_ROOT")
OUTPUT_JSON = os.getenv("OUTPUT_JSON", "sentence_keypoint_mapping_upper_full.json")

def build_sentence_keypoint_mapping(root_dir):
    mappings = []

    for alpha_folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, alpha_folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".npy"):
                # Sentence text from filename (underscores to spaces)
                sentence = os.path.splitext(file)[0].replace("_", " ").lower()
                npy_path = os.path.join(folder_path, file).replace("\\", "/")

                mappings.append({
                    "text": sentence,
                    "npy": npy_path
                })

    return mappings

# === Main ===
if __name__ == "__main__":
    if not KEYPOINT_ROOT:
        raise ValueError("❌ KEYPOINT_ROOT not set in .env file")

    data = build_sentence_keypoint_mapping(KEYPOINT_ROOT)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Mapping saved to {OUTPUT_JSON}")
    print(f"📦 Total mappings: {len(data)}")
