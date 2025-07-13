import os
from dotenv import load_dotenv
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# ========== LOAD ENV VARS ========== #
load_dotenv()

VIDEO_ROOT = os.getenv("VIDEO_ROOT")
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT")
TARGET_SIZE = (128, 128)
NUM_FRAMES = 32
MAX_PROCESSES = 6

# ========== SKIP LIST ========== #
SKIP_FOLDERS = {
    # folders already processed
}


# ========== VIDEO PROCESSING FUNCTION ========== #
def extract_and_save(task):
    video_path, class_name = task
    try:
        original_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(OUTPUT_ROOT, class_name)
        output_path = os.path.join(output_folder, f"{original_name}.npy")

        # ✅ Skip if already processed
        if os.path.exists(output_path):
            return True

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Can't open video: {video_path}")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < NUM_FRAMES:
            print(f"⚠️ Skipping short video ({total_frames} frames): {video_path}")
            return False

        frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)

        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.resize(frame, TARGET_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame / 255.0
                frames.append(frame)
        cap.release()

        while len(frames) < NUM_FRAMES:
            frames.append(np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3)))

        frames_np = np.array(frames[:NUM_FRAMES])
        os.makedirs(output_folder, exist_ok=True)
        np.save(output_path, frames_np)
        return True

    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")
        return False

# ========== TASK GATHERING ========== #
def gather_video_tasks(video_root):
    tasks = []
    for class_name in sorted(os.listdir(video_root)):
        if class_name.strip() in SKIP_FOLDERS:
            print(f"⏭ Skipping folder: {class_name}")
            continue

        class_path = os.path.join(video_root, class_name)
        if not os.path.isdir(class_path):
            continue

        for file in sorted(os.listdir(class_path)):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mpg')):
                video_path = os.path.join(class_path, file)
                tasks.append((video_path, class_name))
    return tasks

# ========== MAIN EXECUTION ========== #
if __name__ == "__main__":
    print("🔍 Scanning videos...")
    video_tasks = gather_video_tasks(VIDEO_ROOT)
    print(f"📦 Total videos to process: {len(video_tasks)}")

    with Pool(processes=MAX_PROCESSES) as pool:
        list(tqdm(pool.imap_unordered(extract_and_save, video_tasks), total=len(video_tasks)))

    print("✅ All done. Dataset saved to:", OUTPUT_ROOT)
