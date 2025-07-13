import os
import time
import psutil
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import signal
import sys
from dotenv import load_dotenv

# ========== LOAD .env CONFIG ========== #
load_dotenv()
INPUT_ROOT = os.getenv("INPUT_ROOT")
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT")

# ========== PROCESSING CONFIG ========== #
MAX_CPU_USAGE = 50
CPU_CHECK_INTERVAL = 2
PROCESS_DELAY = 0.02
MAX_PROCESSES = max(1, cpu_count() // 2)
BATCH_SIZE = 100
TEMP_THRESHOLD = 80
COOLING_DELAY = 2
FRAME_COOLDOWN_INTERVAL = 4
FILE_COOLDOWN = 0.2

# ========== MONITORING ========== #
def check_cpu_status():
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        temps = psutil.sensors_temperatures()
        if temps:
            cpu_temp = max(temp.current for temp_list in temps.values() for temp in temp_list)
            if cpu_temp > TEMP_THRESHOLD:
                return True, f"High temperature: {cpu_temp}°C"
        if cpu_percent > MAX_CPU_USAGE:
            return True, f"High CPU usage: {cpu_percent}%"
        return False, "OK"
    except Exception as e:
        return False, f"Monitor error: {e}"

def adaptive_delay():
    overloaded, status = check_cpu_status()
    if overloaded:
        print(f"🔥 {status} - Cooling down...")
        time.sleep(COOLING_DELAY)
        return True
    return False

# ========== HOLISTIC INSTANCE ========== #
mp_holistic = mp.solutions.holistic

def create_holistic_instance():
    return mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# ========== KEYPOINT EXTRACTION ========== #
def extract_upper_body_keypoints(frame, holistic):
    try:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            frame = np.stack([frame] * 3, axis=-1)

        results = holistic.process(frame)

        keypoints = []

        pose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            keypoints += [[pose[i].x, pose[i].y] for i in pose_indices]
        else:
            keypoints += [[0, 0]] * len(pose_indices)

        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                keypoints += [[lm.x, lm.y] for lm in hand.landmark]
            else:
                keypoints += [[0, 0]] * 21

        if results.face_landmarks:
            keypoints += [[lm.x, lm.y] for lm in results.face_landmarks.landmark]
        else:
            keypoints += [[0, 0]] * 468

        return np.array(keypoints)
    except Exception as e:
        print(f"⚠️ Error extracting keypoints: {e}")
        return np.zeros((531, 2))

# ========== PROCESS FILE ========== #
def process_npy_file(task):
    npy_path, word = task
    try:
        holistic = create_holistic_instance()
        data = np.load(npy_path)
        keypoints = []

        for i, frame in enumerate(data):
            time.sleep(PROCESS_DELAY)
            if i % FRAME_COOLDOWN_INTERVAL == 0:
                adaptive_delay()
            kp = extract_upper_body_keypoints(frame, holistic)
            keypoints.append(kp)

        keypoints = np.array(keypoints)
        out_dir = os.path.join(OUTPUT_ROOT, word)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, os.path.basename(npy_path)), keypoints)
        holistic.close()
        time.sleep(FILE_COOLDOWN)
        adaptive_delay()
        return True
    except Exception as e:
        print(f"❌ Error in {npy_path}: {e}")
        return False

def gather_tasks():
    tasks = []
    skipped = 0
    for word in os.listdir(INPUT_ROOT):
        in_folder = os.path.join(INPUT_ROOT, word)
        out_folder = os.path.join(OUTPUT_ROOT, word)
        if not os.path.isdir(in_folder):
            continue
        os.makedirs(out_folder, exist_ok=True)
        for file in os.listdir(in_folder):
            if file.endswith(".npy"):
                in_path = os.path.join(in_folder, file)
                out_path = os.path.join(out_folder, file)
                if os.path.exists(out_path):
                    skipped += 1
                    continue
                tasks.append((in_path, word))
    print(f"⚠️ Skipped {skipped} files already processed.")
    return tasks

def signal_handler(signum, frame):
    print("\n🛑 Keyboard interrupt! Exiting cleanly...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ========== MAIN ========== #
def main():
    print("🚀 Starting full upper-body keypoint extraction...")
    print(f"📊 CPU limit: {MAX_CPU_USAGE}% | Temp limit: {TEMP_THRESHOLD}°C")
    print(f"⚙️ Max processes: {MAX_PROCESSES} | Batch size: {BATCH_SIZE}")

    tasks = gather_tasks()
    print(f"📦 Total files: {len(tasks)}")
    batches = [tasks[i:i + BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)]

    total_processed = 0
    total_success = 0

    for idx, batch in enumerate(batches):
        print(f"\n🔄 Processing batch {idx + 1}/{len(batches)}")
        overloaded, status = check_cpu_status()
        if overloaded:
            print(f"🔥 {status} - Cooling before batch...")
            time.sleep(COOLING_DELAY)

        pool = Pool(processes=MAX_PROCESSES)
        try:
            batch_results = list(tqdm(
                pool.imap_unordered(process_npy_file, batch),
                total=len(batch),
                desc=f"Batch {idx + 1}",
                leave=False
            ))
            batch_success = sum(batch_results)
            total_processed += len(batch)
            total_success += batch_success
            print(f"✅ Batch {idx + 1}: {batch_success}/{len(batch)} processed")
        except KeyboardInterrupt:
            print("\n🛑 Detected interrupt — terminating pool...")
            pool.terminate()
            pool.join()
            sys.exit(0)
        except Exception as e:
            print(f"❌ Batch error: {e}")
        finally:
            pool.close()
            pool.join()

        if idx < len(batches) - 1:
            print("❄️ Cooling between batches...")
            time.sleep(COOLING_DELAY)
            adaptive_delay()

    print(f"\n🎉 Extraction complete!")
    print(f"📊 Files processed: {total_success}/{total_processed}")
    print(f"📈 Success rate: {(total_success / total_processed) * 100:.1f}%")
    print("✅ Keypoints saved to:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
