import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter1d
from dotenv import load_dotenv

# === ENV CONFIG ===
load_dotenv()
DEFAULT_FILE = os.getenv("KEYPOINT_FILE", "pred_how_are_you.npy")
DEFAULT_SAVE = os.getenv("SAVE_GIF_PATH", None)
FPS = int(os.getenv("ANIMATION_FPS", 30))

# === PARSE ARGUMENTS ===
parser = argparse.ArgumentParser(description="Animate smoothed keypoints from a .npy file")
parser.add_argument("--file", type=str, default=DEFAULT_FILE, help="Path to keypoints .npy file")
parser.add_argument("--save", type=str, default=DEFAULT_SAVE, help="Optional path to save animation as GIF")
args = parser.parse_args()

# === LOAD KEYPOINT DATA ===
print(f"📂 Loading keypoints from {args.file}")
keypoints = np.load(args.file)  # Shape: (32, 529, 2)

# === SMOOTHING FUNCTION ===
def smooth_keypoints(kps, sigma=1.2):
    smoothed = np.zeros_like(kps)
    for i in range(kps.shape[1]):
        for j in range(kps.shape[2]):
            mask = kps[:, i, j] != 0
            if np.sum(mask) > 1:
                smoothed[:, i, j] = gaussian_filter1d(kps[:, i, j], sigma=sigma)
            else:
                smoothed[:, i, j] = kps[:, i, j]
    return smoothed

keypoints_smooth = smooth_keypoints(keypoints)

# === PLOT SETUP ===
POSE_PAIRS = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (11, 23), (12, 24), (23, 24), (0, 11), (0, 12)]
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
                    (0,17), (17,18), (18,19), (19,20)]

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 12))
ax.set_facecolor("black")

def get_color(part, alpha=0.8):
    return {
        "pose": (0.2, 0.6, 1.0, alpha),
        "left_hand": (1.0, 0.3, 0.3, alpha),
        "right_hand": (0.3, 1.0, 0.3, alpha),
        "face": (1.0, 1.0, 0.3, alpha)
    }.get(part, (1, 1, 1, alpha))

def draw_frame(kps):
    ax.clear()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1.1, -0.1)
    ax.axis("off")
    ax.set_facecolor("black")

    # --- Pose ---
    for i, j in POSE_PAIRS:
        if np.all(kps[i] != 0) and np.all(kps[j] != 0):
            ax.plot([kps[i][0], kps[j][0]], [kps[i][1], kps[j][1]],
                    color=get_color("pose"), linewidth=3, solid_capstyle="round")

    # --- Hands ---
    for start, part in [(19, "left_hand"), (40, "right_hand")]:
        for i, j in HAND_CONNECTIONS:
            p1, p2 = kps[start + i], kps[start + j]
            if np.all(p1 != 0) and np.all(p2 != 0):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color=get_color(part), linewidth=2, solid_capstyle="round")

    # --- Keypoint circles ---
    def draw_circles(points, color, sizes):
        for idx, (x, y) in enumerate(points):
            if (x, y) != (0, 0):
                ax.add_patch(Circle((x, y), radius=sizes[idx % len(sizes)],
                                    color=color, zorder=10))

    draw_circles(kps[:19], get_color("pose"), [0.005] * 19)
    draw_circles(kps[19:40], get_color("left_hand"), [0.003] * 21)
    draw_circles(kps[40:61], get_color("right_hand"), [0.003] * 21)
    draw_circles(kps[61:], get_color("face", 0.5), [0.001] * (len(kps) - 61))

    # Subtle grid and title
    ax.grid(True, alpha=0.1, color="white", linewidth=0.5)
    ax.text(0.5, -0.05, "Sign Language Animation",
            transform=ax.transAxes, ha="center", color="white", fontsize=12, fontweight="bold")

def update(i):
    draw_frame(keypoints_smooth[i])
    return []

ani = animation.FuncAnimation(fig, update, frames=len(keypoints_smooth),
                              interval=1000/FPS, blit=False, repeat=True)

plt.tight_layout()
plt.show()

# === Optional: Save as GIF ===
if args.save:
    print(f"💾 Saving GIF to {args.save}")
    ani.save(args.save, writer="pillow", fps=FPS, dpi=100)
    print("✅ Saved.")
