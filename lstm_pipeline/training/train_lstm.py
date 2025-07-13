import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "sentence_keypoint_dataset_upper_full.npz")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/sentence2keypoints_upper_full.h5")
FINAL_MODEL_PATH = os.getenv("FINAL_MODEL_PATH", "sentence2keypoints_upper_full_final.h5")
EPOCHS = int(os.getenv("EPOCHS", 50))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

# === Load dataset ===
print("📦 Loading dataset from:", DATA_PATH)
data = np.load(DATA_PATH)
X = data["X"]  # Sentence embeddings (e.g., shape: [100, 384])
Y = data["Y"]  # Keypoints (e.g., shape: [100, 32, 529, 2])

# === Flatten keypoints from (32, 529, 2) → (32, 1058)
Y = Y.reshape(Y.shape[0], Y.shape[1], -1)

# === Train-validation split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
print(f"🧠 X_train: {X_train.shape}, 🖐️ Y_train: {Y_train.shape}")

# === Build LSTM model ===
input_emb = layers.Input(shape=(X.shape[1],))
x = layers.Dense(512, activation='relu')(input_emb)
x = layers.RepeatVector(Y.shape[1])(x)  # Repeat for 32 frames
x = layers.LSTM(512, return_sequences=True)(x)
x = layers.LSTM(256, return_sequences=True)(x)
output = layers.TimeDistributed(layers.Dense(Y.shape[2]))(x)

model = models.Model(inputs=input_emb, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# === Ensure checkpoint directory exists
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

# === Training ===
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True)
    ]
)

# === Save final model
model.save(FINAL_MODEL_PATH)
print(f"✅ Training complete. Final model saved to: {FINAL_MODEL_PATH}")
