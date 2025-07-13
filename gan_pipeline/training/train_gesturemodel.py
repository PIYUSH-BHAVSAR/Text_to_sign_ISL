# train_gan.py - GAN training script for sign language video generation

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from Dataset_builder import GestureDataGenerator
import datetime
from tqdm import tqdm
import pickle
from dotenv import load_dotenv

# === Load Environment Variables ===
load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "./dataset")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./checkpoints")
LOG_BASE_DIR = os.getenv("LOG_DIR", "./logs")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
EPOCHS = int(os.getenv("EPOCHS", 10))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 64))
IMG_SHAPE = (16, 64, 64, 3)

LOG_DIR = os.path.join(LOG_BASE_DIR, "gesture_gan", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === TensorFlow Config ===
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"⚠️ GPU memory growth error: {e}")

# === 1. Load Word Labels ===
print("🔍 Scanning dataset and extracting labels...")
all_labels, word_count = [], {}

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            word = os.path.splitext(file)[0].lower()
            all_labels.append(word)
            word_count[word] = word_count.get(word, 0) + 1

print(f"📊 Found {len(set(all_labels))} unique words with {len(all_labels)} total samples")

# Fit label encoder
label_encoder = LabelEncoder()
label_encoder.fit(list(set(all_labels)))
num_classes = len(label_encoder.classes_)
print(f"🎯 Total classes: {num_classes}")

# === 2. Prepare Data Generator ===
train_gen = GestureDataGenerator(
    DATASET_DIR, label_encoder,
    batch_size=BATCH_SIZE,
    target_shape=IMG_SHAPE,
    shuffle=True,
    max_samples_per_word=20
)

# === 3. Build Models ===
def build_generator():
    label_input = layers.Input(shape=(1,), dtype='int32')
    x = layers.Embedding(num_classes, EMBEDDING_DIM)(label_input)
    x = layers.Flatten()(x)
    x = layers.Dense(4 * 4 * 128, activation='relu')(x)
    x = layers.Reshape((4, 4, 128))(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(16, 4, strides=2, padding='same', activation='relu')(x)
    x = layers.Reshape((1, 32, 32, 16))(x)
    x = layers.ConvLSTM2D(8, 3, padding='same', return_sequences=True)(x)
    x = layers.ConvLSTM2D(3, 3, activation='sigmoid', padding='same', return_sequences=True)(x)
    x = layers.UpSampling3D((IMG_SHAPE[0], 2, 2))(x)
    x = layers.Reshape(IMG_SHAPE)(x)
    return models.Model(label_input, x, name='generator')

def build_discriminator():
    video_input = layers.Input(shape=IMG_SHAPE)
    label_input = layers.Input(shape=(1,), dtype='int32')

    y = layers.Embedding(num_classes, EMBEDDING_DIM)(label_input)
    y = layers.Flatten()(y)
    y = layers.Dense(np.prod(IMG_SHAPE[:-1]))(y)
    y = layers.Reshape(IMG_SHAPE[:-1] + (1,))(y)

    x = layers.Concatenate()([video_input, y])
    x = layers.Conv3D(32, 4, strides=(1, 2, 2), padding='same', activation='relu')(x)
    x = layers.Conv3D(64, 4, strides=(2, 2, 2), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return models.Model([video_input, label_input], x, name='discriminator')

# Build & compile models
generator = build_generator()
discriminator = build_discriminator()
print(f"📐 Generator params: {generator.count_params():,}")
print(f"📐 Discriminator params: {discriminator.count_params():,}")

discriminator.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
)

# Build combined GAN
label_input = layers.Input(shape=(1,))
generated_video = generator(label_input)
discriminator.trainable = False
validity = discriminator([generated_video, label_input])
combined = models.Model(label_input, validity)
combined.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
)

# === 4. Logging ===
summary_writer = tf.summary.create_file_writer(LOG_DIR)

# === 5. Training Loop ===
print("🚀 Starting GAN training")
real = np.ones((BATCH_SIZE, 1))
fake = np.zeros((BATCH_SIZE, 1))

for epoch in range(1, EPOCHS + 1):
    print(f"\n🌀 Epoch {epoch}/{EPOCHS}")
    d_losses, g_losses = [], []
    pbar = tqdm(range(len(train_gen)), desc=f"Epoch {epoch}")

    for step in pbar:
        try:
            X_real, y_real = train_gen[step]
            current_batch = X_real.shape[0]
            if current_batch < 2:
                continue

            y_fake = np.random.randint(0, num_classes, (current_batch, 1))
            X_fake = generator.predict(y_fake, verbose=0)

            d_loss_real = discriminator.train_on_batch([X_real, y_real], real[:current_batch])
            d_loss_fake = discriminator.train_on_batch([X_fake, y_fake], fake[:current_batch])
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            g_input = np.random.randint(0, num_classes, (current_batch, 1))
            g_loss = combined.train_on_batch(g_input, real[:current_batch])

            d_losses.append(d_loss)
            g_losses.append(g_loss)

            pbar.set_postfix({'D_loss': f'{d_loss:.4f}', 'G_loss': f'{g_loss:.4f}'})
        except Exception as e:
            print(f"⚠️ Step {step} failed: {e}")
            continue

    avg_d, avg_g = np.mean(d_losses), np.mean(g_losses)
    print(f"📉 Epoch {epoch} - D loss: {avg_d:.4f}, G loss: {avg_g:.4f}")

    if epoch % 5 == 0 or epoch == EPOCHS:
        generator.save(os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch}.h5"))
        discriminator.save(os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch}.h5"))
        print(f"💾 Saved models for epoch {epoch}")

    with summary_writer.as_default():
        tf.summary.scalar("Discriminator Loss", avg_d, step=epoch)
        tf.summary.scalar("Generator Loss", avg_g, step=epoch)
        summary_writer.flush()

# === 6. Final Save ===
generator.save("sign_language_word_generator.h5")
discriminator.save("sign_language_word_discriminator.h5")
with open("word_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Training complete. Final models and label encoder saved.")
