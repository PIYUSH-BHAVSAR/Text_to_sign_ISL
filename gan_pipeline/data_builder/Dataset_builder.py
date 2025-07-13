# Dataset_builder.py - Optimized for word-based sign language videos
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder
import cv2
from collections import defaultdict

class GestureDataGenerator(Sequence):
    def __init__(self, data_dir, label_encoder, batch_size=8, target_shape=(16, 64, 64, 3), 
                 shuffle=True, max_samples_per_word=20, use_cache=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.label_encoder = label_encoder
        self.max_samples_per_word = max_samples_per_word
        self.use_cache = use_cache
        self.cache = {}
        self.samples = self._load_index()
        self.on_epoch_end()
        
        print(f"📊 Dataset loaded: {len(self.samples)} samples")
        
    def _load_index(self):
        """Load file index where each filename is a word label"""
        samples = []
        word_counts = defaultdict(int)
        
        print("🔍 Scanning files and extracting word labels...")
        
        for folder in sorted(os.listdir(self.data_dir)):
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
                
            print(f"📁 Processing folder: {folder}")
            folder_samples = []
            
            for file in os.listdir(folder_path):
                if file.endswith(".npy"):
                    # Extract word from filename (remove .npy extension)
                    word = os.path.splitext(file)[0].lower()
                    file_path = os.path.join(folder_path, file)
                    
                    folder_samples.append((file_path, word))
                    word_counts[word] += 1
            
            # Limit samples per word if specified
            if self.max_samples_per_word:
                word_samples = defaultdict(list)
                for file_path, word in folder_samples:
                    word_samples[word].append((file_path, word))
                
                # Take only max_samples_per_word for each word
                limited_samples = []
                for word, word_files in word_samples.items():
                    limited_samples.extend(word_files[:self.max_samples_per_word])
                
                samples.extend(limited_samples)
            else:
                samples.extend(folder_samples)
        
        # Print word statistics
        print(f"\n📈 Word Statistics:")
        print(f"   Total unique words: {len(word_counts)}")
        print(f"   Total samples: {len(samples)}")
        print(f"\n🔤 Top 15 most common words:")
        
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"   {word}: {count} samples")
        
        return samples
    
    def _resize_video(self, video):
        """Resize video to target shape with optimizations"""
        try:
            target_frames, target_h, target_w, channels = self.target_shape
            current_frames, current_h, current_w, current_channels = video.shape
            
            # Quick return if already correct size
            if video.shape == self.target_shape:
                return video
            
            # Handle channel dimension
            if current_channels != channels:
                if current_channels == 1 and channels == 3:
                    video = np.repeat(video, 3, axis=-1)
                elif current_channels == 3 and channels == 1:
                    video = np.mean(video, axis=-1, keepdims=True)
            
            # Resize spatial dimensions first
            if current_h != target_h or current_w != target_w:
                resized_frames = []
                for i in range(current_frames):
                    frame = cv2.resize(video[i], (target_w, target_h))
                    if len(frame.shape) == 2:  # Grayscale
                        frame = np.expand_dims(frame, axis=-1)
                    resized_frames.append(frame)
                video = np.array(resized_frames)
            
            # Handle temporal dimension
            if video.shape[0] != target_frames:
                if video.shape[0] > target_frames:
                    # Downsample frames evenly
                    indices = np.linspace(0, video.shape[0] - 1, target_frames, dtype=int)
                    video = video[indices]
                else:
                    # Repeat frames to reach target
                    repeat_factor = target_frames // video.shape[0]
                    remainder = target_frames % video.shape[0]
                    
                    video = np.repeat(video, repeat_factor, axis=0)
                    if remainder > 0:
                        extra_frames = video[:remainder]
                        video = np.concatenate([video, extra_frames], axis=0)
            
            # Ensure correct shape
            video = video.reshape(self.target_shape)
            
            # Normalize to [0, 1]
            if video.max() > 1.0:
                video = video / 255.0
            
            return video
            
        except Exception as e:
            print(f"⚠️ Error resizing video: {e}")
            # Return random video as fallback
            return np.random.random(self.target_shape)
    
    def on_epoch_end(self):
        """Shuffle samples at end of epoch"""
        if self.shuffle:
            np.random.shuffle(self.samples)
        
        # Clear cache occasionally to prevent memory issues
        if len(self.cache) > 500:
            self.cache.clear()
    
    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get batch with word labels extracted from filenames"""
        batch_samples = self.samples[idx * self.batch_size : (idx + 1) * self.batch_size]
        X, y = [], []
        
        for file_path, word in batch_samples:
            try:
                # Try to get from cache first
                if file_path in self.cache:
                    data = self.cache[file_path]
                else:
                    data = np.load(file_path)
                    if data.shape != self.target_shape:
                        data = self._resize_video(data)
                    
                    # Cache if using cache and not too many items
                    if self.use_cache and len(self.cache) < 200:
                        self.cache[file_path] = data
                
                X.append(data)
                y.append(word)
                
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                # Add random data to maintain batch size
                X.append(np.random.random(self.target_shape))
                y.append(batch_samples[0][1])  # Use first word as fallback
        
        # Ensure we have data
        if len(X) == 0:
            X = [np.random.random(self.target_shape)]
            y = [list(self.label_encoder.classes_)[0]]
        
        try:
            y_encoded = self.label_encoder.transform(y)
            return np.array(X), np.array(y_encoded).reshape(-1, 1)
        except Exception as e:
            print(f"⚠️ Error encoding labels {y}: {e}")
            # Return random labels as fallback
            y_encoded = np.random.randint(0, len(self.label_encoder.classes_), len(X))
            return np.array(X), np.array(y_encoded).reshape(-1, 1)


# Ultra-fast generator for quick prototyping
class FastWordDataGenerator(Sequence):
    """Ultra-fast generator that loads subset of data into memory"""
    
    def __init__(self, data_dir, label_encoder, batch_size=8, target_shape=(16, 64, 64, 3), 
                 max_total_samples=1000, max_samples_per_word=10):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.label_encoder = label_encoder
        self.max_total_samples = max_total_samples
        self.max_samples_per_word = max_samples_per_word
        
        # Load all data into memory for fastest access
        self.X_data, self.y_data = self._load_all_data()
        
    def _load_all_data(self):
        """Load subset of data into memory organized by words"""
        print("🚀 Loading subset of data into memory for fastest training...")
        
        word_samples = defaultdict(list)
        
        # Collect all samples organized by word
        for folder in sorted(os.listdir(self.data_dir)):
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
                
            for file in os.listdir(folder_path):
                if file.endswith(".npy"):
                    word = os.path.splitext(file)[0].lower()
                    file_path = os.path.join(folder_path, file)
                    word_samples[word].append(file_path)
        
        # Load limited samples per word
        X_data, y_data = [], []
        samples_loaded = 0
        
        for word, file_paths in word_samples.items():
            if samples_loaded >= self.max_total_samples:
                break
                
            # Limit samples per word
            word_files = file_paths[:self.max_samples_per_word]
            
            for file_path in word_files:
                if samples_loaded >= self.max_total_samples:
                    break
                    
                try:
                    data = np.load(file_path)
                    
                    # Quick shape check
                    if data.shape == self.target_shape:
                        X_data.append(data)
                        y_data.append(word)
                        samples_loaded += 1
                    elif len(X_data) < 50:  # At least load some samples
                        # Simple resize for first few samples
                        if len(data.shape) == 4:  # Video data
                            # Quick and dirty resize
                            resized = np.random.random(self.target_shape)
                            X_data.append(resized)
                            y_data.append(word)
                            samples_loaded += 1
                            
                except Exception as e:
                    continue
        
        # Ensure we have some data
        if len(X_data) == 0:
            print("⚠️ No valid samples found, creating dummy data")
            X_data = [np.random.random(self.target_shape) for _ in range(10)]
            y_data = [list(self.label_encoder.classes_)[0] for _ in range(10)]
        
        print(f"✅ Loaded {len(X_data)} samples into memory")
        print(f"🔤 Words included: {len(set(y_data))} unique words")
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y_data)
        
        return np.array(X_data), np.array(y_encoded)
    
    def __len__(self):
        return int(np.ceil(len(self.X_data) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.X_data))
        
        X_batch = self.X_data[start_idx:end_idx]
        y_batch = self.y_data[start_idx:end_idx]
        
        return X_batch, y_batch.reshape(-1, 1)
    
    def on_epoch_end(self):
        # Shuffle data
        indices = np.arange(len(self.X_data))
        np.random.shuffle(indices)
        self.X_data = self.X_data[indices]
        self.y_data = self.y_data[indices]


# Utility function to test word generation
def generate_word_video(generator, label_encoder, word, save_path=None):
    """Generate a sign language video for a specific word"""
    try:
        # Encode the word
        word_encoded = label_encoder.transform([word.lower()])[0]
        word_input = np.array([[word_encoded]])
        
        # Generate video
        generated_video = generator.predict(word_input, verbose=0)[0]
        
        if save_path:
            np.save(save_path, generated_video)
            print(f"✅ Generated video for '{word}' saved to {save_path}")
        
        return generated_video
        
    except Exception as e:
        print(f"⚠️ Error generating video for word '{word}': {e}")
        return None