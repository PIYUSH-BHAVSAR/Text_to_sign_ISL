# Sign Language Generation Project

A comprehensive machine learning project for generating Indian Sign Language (ISL) videos and keypoint animations from text input. This project implements two different approaches: **GAN-based video generation** and **LSTM-based keypoint prediction**.

## 🎯 Project Overview

This project provides two complementary pipelines:

1. **GAN Pipeline**: Generates sign language videos directly from word inputs using Generative Adversarial Networks
2. **LSTM Pipeline**: Converts sentences to keypoint sequences and animates them for sign language visualization

## 📁 Project Structure

```
├── gan_pipeline/                    # GAN-based video generation
│   ├── data_builder/               # Dataset preparation for GAN
│   ├── training/                   # GAN model training
│   ├── inference(testing)/         # Video generation from trained models
│   └── requirements.txt
├── lstm_pipeline/                  # LSTM-based keypoint generation
│   ├── preprocessing/              # Data preprocessing for LSTM
│   ├── training/                   # LSTM model training
│   ├── inference(testing)/         # Keypoint prediction and animation
│   └── requirements.txt
├── shared/                         # Common utilities
│   └── isl_video_generator.py      # Video-to-frames conversion
├── env_example_gan.txt             # Environment variables for GAN
├── env_example_lstm.txt            # Environment variables for LSTM
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- TensorFlow 2.9.1
- OpenCV
- MediaPipe
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sign-language-generation
```

2. Choose your pipeline and install dependencies:

**For GAN Pipeline:**
```bash
cd gan_pipeline
pip install -r requirements.txt
```

**For LSTM Pipeline:**
```bash
cd lstm_pipeline
pip install -r requirements.txt
```

### Environment Setup

1. Copy the appropriate environment template:
```bash
# For GAN
cp env_example_gan.txt .env

# For LSTM
cp env_example_lstm.txt .env
```

2. Edit the `.env` file with your specific paths and configurations.

## 🎬 GAN Pipeline - Video Generation

### Overview
The GAN pipeline generates sign language videos directly from word inputs using a conditional GAN architecture.

### Workflow

1. **Data Preparation**: Convert ISL videos to processed numpy arrays
2. **Training**: Train Generator and Discriminator networks
3. **Inference**: Generate videos for words or sentences

### Usage

#### 1. Dataset Preparation
```bash
cd gan_pipeline
python data_builder/Dataset_builder.py
```

#### 2. Training
```bash
python training/train_gesturemodel.py
```

#### 3. Generate Videos
```bash
python inference/test_model.py
```

### Configuration (env_example_gan.txt)
```bash
DATASET_DIR=./isl_processed_dataset
CHECKPOINT_DIR=./checkpoints
BATCH_SIZE=8
EPOCHS=10
GENERATOR_PATH=sign_language_word_generator.h5
ENCODER_PATH=word_label_encoder.pkl
```

## 🎯 LSTM Pipeline - Keypoint Generation

### Overview
The LSTM pipeline converts sentences to keypoint sequences representing sign language gestures, focusing on upper body movements.

### Workflow

1. **Video Processing**: Extract frames from ISL videos
2. **Keypoint Extraction**: Extract MediaPipe keypoints from frames
3. **Dataset Creation**: Map sentences to keypoint sequences
4. **Training**: Train LSTM to predict keypoints from sentence embeddings
5. **Inference**: Generate and animate keypoints for new sentences

### Usage

#### 1. Process Videos to Frames
```bash
python shared/isl_video_generator.py
```

#### 2. Extract Keypoints
```bash
python lstm_pipeline/preprocessing/npy_frames_to_keypoints.py
```

#### 3. Create Sentence Mappings
```bash
python lstm_pipeline/preprocessing/mapping_genrator.py
```

#### 4. Generate Training Dataset
```bash
python lstm_pipeline/preprocessing/sent2embed.py
```

#### 5. Train LSTM Model
```bash
python lstm_pipeline/training/train_lstm.py
```

#### 6. Generate Keypoints for New Sentences
```bash
python lstm_pipeline/inference/predictkeypoints.py --text "hello how are you" --output predicted_keypoints.npy
```

#### 7. Animate Keypoints
```bash
python lstm_pipeline/inference/animatekeypoints.py --file predicted_keypoints.npy --save animation.gif
```

### Configuration (env_example_lstm.txt)
```bash
VIDEO_ROOT=path/to/ISL_Dictionary
OUTPUT_ROOT=path/to/processed_npy
KEYPOINT_ROOT=path/to/isl_keypoints_dataset
MAPPING_JSON=sentence_keypoint_mapping.json
EMBEDDING_MODEL=all-MiniLM-L6-v2
EPOCHS=50
BATCH_SIZE=16
```

## 📊 Key Features

### GAN Pipeline
- **Conditional GAN**: Generates videos conditioned on word labels
- **Word-based Generation**: Creates sign language videos for individual words
- **Sentence Composition**: Combines word videos to form sentences
- **Video Output**: Direct video/GIF generation

### LSTM Pipeline
- **Sentence Understanding**: Uses transformer-based sentence embeddings
- **Upper Body Focus**: Extracts pose, hand, and facial keypoints
- **Smooth Animation**: Applies smoothing filters for natural movement
- **Keypoint Visualization**: Creates animated visualizations of sign language

## 🔧 Technical Details

### GAN Architecture
- **Generator**: ConvLSTM2D layers for temporal video generation
- **Discriminator**: 3D convolutional layers for video discrimination
- **Conditioning**: Word embeddings for controlled generation
- **Output**: 16-frame videos at 64x64 resolution

### LSTM Architecture
- **Input**: 384-dimensional sentence embeddings (MiniLM)
- **Output**: 32-frame sequences of 529 keypoints (2D coordinates)
- **Network**: Multi-layer LSTM with dense layers
- **Keypoints**: 19 pose + 42 hand + 468 face landmarks

### Performance Optimizations
- **Multiprocessing**: Parallel processing for dataset creation
- **GPU Acceleration**: CUDA support for training
- **Memory Management**: Efficient data loading and caching
- **CPU Monitoring**: Adaptive processing based on system load

## 📈 Results and Evaluation

### GAN Pipeline
- Generates recognizable sign language gestures for trained words
- Smooth temporal transitions in generated videos
- Scalable to new vocabularies with retraining

### LSTM Pipeline
- Produces anatomically plausible keypoint sequences
- Maintains temporal coherence across frames
- Supports complex sentence structures

## 🛠️ Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   - Reduce batch size in environment variables
   - Use gradient checkpointing for large models

2. **Dataset Path Errors**
   - Verify all paths in `.env` file
   - Ensure dataset structure matches expected format

3. **Model Loading Errors**
   - Check TensorFlow version compatibility
   - Verify model file paths and permissions

4. **Keypoint Extraction Slow**
   - Reduce number of processes in multiprocessing
   - Monitor CPU usage and temperature

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{text_to_sign_isl,
  title={Text-to-Indian Sign Language Translation using GANs and LSTMs},
  author={Piyush Bhavsar},
  year={2025},
  url={https://github.com/PIYUSH-BHAVSAR/Text_to_sign_ISL}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## 🙏 Acknowledgments

- MediaPipe for pose and hand detection
- Sentence Transformers for text embeddings
- TensorFlow team for the deep learning framework
- ISL Dictionary contributors for the dataset

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the code comments for detailed explanations

---

**Note**: This project is intended for research and educational purposes only. The datasets used in this work are official and have been utilized with appropriate permission solely for academic and non-commercial use.
