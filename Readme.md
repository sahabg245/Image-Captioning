---
title: Neural Storyteller - Image Captioning
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Neural Storyteller - Image Captioning

An AI-powered image captioning model that generates natural language descriptions for images.

## Features
- Encoder-Decoder architecture with LSTM
- Trained on Flickr30k dataset (31,000 images)
- Beam Search for better caption quality
- Dual GPU training

## Model Architecture
- **Encoder:** ResNet50 → Linear(512)
- **Decoder:** 2-layer LSTM with embedding
- **Vocabulary:** ~8000 words

## Performance
- **BLEU-4 Score:** 0.20-0.25
- **Training Loss:** 2.01
- **Validation Loss:** 2.82

## Usage
1. Upload an image
2. Choose caption generation method
3. Click "Generate Caption"

## Training Details
- Dataset: Flickr30k
- Epochs: 30
- Hardware: 2x Tesla T4 GPUs
- Batch Size: 128
- Optimizer: Adam (lr=5e-4)

Built with PyTorch and Gradio