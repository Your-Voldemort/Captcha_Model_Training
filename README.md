# CAPTCHA Model Training

A GPU-accelerated CAPTCHA solver using deep learning. This project implements a complete pipeline for training, evaluating, and deploying a CRNN (Convolutional Recurrent Neural Network) model for CAPTCHA recognition.

## Features

- **CRNN Architecture**: Combines CNNs for feature extraction with RNNs for sequence modeling
- **GPU Acceleration**: Full CUDA support with automatic fallback to CPU
- **Data Pipeline**: Automated dataset handling with efficient batching and preprocessing
- **Manual Labeling Tool**: Interactive GUI for labeling CAPTCHA images
- **Batch Processing**: Multi-threaded processing with proxy rotation for data collection
- **FastAPI Server**: REST API for serving predictions with automatic model loading
- **Model Evaluation**: Comprehensive metrics (accuracy, loss, character-level metrics)
- **Mixed Precision Training**: Automatic mixed precision (AMP) for faster training and lower memory usage

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support (CPU fallback available)
- PyTorch with CUDA 12.4 support

### Installation

Create a virtual environment and install dependencies:

```bash
# Using uv (recommended)
uv venv --python 3.11
uv sync

# Or using pip
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
```

### Basic Usage

#### Train the Model

```bash
python train.py --data-dir data/ --checkpoint-dir checkpoints/ --epochs 50 --batch-size 32
```

**Key options:**
- `--device cuda|cpu` - Training device (default: auto-detect)
- `--learning-rate` - Initial learning rate (default: 0.001)
- `--use-amp` - Enable automatic mixed precision

#### Run Inference

```bash
python predict.py --model checkpoints/best.pt --image path/to/captcha.png
```

#### Label Data Manually

Interactive GUI for labeling CAPTCHA images:

```bash
python label.py --images-dir data/raw/ --labels labels.csv
```

#### Start API Server

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Make predictions via HTTP:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@captcha.png"
```

#### Batch Processing

Process multiple images with automated labeling and proxy rotation:

```bash
python api.py \
  --urls-file urls.txt \
  --output-dir data/downloaded/ \
  --labels-file labels.csv \
  --threads 5 \
  --batch-size 50
```

## Project Structure

```
.
├── train.py              # Model training pipeline
├── predict.py            # Inference and model loading
├── server.py             # FastAPI server for predictions
├── api.py                # Batch processing with proxy rotation
├── label.py              # GUI labeling tool
├── evaluate_model.py     # Model evaluation metrics
├── model.py              # CRNN model architecture
├── dataset.py            # Dataset loading and preprocessing
├── charset.py            # Character set and encoding/decoding
├── checkpoints/          # Trained model checkpoints
├── data/                 # Training data directory
└── requirements.txt      # Python dependencies
```

## Model Architecture

The CRNN model consists of:

1. **CNN Backbone** (ResNet-18): Extracts visual features from CAPTCHA images
2. **RNN Encoder** (LSTM): Sequences the features temporally
3. **CTC Loss**: Handles variable-length predictions

## Training Details

- **Optimizer**: AdamW with cosine annealing scheduler
- **Loss Function**: CTC (Connectionist Temporal Classification)
- **Image Size**: 128×32 pixels (grayscale)
- **Character Set**: Alphanumeric + special characters
- **Evaluation Metrics**:
  - Exact match accuracy
  - Character-level accuracy
  - CTC loss

## API Endpoints

### POST /predict

Predict a CAPTCHA from an uploaded image.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.png"
```

**Response:**
```json
{
  "prediction": "ABC123",
  "confidence": 0.98
}
```

## Batch Processing

The batch processor supports:

- **Multi-threading**: Concurrent image downloads and processing
- **Proxy Rotation**: Automatic proxy cycling to avoid IP blocks
- **Rate Limiting**: Configurable delays between requests
- **CSV Logging**: Automated label storage with image metadata

## Requirements

Core dependencies:
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `Pillow` - Image processing
- `numpy` - Numerical computing
- `tqdm` - Progress bars
- `fastapi` - REST API framework
- `fake-useragent` - User-agent rotation

## Tips

> **GPU Memory**: For out-of-memory errors, reduce batch size with `--batch-size 16`

> **Data Preparation**: Ensure CAPTCHA images are approximately 128×32 pixels for best results

> **Model Checkpoints**: The best model is automatically saved to `checkpoints/best.pt` during training

## Performance

On an NVIDIA GPU:
- Training: ~200 images/sec
- Inference: ~1000 images/sec
- Model size: ~41 MB

## Troubleshooting

**CUDA not found:**
```bash
# Verify PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Model checkpoint missing:**
```bash
# Train a new model first
python train.py --data-dir data/
```

**Server won't start:**
```bash
# Check if port 8000 is available
lsof -i :8000
```
