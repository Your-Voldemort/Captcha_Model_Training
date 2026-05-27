import io
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

# Import existing model architecture and preprocessing configuration
from charset import IMG_HEIGHT, IMG_WIDTH
from dataset import decode_indices
from predict import load_model

# Constants
CHECKPOINT_PATH = Path("checkpoints/best.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on startup and clear memory on shutdown."""
    global model
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}. Please train the model first.")
    
    print(f"Loading model on {DEVICE}...")
    model = load_model(CHECKPOINT_PATH, DEVICE)
    yield
    # Cleanup (if any) on shutdown
    model = None

# Initialize FastAPI with lifespan management
app = FastAPI(
    title="CAPTCHA Solver API",
    description="API for solving CAPTCHAs using the trained CRNN model",
    lifespan=lifespan
)

def preprocess_image_from_bytes(image_bytes: bytes) -> torch.Tensor:
    """Preprocess the image bytes into the format expected by the CRNN model."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
        tensor = (tensor - 0.5) / 0.5
        return tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an uploaded CAPTCHA image and return the predicted text."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # Read image file bytes
    contents = await file.read()
    
    # Preprocess
    tensor = preprocess_image_from_bytes(contents).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        log_probs = model(tensor)
    
    # Decode predicted indices into text
    sequence = log_probs.argmax(dim=2).squeeze(1).tolist()
    pred_text = decode_indices(sequence)
    
    return {
        "status": "Success",
        "predicted_text": pred_text,
        "filename": file.filename
    }

@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}
