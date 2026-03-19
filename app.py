import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI(title="AuditMind Deepfake API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. THE EXACT KAGGLE ARCHITECTURE ---
def get_log_magnitude_spectrum(tensor_batch):
    fft = torch.fft.fft2(tensor_batch)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shift) + 1e-8
    return torch.log(magnitude)

class FrequencyTemporalViT(nn.Module):
    def __init__(self, num_classes=1):
        super(FrequencyTemporalViT, self).__init__()
        # Must match Kaggle exactly: vit_tiny_patch16_224
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)
        self.feature_dim = self.vit.num_features
        self.temporal_attention = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W) 
        spatial_features = self.vit(x) 
        temporal_features = spatial_features.view(B, T, -1) 
        attn_output, _ = self.temporal_attention(temporal_features, temporal_features, temporal_features)
        video_feature = attn_output.mean(dim=1) 
        logits = self.fc(video_feature) 
        return logits.squeeze(-1)

# --- 2. LOAD MODEL PIPELINE ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FrequencyTemporalViT().to(device)

# We will load the weights ONLY if the file exists (prevents crashing before you upload the model tomorrow)
MODEL_PATH = "best_deepfake_model.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("✅ AuditMind Model Loaded Successfully!")
else:
    print("⚠️ Warning: Model weights not found. Waiting for best_deepfake_model.pth")

# --- 3. INFERENCE ENDPOINT ---
@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    if not file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported.")
    
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model is currently training. Please try again later.")

    # Save uploaded video temporarily
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        # Extract exactly 8 evenly spaced frames using OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
        cap.release()
        
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="Could not extract frames from video.")
            
        # Pad if fewer than 8 frames were somehow extracted
        stacked_frames = torch.stack(frames)
        if stacked_frames.shape[0] < 8:
            stacked_frames = F.pad(stacked_frames, (0, 0, 0, 0, 0, 0, 0, 8 - stacked_frames.shape[0]))
            
        # Run Kaggle FFT Preprocessing
        freq_tensor = get_log_magnitude_spectrum(stacked_frames)
        freq_tensor = (freq_tensor - freq_tensor.mean()) / (freq_tensor.std() + 1e-6)
        
        # Add Batch Dimension [1, 8, 3, 224, 224] and predict
        freq_tensor = freq_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logit = model(freq_tensor)
            prob = torch.sigmoid(logit).item()
            
        prediction = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prediction == "FAKE" else (1 - prob)
        
        return {
            "prediction": prediction,
            "confidence": f"{confidence * 100:.2f}%",
            "raw_probability": prob
        }
        
    finally:
        # Cleanup
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@app.get("/")
def read_root():
    # This tells FastAPI to serve your frontend UI when someone visits the site
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "UI not found. Ensure index.html is in the root directory."}