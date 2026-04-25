# src/main.py 
# ───────────────────────────────────────────────────────────────── 
# FastAPI server exposing /predict endpoint. 
# Run: uvicorn src.main:app --reload --port 8000 
# ───────────────────────────────────────────────────────────────── 

import io, json 
import numpy as np 
import torch 
import torch.nn as nn 
import joblib 
from fastapi import FastAPI, File, UploadFile, Form, HTTPException 
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import JSONResponse 
from torchvision import transforms, models 
from PIL import Image 
from pydantic import BaseModel 
from typing import List 

# ── App Initialisation ─────────────────────────────────────────── 
app = FastAPI( 
    title="Crop Disease Detection API", 
    description="CNN + XGBoost hybrid pipeline for leaf disease detection and yield prediction", 
    version="1.0.0" 
) 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allows all origins during development
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ─────────────────────────────────────────── 
cnn_model   = None 
xgb_model   = None 
class_names = None 
img_transform = None 

@app.on_event("startup") 
async def startup(): 
    global cnn_model, xgb_model, class_names, img_transform 

    with open("models/class_names.json") as f: 
        class_names = json.load(f) 

    num_classes = len(class_names) 

    # Load CNN 
    backbone = models.resnet50(weights=None) 
    in_f     = backbone.fc.in_features 
    backbone.fc = nn.Sequential( 
        nn.BatchNorm1d(in_f), nn.Dropout(0.5), 
        nn.Linear(in_f, 512), nn.ReLU(inplace=True), 
        nn.Dropout(0.3), nn.Linear(512, num_classes) 
    ) 
    backbone.load_state_dict( 
        torch.load("models/best_cnn.pth", map_location="cpu") 
    ) 
    backbone.eval() 
    cnn_model = backbone 

    # Load XGBoost 
    xgb_model = joblib.load("models/yield_model.pkl") 

    # Image transform (no augmentation at inference) 
    img_transform = transforms.Compose([ 
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]) 
    ]) 

    print("Models loaded successfully.") 

# ── Response Schema ────────────────────────────────────────────── 
class ClassEntry(BaseModel): 
    disease_class: str 
    probability:   float 

class PredictionResponse(BaseModel): 
    disease:           str 
    confidence:        float 
    severity:          str 
    yield_loss_pct:    float 
    recommendation:    dict 
    top5:              List[ClassEntry] 

# ── Predict Endpoint ───────────────────────────────────────────── 
@app.post("/predict", response_model=PredictionResponse) 
async def predict( 
    file:           UploadFile = File(...), 
    temperature:    float = Form(30.0), 
    humidity:       float = Form(70.0), 
    rainfall:       float = Form(50.0), 
    soil_ph:        float = Form(6.5), 
    nitrogen:       float = Form(50.0), 
    phosphorus:     float = Form(30.0), 
    region:         int   = Form(0), 
    days_sowing:    int   = Form(60), 
): 
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]: 
        raise HTTPException(400, "Only JPEG/PNG images are accepted") 

    # ── CNN Inference ──────────────────────────────────────────── 
    img_bytes = await file.read() 
    image     = Image.open(io.BytesIO(img_bytes)).convert("RGB") 
    tensor    = img_transform(image).unsqueeze(0) 

    with torch.no_grad(): 
        logits = cnn_model(tensor) 
        probs  = torch.softmax(logits, dim=1) 
        top5   = probs.topk(5) 

    pred_idx  = top5.indices[0][0].item() 
    pred_conf = top5.values[0][0].item() 
    disease   = class_names[pred_idx] 

    # ── Severity ───────────────────────────────────────────────── 
    severity = ("severe"   if pred_conf >= 0.80 else 
                "moderate" if pred_conf >= 0.60 else 
                "mild") 
    sev_int  = {"mild": 1, "moderate": 2, "severe": 3}.get(severity, 1) 

    # ── XGBoost Yield Prediction ────────────────────────────────── 
    features = np.array([[ 
        sev_int, pred_conf, pred_idx, 
        temperature, humidity, rainfall, 
        soil_ph, nitrogen, phosphorus, region, days_sowing 
    ]]) 
    yield_loss = float(xgb_model.predict(features)[0]) 
    yield_loss = max(0.0, min(yield_loss, 85.0)) 

    # ── Recommendation ──────────────────────────────────────────── 
    from src.treatment_db import get_recommendation 
    rec = get_recommendation(disease, pred_conf, region) 

    # ── Build Response ──────────────────────────────────────────── 
    top5_list = [ 
        ClassEntry( 
            disease_class=class_names[top5.indices[0][i].item()], 
            probability=round(top5.values[0][i].item(), 4) 
        ) for i in range(5) 
    ] 

    return PredictionResponse( 
        disease        = disease, 
        confidence     = round(pred_conf, 4), 
        severity       = severity, 
        yield_loss_pct = round(yield_loss, 2), 
        recommendation = rec, 
        top5           = top5_list 
    ) 

@app.get("/health") 
def health(): return {"status": "ok", "models_loaded": cnn_model is not None} 

@app.get("/classes") 
def get_classes(): return {"classes": class_names, "count": len(class_names)}