# src/gradcam.py 
# ───────────────────────────────────────────────────────────────── 
# Generates Grad-CAM heatmap overlays for individual leaf images. 
# Usage: python gradcam.py --image path/to/leaf.jpg 
# ───────────────────────────────────────────────────────────────── 

import json, argparse 
import numpy as np 
import torch 
import torch.nn as nn 
from torchvision import transforms, models 
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget 
from PIL import Image 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 

def load_model(model_path, num_classes): 
    model = models.resnet50(weights=None) 
    in_f  = model.fc.in_features 
    model.fc = nn.Sequential( 
        nn.BatchNorm1d(in_f), nn.Dropout(0.5), 
        nn.Linear(in_f, 512), nn.ReLU(inplace=True), 
        nn.Dropout(0.3), nn.Linear(512, num_classes) 
    ) 
    model.load_state_dict(torch.load(model_path, map_location="cpu")) 
    model.eval() 
    return model 

def run_gradcam(image_path, model_path="models/best_cnn.pth", 
                classes_path="models/class_names.json", 
                output_path="gradcam_output.png"): 

    # Load class names 
    with open(classes_path) as f: 
        class_names = json.load(f) 

    model = load_model(model_path, len(class_names)) 

    # Preprocessing 
    MEAN = [0.485, 0.456, 0.406] 
    STD  = [0.229, 0.224, 0.225] 

    # Load image in two forms: 
    # 1. float32 numpy [0,1] for overlay 
    # 2. normalised tensor for model input 
    pil_img  = Image.open(image_path).convert("RGB").resize((224, 224)) 
    rgb_img  = np.array(pil_img, dtype=np.float32) / 255.0 
    input_t  = preprocess_image(rgb_img, mean=MEAN, std=STD) 

    # Run forward pass to get prediction 
    with torch.no_grad(): 
        logits = model(input_t) 
        probs  = torch.softmax(logits, dim=1) 
        top5   = probs.topk(5) 

    pred_idx  = top5.indices[0][0].item() 
    pred_conf = top5.values[0][0].item() 
    pred_name = class_names[pred_idx] 

    print(f"Prediction: {pred_name}  (confidence: {pred_conf:.2%})") 
    print("Top-5 predictions:") 
    for i in range(5): 
        idx = top5.indices[0][i].item() 
        print(f"  {class_names[idx]:<40} {top5.values[0][i].item():.2%}") 

    # ── Grad-CAM ──────────────────────────────────────────────── 
    # Target the last residual block — richest spatial features 
    target_layers = [model.layer4[-1]] 

    cam    = GradCAM(model=model, target_layers=target_layers) 
    targets = [ClassifierOutputTarget(pred_idx)] 

    grayscale_cam = cam(input_tensor=input_t, targets=targets)[0] 
    cam_image     = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True) 

    # ── Visualisation ─────────────────────────────────────────── 
    fig = plt.figure(figsize=(12, 5)) 
    gs  = gridspec.GridSpec(1, 3, figure=fig) 

    ax1 = fig.add_subplot(gs[0]) 
    ax1.imshow(pil_img) 
    ax1.set_title("Original Leaf", fontsize=12, fontweight="bold") 
    ax1.axis("off") 

    ax2 = fig.add_subplot(gs[1]) 
    heatmap = ax2.imshow(grayscale_cam, cmap="jet", vmin=0, vmax=1) 
    plt.colorbar(heatmap, ax=ax2) 
    ax2.set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold") 
    ax2.axis("off") 

    ax3 = fig.add_subplot(gs[2]) 
    ax3.imshow(cam_image) 
    title = f"{pred_name.replace("___", chr(10)).replace("_", " ")}", 
    ax3.set_title( 
        f"Overlay\n{pred_name.split("___")[1].replace("_", " ")}\n{pred_conf:.1%}", 
        fontsize=11, fontweight="bold", color="darkgreen") 
    ax3.axis("off") 

    plt.tight_layout() 
    plt.savefig(output_path, dpi=150, bbox_inches="tight") 
    print(f"Saved Grad-CAM visualisation: {output_path}") 
    return pred_name, pred_conf, grayscale_cam 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--image",  required=True) 
    parser.add_argument("--output", default="gradcam_output.png") 
    args = parser.parse_args() 
    run_gradcam(args.image, output_path=args.output) 