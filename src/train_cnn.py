# src/train_cnn.py 
# ───────────────────────────────────────────────────────────────── 
# Trains ResNet-50 on PlantVillage for crop disease classification. 
# Saves best model weights and class_names.json. 
# ───────────────────────────────────────────────────────────────── 

import os, json, copy 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.optim.lr_scheduler import StepLR 
from torch.utils.data import DataLoader, random_split 
from torchvision import datasets, transforms, models 
from sklearn.metrics import classification_report 
import matplotlib.pyplot as plt 
import numpy as np 

# ── Configuration ─────────────────────────────────────────────── 
DATA_DIR     = "data/PlantVillage" 
MODEL_PATH   = "models/best_cnn.pth" 
CLASSES_PATH = "models/class_names.json" 
BATCH_SIZE   = 32 
NUM_EPOCHS   = 20 
LR           = 1e-3 
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

print(f"Training on: {DEVICE}") 

# ── Data Transforms ───────────────────────────────────────────── 
# ImageNet mean/std because ResNet-50 was pretrained on ImageNet. 
# Applying the same normalisation ensures the pretrained filters 
# see input data in the expected range. 
MEAN = [0.485, 0.456, 0.406] 
STD  = [0.229, 0.224, 0.225] 

train_transform = transforms.Compose([ 
    transforms.Resize((256, 256)), 
    transforms.RandomCrop(224),           # random spatial crop 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.2), 
    transforms.RandomRotation(degrees=20), 
    transforms.ColorJitter( 
        brightness=0.3, contrast=0.3, 
        saturation=0.2, hue=0.1 
    ), 
    transforms.ToTensor(), 
    transforms.Normalize(MEAN, STD) 
]) 

val_transform = transforms.Compose([ 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(MEAN, STD) 
]) 

# ── Dataset Loading ────────────────────────────────────────────── 
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform) 
class_names  = full_dataset.classes 
num_classes  = len(class_names) 

# Save class names for inference 
os.makedirs("models", exist_ok=True) 
with open(CLASSES_PATH, "w") as f: 
    json.dump(class_names, f) 

print(f"Classes: {num_classes}  |  Total images: {len(full_dataset)}") 

# 80/10/10 split 
total      = len(full_dataset) 
train_size = int(0.80 * total) 
val_size   = int(0.10 * total) 
test_size  = total - train_size - val_size 
train_set, val_set, test_set = random_split( 
    full_dataset, 
    [train_size, val_size, test_size], 
    generator=torch.Generator().manual_seed(42) 
) 

# Apply val transform to val/test sets 
val_set.dataset  = copy.copy(full_dataset) 
val_set.dataset.transform  = val_transform 
test_set.dataset = copy.copy(full_dataset) 
test_set.dataset.transform = val_transform 

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, 
                        shuffle=True,  num_workers=4, pin_memory=True) 
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=4, pin_memory=True) 
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=4, pin_memory=True) 

print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}") 

# ── Model Definition ──────────────────────────────────────────── 
def build_resnet50(num_classes, freeze_backbone=True): 
    """ 
    Load ImageNet-pretrained ResNet-50 and replace the final 
    fully-connected layer with a custom classification head. 
    freeze_backbone=True means only the head is trained initially. 
    """ 
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) 

    if freeze_backbone: 
        for param in model.parameters(): 
            param.requires_grad = False 

    in_features = model.fc.in_features  # 2048 for ResNet-50 
    model.fc = nn.Sequential( 
        nn.BatchNorm1d(in_features), 
        nn.Dropout(p=0.5), 
        nn.Linear(in_features, 512), 
        nn.ReLU(inplace=True), 
        nn.Dropout(p=0.3), 
        nn.Linear(512, num_classes) 
    ) 
    return model 

model = build_resnet50(num_classes, freeze_backbone=True).to(DEVICE) 

# Only optimise the head parameters initially 
optimizer = optim.Adam( 
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LR, weight_decay=1e-4 
) 
criterion = nn.CrossEntropyLoss() 
scheduler = StepLR(optimizer, step_size=7, gamma=0.1) 

# ── Training Loop ─────────────────────────────────────────────── 
def train_epoch(model, loader, optimizer, criterion): 
    model.train() 
    total_loss, correct, total = 0.0, 0, 0 
    for imgs, labels in loader: 
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE) 
        optimizer.zero_grad() 
        outputs = model(imgs) 
        loss    = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 
        total_loss += loss.item() * imgs.size(0) 
        _, preds = outputs.max(1) 
        correct  += preds.eq(labels).sum().item() 
        total    += labels.size(0) 
    return total_loss / total, 100.0 * correct / total 

def evaluate(model, loader, criterion): 
    model.eval() 
    total_loss, correct, total = 0.0, 0, 0 
    with torch.no_grad(): 
        for imgs, labels in loader: 
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE) 
            outputs = model(imgs) 
            loss    = criterion(outputs, labels) 
            total_loss += loss.item() * imgs.size(0) 
            _, preds = outputs.max(1) 
            correct  += preds.eq(labels).sum().item() 
            total    += labels.size(0) 
    return total_loss / total, 100.0 * correct / total 

best_val_acc   = 0.0 
train_acc_hist = [] 
val_acc_hist   = [] 

for epoch in range(NUM_EPOCHS): 
    # After epoch 10, unfreeze all layers and fine-tune 
    if epoch == 10: 
        print("\n>>> Unfreezing full backbone for fine-tuning...") 
        for param in model.parameters(): 
            param.requires_grad = True 
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4) 
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5) 

    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion) 
    vl_loss, vl_acc = evaluate(model, val_loader, criterion) 
    scheduler.step() 

    train_acc_hist.append(tr_acc) 
    val_acc_hist.append(vl_acc) 

    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | " 
        f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}% | " 
        f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.2f}%") 

    if vl_acc > best_val_acc: 
        best_val_acc = vl_acc 
        torch.save(model.state_dict(), MODEL_PATH) 
        print(f"  Saved best model (val_acc={vl_acc:.2f}%)") 

# ── Final Evaluation on Test Set ──────────────────────────────── 
model.load_state_dict(torch.load(MODEL_PATH)) 
model.eval() 
all_preds, all_labels = [], [] 
with torch.no_grad(): 
    for imgs, labels in test_loader: 
        outputs = model(imgs.to(DEVICE)) 
        _, preds = outputs.max(1) 
        all_preds.extend(preds.cpu().numpy()) 
        all_labels.extend(labels.numpy()) 

print("\n=== Test Set Classification Report ===") 
print(classification_report(all_labels, all_preds, target_names=class_names)) 

# ── Plot Learning Curves ───────────────────────────────────────── 
plt.figure(figsize=(10, 4)) 
plt.plot(train_acc_hist, label="Train Accuracy", color="green") 
plt.plot(val_acc_hist,   label="Val Accuracy",   color="blue") 
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)") 
plt.title("Training Curves — ResNet-50 on PlantVillage") 
plt.legend(); plt.grid(True) 
plt.savefig("training_curves.png", dpi=150) 
print("Saved: training_curves.png") 