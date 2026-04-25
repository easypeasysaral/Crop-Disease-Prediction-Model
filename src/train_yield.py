# src/train_yield.py 
# ───────────────────────────────────────────────────────────────── 
# Trains XGBoost yield loss predictor and generates SHAP plots. 
# ───────────────────────────────────────────────────────────────── 

import numpy as np 
import pandas as pd 
import joblib, os 
import matplotlib.pyplot as plt 
import shap 
from xgboost import XGBRegressor 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 

os.makedirs("models", exist_ok=True) 

# ── Synthetic Dataset Generation ──────────────────────────────── 
# Domain-realistic relationships between disease severity, 
# weather, soil quality and yield loss. 
# NOTE: Label this as simulated data in your README. 
# Production use would integrate IMD weather API + ICAR soil DB. 

def generate_dataset(n=8000, seed=42): 
    rng = np.random.default_rng(seed) 
    df  = pd.DataFrame({ 
        "disease_severity":    rng.integers(0, 4,  n),       # 0-3 
        "cnn_confidence":      rng.uniform(0.4, 1.0, n), 
        "disease_class":       rng.integers(0, 38, n), 
        "temperature_c":       rng.uniform(15, 42, n), 
        "humidity_pct":        rng.uniform(25, 98, n), 
        "rainfall_mm":         rng.uniform(0, 250, n), 
        "soil_pH":             rng.uniform(4.5, 9.0, n), 
        "nitrogen_kg_ha":      rng.uniform(0, 120, n), 
        "phosphorus_kg_ha":    rng.uniform(0, 80, n), 
        "region":              rng.integers(0, 5, n), 
        "days_since_sowing":   rng.integers(20, 120, n), 
    }) 

    # Yield loss based on domain knowledge: 
    # • Severe disease is the primary driver 
    # • High temp + humidity accelerates fungal spread 
    # • Low nitrogen weakens plant immune response 
    # • Late-stage detection (high day count) compounds loss 

    yield_loss = ( 
        df["disease_severity"] * 10.0 
        + df["cnn_confidence"] * df["disease_severity"] * 3.0 
        + np.where(df["temperature_c"] > 33, 7, 0) 
        + np.where(df["humidity_pct"]  > 80, 5, 0) 
        + np.where(df["soil_pH"] < 5.5,  3, 0) 
        + np.where(df["nitrogen_kg_ha"] < 20, 4, 0) 
        + df["days_since_sowing"] * 0.05 
        + rng.normal(0, 2.5, n) 
    ).clip(0, 85) 

    df["yield_loss_pct"] = yield_loss 
    return df 

df = generate_dataset() 
print("Dataset shape:", df.shape) 
print(df.describe()) 

FEATURE_COLS = [ 
    "disease_severity", "cnn_confidence", "disease_class", 
    "temperature_c", "humidity_pct", "rainfall_mm", 
    "soil_pH", "nitrogen_kg_ha", "phosphorus_kg_ha", 
    "region", "days_since_sowing" 
] 
TARGET = "yield_loss_pct" 

X = df[FEATURE_COLS] 
y = df[TARGET] 

X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42 
) 

# ── Train XGBoost ─────────────────────────────────────────────── 
model = XGBRegressor( 
    n_estimators=300, 
    max_depth=6, 
    learning_rate=0.05, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    min_child_weight=3, 
    reg_alpha=0.1,          # L1 regularisation 
    reg_lambda=1.0,         # L2 regularisation 
    random_state=42, 
    verbosity=1 
) 

model.fit( 
    X_train, y_train, 
    eval_set=[(X_test, y_test)], 
    verbose=50 
) 

# ── Evaluation ────────────────────────────────────────────────── 
y_pred = model.predict(X_test) 
rmse   = np.sqrt(mean_squared_error(y_test, y_pred)) 
mae    = mean_absolute_error(y_test, y_pred) 
r2     = r2_score(y_test, y_pred) 

cv_scores = cross_val_score( 
    model, X, y, cv=5, 
    scoring="neg_root_mean_squared_error" 
) 

print(f"\nTest RMSE : {rmse:.3f}%") 
print(f"Test MAE  : {mae:.3f}%") 
print(f"Test R²   : {r2:.4f}") 
print(f"CV RMSE   : {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}") 

joblib.dump(model, "models/yield_model.pkl") 
print("Saved: models/yield_model.pkl") 

# ── SHAP Analysis ─────────────────────────────────────────────── 
print("\nComputing SHAP values...") 
explainer   = shap.Explainer(model) 
shap_values = explainer(X_test) 

# Global feature importance 
plt.figure(figsize=(10, 6)) 
shap.summary_plot(shap_values, X_test, 
                feature_names=FEATURE_COLS, 
                show=False) 
plt.title("SHAP Summary — Yield Loss Feature Importance") 
plt.tight_layout() 
plt.savefig("shap_summary.png", dpi=150) 
print("Saved: shap_summary.png") 

# Single-prediction explanation (sample index 0) 
plt.figure() 
shap.waterfall_plot(shap_values[0], show=False) 
plt.tight_layout() 
plt.savefig("shap_waterfall.png", dpi=150) 
print("Saved: shap_waterfall.png") 