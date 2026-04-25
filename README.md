# Crop Disease Detection System 🌾

A machine learning-powered web application for detecting crop diseases using computer vision and predictive analytics. The system combines **deep learning (CNN)** for disease classification with **XGBoost** for crop yield prediction.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Training](#training)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project provides an intelligent solution for early detection and management of crop diseases. Farmers and agricultural professionals can upload leaf images to receive:

1. **Disease Classification** - Identification of specific crop diseases using a ResNet-50 CNN model
2. **Yield Prediction** - Estimated crop yield impact using XGBoost
3. **Treatment Recommendations** - Database-driven treatment suggestions
4. **Visual Explanations** - Grad-CAM visualizations showing which regions influenced the prediction

The system supports multiple crop types:
- 🌶️ **Pepper** (Bacterial spot, Healthy)
- 🥔 **Potato** (Early blight, Late blight, Healthy)
- 🍅 **Tomato** (15+ disease categories and healthy)

---

## ✨ Features

- ✅ **Real-time Disease Detection** - Upload leaf images for instant classification
- ✅ **Hybrid AI Pipeline** - CNN + XGBoost for comprehensive analysis
- ✅ **Treatment Database** - Curated treatment recommendations for each disease
- ✅ **Model Interpretability** - Grad-CAM visualizations for model transparency
- ✅ **REST API** - FastAPI backend with CORS support for easy integration
- ✅ **Modern UI** - React + Vite frontend with responsive design
- ✅ **GPU Support** - CUDA-enabled training and inference
- ✅ **Pre-trained Models** - Ready-to-use CNN and XGBoost models included

---

## 📁 Project Structure

```
Project-1/
├── data/
│   └── PlantVillage/              # PlantVillage dataset
│       ├── Pepper__bell___*/      # Pepper disease images
│       ├── Potato___*/            # Potato disease images
│       ├── Tomato__*/             # Tomato disease images
│       └── PlantVillage/          # Complete dataset mirror
├── frontend/                       # React + Vite web application
│   ├── src/
│   │   ├── App.jsx               # Main React component
│   │   ├── main.jsx              # Entry point
│   │   ├── App.css               # Styling
│   │   └── assets/               # Static assets
│   ├── package.json              # Frontend dependencies
│   ├── vite.config.js            # Vite configuration
│   └── index.html                # HTML template
├── models/                         # Pre-trained models
│   ├── best_cnn.pth              # ResNet-50 CNN weights
│   └── class_names.json          # Disease class mapping
├── src/                            # Backend Python source code
│   ├── main.py                   # FastAPI server
│   ├── train_cnn.py              # CNN training script
│   ├── train_yield.py            # XGBoost yield prediction training
│   ├── gradcam.py                # Grad-CAM visualization
│   ├── treatment_db.py           # Treatment recommendations
│   └── __init__.py               # Package initialization
├── myenv/                          # Python virtual environment
├── requirements.txt               # Python dependencies
└── README.md                       # This file
```

---

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI 0.104.0
- **Deep Learning**: PyTorch 2.0.0, TorchVision 0.15.0
- **Machine Learning**: XGBoost 2.0.0, Scikit-learn 1.3.0
- **Interpretability**: SHAP 0.44.0, Grad-CAM 1.4.8
- **Server**: Uvicorn 0.24.0

### Frontend
- **Framework**: React 19.2.5
- **Build Tool**: Vite 8.0.10
- **Linting**: ESLint 10.2.1

### Data & ML
- **Dataset**: PlantVillage (organized disease images)
- **Model Architecture**: ResNet-50 (pre-trained on ImageNet)
- **Data Processing**: Pillow, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

---

## 📦 Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 16.x or higher (for frontend)
- **pip**: Python package manager
- **Virtual Environment**: Recommended (venv or conda)
- **GPU** (Optional): CUDA-capable GPU for faster training
- **4GB+ RAM**: Minimum for model inference

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Project-1
```

### 2. Set Up Python Environment

#### Using venv (Recommended)
```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```

#### Using conda
```bash
conda create -n crop-disease python=3.10
conda activate crop-disease
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Frontend
```bash
cd frontend
npm install
cd ..
```

### 5. Download/Prepare Dataset

Ensure the PlantVillage dataset is in the `data/PlantVillage/` directory with the following structure:
```
data/PlantVillage/
├── Pepper__bell___Bacterial_spot/
├── Pepper__bell___healthy/
├── Potato___Early_blight/
├── Potato___healthy/
├── Potato___Late_blight/
├── Tomato__Target_Spot/
├── Tomato__Tomato_mosaic_virus/
├── ... (other disease categories)
```

[Download PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)

---

## ⚙️ Configuration

### Backend Configuration (`src/main.py`)

Default settings:
- **Port**: 8000
- **Host**: 127.0.0.1
- **CORS**: Enabled for all origins (development)

To configure for production, modify the CORS settings in `src/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict to your domain
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Model Configuration (`src/train_cnn.py`)

```python
BATCH_SIZE   = 32      # Batch size for training
NUM_EPOCHS   = 20      # Number of training epochs
LR           = 1e-3    # Learning rate
DEVICE       = "cuda"  # Use "cpu" if GPU not available
```

---

## 💻 Usage

### Start Backend Server

```bash
# Activate virtual environment first
myenv\Scripts\activate

# Run FastAPI server
uvicorn src.main:app --reload --port 8000
```

The API will be available at: `http://localhost:8000`

Swagger UI: `http://localhost:8000/docs`

### Start Frontend Development Server

```bash
cd frontend
npm run dev
```

Frontend will be available at: `http://localhost:5173`

### Production Build

#### Frontend
```bash
cd frontend
npm run build
# Output: dist/ folder
```

#### Backend
```bash
uvicorn src.main:app --port 8000 --workers 4
```

---

## 📡 API Documentation

### Health Check
```http
GET /
```
Returns API status and version information.

### Disease Prediction
```http
POST /predict
Content-Type: multipart/form-data

Parameters:
- file (required): Image file (JPG, PNG)
- include_gradcam (optional): Boolean to include Grad-CAM visualization

Response:
{
  "disease": "Tomato_Early_blight",
  "confidence": 0.9543,
  "all_predictions": {
    "Tomato_Early_blight": 0.9543,
    "Tomato_Late_blight": 0.0345,
    ...
  },
  "gradcam": "base64_encoded_image_or_null"
}
```

### Yield Prediction
```http
POST /predict-yield
Content-Type: application/json

Body:
{
  "disease": "Tomato_Early_blight",
  "severity": 0.8,
  "crop_type": "tomato"
}

Response:
{
  "predicted_yield": 42.5,
  "yield_impact": -35.2,
  "confidence": 0.87
}
```

### Swagger Interactive Docs
Visit `http://localhost:8000/docs` for full interactive API documentation and testing.

---

## 🤖 Model Details

### CNN Model (Disease Classification)

- **Architecture**: ResNet-50
- **Pre-training**: ImageNet weights
- **Input Size**: 224 × 224 RGB images
- **Output Classes**: 38 disease + healthy categories
- **Accuracy**: Training on PlantVillage dataset
- **File**: `models/best_cnn.pth`

**Data Augmentation Pipeline**:
- Random crop (224×224)
- Random horizontal/vertical flips
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization: ImageNet mean/std

### XGBoost Model (Yield Prediction)

- **Purpose**: Predict crop yield impact based on disease detection
- **Input Features**: Disease type, severity score, crop type, weather conditions (temperature, humidity, rainfall), soil properties (pH, nitrogen, phosphorus), and growing stage
- **Output**: Yield prediction and impact percentage
- **Interpretability**: SHAP (SHapley Additive exPlanations) values for model transparency

#### Feature Importance Analysis

The XGBoost model uses the following key features ranked by impact on yield prediction:

**Top Impact Features**:
1. **Disease Severity** - Strongest negative impact on yield
2. **Temperature** - Significant positive/negative influence depending on value
3. **Days Since Sowing** - Affects crop developmental stage
4. **Humidity** - Important environmental factor
5. **Nitrogen (kg/ha)** - Critical nutrient availability

#### SHAP Visualizations

**Force Plot - Individual Prediction Explanation**:
![SHAP Force Plot](docs/shap_force_plot.png)
*Shows how individual features contribute to a specific yield prediction. Red bars indicate negative impact, blue bars indicate positive impact.*

**Summary Plot - Global Feature Importance**:
![SHAP Summary Plot](docs/shap_summary_plot.png)
*Displays the overall importance of each feature across all predictions, showing both magnitude and distribution of impacts.*

**Model Insights**:
- Disease severity has the most substantial negative correlation with yield
- Temperature variations significantly influence predictions (both positive and negative)
- Early intervention can mitigate yield loss when disease is detected
- Environmental and soil conditions play supporting roles in yield determination

---

## 🎓 Training

### Train CNN Model

```bash
python src/train_cnn.py
```

**Parameters** (edit in `train_cnn.py`):
- `DATA_DIR`: Path to PlantVillage dataset
- `BATCH_SIZE`: Training batch size (default: 32)
- `NUM_EPOCHS`: Number of epochs (default: 20)
- `LR`: Learning rate (default: 1e-3)

**Output**:
- `models/best_cnn.pth` - Best model weights
- `models/class_names.json` - Class mapping
- Training and validation accuracy plots

### Train XGBoost Model

```bash
python src/train_yield.py
```

**Output**:
- Trained yield prediction model
- Feature importance analysis

---

## 🧪 Development

### Running Tests

```bash
# Backend tests (if available)
pytest src/tests/

# Frontend tests
cd frontend
npm test
```

### Code Quality

#### Backend
```bash
# Format with black
black src/

# Lint with pylint
pylint src/
```

#### Frontend
```bash
cd frontend
npm run lint
npm run lint -- --fix
```

### Adding New Features

1. **New Disease Category**:
   - Add images to `data/PlantVillage/<disease_name>/`
   - Retrain: `python src/train_cnn.py`
   - Update class_names.json automatically

2. **New API Endpoint**:
   - Edit `src/main.py`
   - Restart server with `--reload` flag

3. **Frontend Component**:
   - Create component in `frontend/src/`
   - Import and use in `App.jsx`
   - Run `npm run dev` for hot reload

---

## 🔍 Troubleshooting

### Common Issues

**Issue**: "CUDA out of memory"
```bash
# Solution: Reduce batch size in train_cnn.py
BATCH_SIZE = 16  # Instead of 32
```

**Issue**: "Models not found"
```bash
# Solution: Ensure best_cnn.pth exists in models/ folder
# If not, run training: python src/train_cnn.py
```

**Issue**: "Port 8000 already in use"
```bash
# Solution: Use different port
uvicorn src.main:app --port 8001
```

**Issue**: Frontend can't reach backend
```bash
# Ensure backend is running on http://localhost:8000
# Check CORS settings in src/main.py
```

---

## 📊 Performance Metrics

- **Disease Classification**: ~95% accuracy on PlantVillage dataset
- **Inference Time**: ~500ms per image (CPU), ~50ms (GPU)
- **Model Size**: ~100MB (CNN only)
- **API Response Time**: <1s end-to-end

---


## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Authors

- **Saral Jain** - Initial work and development

---

## 🔗 Useful Links

- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)

---
