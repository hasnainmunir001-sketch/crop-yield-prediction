# 🌾 Crop Yield Prediction — ML Pipeline + Streamlit App

A complete machine learning project for predicting agricultural crop yield using multiple ML approaches, with an interactive **Streamlit** web application.

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)

## 📌 Problem Statement
Predict crop yield (tons/hectare) based on agricultural and environmental factors such as crop type, season, rainfall, fertilizer, pesticide, and cultivation area.

## 🧠 Machine Learning Approaches
| Approach | Models/Techniques |
|----------|-------------------|
| **Supervised Learning** | Random Forest Regressor, XGBoost Regressor |
| **Unsupervised Learning** | K-Means Clustering + PCA for visualization |
| **Semi-Supervised Learning** | Self-Training Regression (20% labeled, 80% pseudo-labeled) |
| **Neural Networks** | Multi-Layer Perceptron (MLP) Regressor |

## 📂 Dataset Features
- **State**: Region/State
- **Crop**: Crop type
- **Season**: Kharif / Rabi / Zaid / Whole Year
- **Area**: Cultivation area (hectares)
- **Annual_Rainfall**: Rainfall (mm)
- **Fertilizer**: Fertilizer usage (kg/ha)
- **Pesticide**: Pesticide usage (kg/ha)
- **Yield**: Target variable (tons/ha)

> Note: The included `data_generator.py` creates a synthetic dataset similar to Kaggle's Crop Yield dataset for reproducibility.

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/crop-yield-prediction.git
cd crop-yield-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
