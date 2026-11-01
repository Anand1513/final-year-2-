# Project Summary: Crop Recommendation via Machine Learning

## 1. Overview
The "Crop Recommendation" project is a B.Tech Final Year Project designed to assist farmers and agricultural professionals in selecting the optimal crop for specific environmental and soil conditions. It transitions a 7-parameter environmental model into a practical agricultural decision-support interface, utilizing Machine Learning algorithms.

## 2. Problem Statement
Crop selection is heavily dependent on localized agricultural conditions. Relying on generalized advice or intuition can lead to suboptimal yields. This project solves this by using data-driven classification algorithms to analyze soil nutrients and weather conditions, subsequently recommending the historically most successful crop.

## 3. Key Features
- **Basic Prediction:** Recommends crops based on 7 exact user inputs: Nitrogen (N), Phosphorous (P), Potassium (K), Temperature, Humidity, pH, and Rainfall.
- **Regional Prediction:** Recommends crops based on State, District, Season, and basic soil nutrients. It intelligently looks up missing agroclimatic and rainfall data from pre-processed geographical datasets (`Agroclimatic_regions.shp`, `district wise rainfall normal.csv`).
- **Hybrid Prediction:** Leverages historical feature datasets and integrates with the **Open-Meteo API** to fetch real-time weather (temperature and humidity) for a specific district, enhancing the accuracy of the prediction.

## 4. Technology Stack
- **Backend:** Python, FastAPI (for exposing prediction endpoints).
- **Machine Learning:** Scikit-Learn (Random Forest classifiers), Pandas, NumPy, GeoPandas.
- **Model Files:** Uses pre-trained serialized models (`.pkl` files) such as `crop_recommendation_rf_model.pkl` and `regional_rf_model.pkl`.
- **Frontend:** HTML5, Vanilla CSS (with responsive design and animations), and JavaScript. 
- **Integrations:** Open-Meteo API for live weather geocoding and forecasting.

## 5. System Architecture
- **API Endpoints:** 
  - `POST /predict/`: Handles basic 7-parameter predictions.
  - `POST /predict_regional/`: Handles district/season-based predictions.
  - `POST /predict_hybrid/`: Handles predictions by fetching live weather for the provided location.
- **User Interface:** A bespoke, responsive web page (`index.html`) featuring a central timeline, slideshow, and dedicated sections for project overview, research, and prototype interaction.

## 6. Directory Structure Highlights
- `main.py`: The core backend API server built with FastAPI.
- `index.html` / `style.css` / `index.js`: The frontend user interface.
- `model/`, `model1/`, `model2/`: Directories containing ML training scripts, raw/processed data, and serialized models.
- `utils/`: Contains helper scripts like `explainer.py` (for SHAP visualizations) and prediction utility functions.
