import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from model3.src.weather_service import fetch_multi_horizon_weather

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load Artifacts globally for high-speed sub-millisecond inference
with open(os.path.join(MODELS_DIR, 'crop_recommender_model3.pkl'), 'rb') as f:
    MODEL = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'model3_mlb.pkl'), 'rb') as f:
    MLB = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'model3_encoders.pkl'), 'rb') as f:
    ENCODERS = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'model3_metrics.json'), 'r') as f:
    METRICS = json.load(f)

with open(os.path.join(DATA_DIR, 'district_profiles.json'), 'r') as f:
    DISTRICT_PROFILES = json.load(f)

with open(os.path.join(DATA_DIR, 'crop_metadata.json'), 'r') as f:
    CROP_METADATA = json.load(f)

with open(os.path.join(DATA_DIR, 'district_season_crops.json'), 'r') as f:
    STRATA_CROPS = json.load(f)

TAU_VIABLE = METRICS.get("viability_threshold", 0.28)

def encode_value(le, val: str) -> int:
    """Safely encode categorical value with unknown fallback."""
    val_clean = str(val).strip().upper()
    if val_clean in le.classes_:
        return int(le.transform([val_clean])[0])
    if 'UNKNOWN' in le.classes_:
        return int(le.transform(['UNKNOWN'])[0])
    return 0

def evaluate_agronomic_feasibility(
    state: str,
    district: str,
    season: str,
    current_temp: float,
    current_humidity: float,
    upcoming_rain_mm: float,
    available_seasons: List[str]
) -> Tuple[bool, str]:
    """
    Check physical and agronomic feasibility boundaries:
    - Extreme frost (< 3.0°C)
    - Extreme thermal heatwave (> 48.0°C)
    - Off-season detection
    """
    # 1. Thermal physiological boundaries
    if current_temp < 3.0:
        return False, f"Extreme cold / frost condition ({current_temp:.1f}°C) prevents open-field crop cultivation in this region."
        
    if current_temp > 48.0:
        return False, f"Severe thermal heatwave ({current_temp:.1f}°C) exceeds the cellular physiological threshold for crop growth."
        
    # 2. Season availability check
    season_clean = season.strip().upper()
    if available_seasons and season_clean not in available_seasons:
        # Check if season is traditionally uncultivated in this district
        if season_clean in ['WINTER', 'SUMMER', 'AUTUMN']:
            return False, f"No agricultural cultivation cycle is recorded in {district.title()} during '{season.title()}' season. Farmers typically cultivate during: {', '.join([s.title() for s in available_seasons])}."
            
    return True, "Agronomically viable"

def recommend_district_crops(
    state: str,
    district: str,
    season: str,
    custom_soil: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Core Model 3 Recommendation Engine:
    - Automatically fetches live current weather, 7-day upcoming forecast, and IMD historical normals.
    - Evaluates physical and meteorological viability.
    - If viable: returns ALL possible crops ranked by suitability confidence.
    - If not viable: returns status 'NO_CROPS_POSSIBLE' with descriptive agronomic reasons.
    """
    st_clean = state.strip().upper()
    dist_clean = district.strip().upper()
    seas_clean = season.strip().upper()
    dist_key = f"{st_clean}_{dist_clean}"
    
    # 1. Retrieve district profile
    prof = DISTRICT_PROFILES.get(dist_key)
    if not prof:
        # Try finding district across other states if spelling mismatch
        for k, p in DISTRICT_PROFILES.items():
            if p['district'] == dist_clean:
                prof = p
                st_clean = p['state']
                dist_key = k
                break
                
    if not prof:
        # Fallback profile
        prof = {
            'state': st_clean,
            'district': dist_clean,
            'soil': {'NITROGEN': 60.0, 'PHOSPHOROUS': 45.0, 'POTASSIUM': 45.0, 'PH': 6.5},
            'rainfall': {'ANNUAL': 1100.0, 'Jan-Feb': 35.0, 'Mar-May': 120.0, 'Jun-Sep': 750.0, 'Oct-Dec': 120.0},
            'agroclimatic': {'avgtmp_jan': 18.0, 'avgtmp_jul': 30.0, 'avgann_rf': 1100.0},
            'available_seasons': ['KHARIF', 'RABI', 'WHOLE YEAR']
        }
        
    # 2. Automated Multi-Horizon Weather Fetching via API
    weather_data = fetch_multi_horizon_weather(st_clean, dist_clean, seas_clean, historical_normals=prof)
    
    curr_weather = weather_data.get("current", {})
    forecast = weather_data.get("upcoming_forecast", {})
    normals = weather_data.get("historical_normals", {})
    
    current_temp = float(curr_weather.get("temperature", 27.0))
    current_hum = float(curr_weather.get("relative_humidity", 65.0))
    upcoming_rain = float(forecast.get("expected_rainfall_sum_mm", 0.0))
    available_seasons = prof.get("available_seasons", [])
    
    # 3. Agronomic Feasibility Gate
    is_feasible, feasibility_reason = evaluate_agronomic_feasibility(
        st_clean, dist_clean, seas_clean, current_temp, current_hum, upcoming_rain, available_seasons
    )
    
    # 4. Resolve Soil Nutrient Values
    soil_data = prof.get('soil', {})
    n_val = float(custom_soil.get('nitrogen', soil_data.get('NITROGEN', 60.0))) if custom_soil else float(soil_data.get('NITROGEN', 60.0))
    p_val = float(custom_soil.get('phosphorous', soil_data.get('PHOSPHOROUS', 45.0))) if custom_soil else float(soil_data.get('PHOSPHOROUS', 45.0))
    k_val = float(custom_soil.get('potassium', soil_data.get('POTASSIUM', 45.0))) if custom_soil else float(soil_data.get('POTASSIUM', 45.0))
    ph_val = float(custom_soil.get('ph', soil_data.get('PH', 6.5))) if custom_soil else float(soil_data.get('PH', 6.5))
    
    soil_summary = {
        "nitrogen": round(n_val, 1),
        "phosphorous": round(p_val, 1),
        "potassium": round(k_val, 1),
        "ph": round(ph_val, 1),
        "source": "Farmer Custom Input" if custom_soil else "ICAR District Soil Health Profile"
    }
    
    if not is_feasible:
        return {
            "status": "NO_CROPS_POSSIBLE",
            "message": f"No crops possible in {district.title()} for '{season.title()}' season.",
            "reason": feasibility_reason,
            "state": st_clean.title(),
            "district": dist_clean.title(),
            "season": seas_clean.title(),
            "weather_report": weather_data,
            "soil_profile": soil_summary,
            "total_crops_possible": 0,
            "crops": []
        }
        
    # 5. Build ML Feature Vector
    state_enc = encode_value(ENCODERS['state'], st_clean)
    dist_enc = encode_value(ENCODERS['district'], dist_clean)
    seas_enc = encode_value(ENCODERS['season'], seas_clean)
    
    rf_data = prof.get('rainfall', {})
    ann_rf = float(rf_data.get('ANNUAL', 1100.0))
    
    if 'KHARIF' in seas_clean:
        seasonal_rf = float(rf_data.get('Jun-Sep', 750.0))
    elif 'RABI' in seas_clean:
        seasonal_rf = float(rf_data.get('Jan-Feb', 35.0)) + float(rf_data.get('Oct-Dec', 90.0))
    elif 'SUMMER' in seas_clean:
        seasonal_rf = float(rf_data.get('Mar-May', 110.0))
    else:
        seasonal_rf = ann_rf
        
    agro_data = prof.get('agroclimatic', {})
    temp_jan = float(agro_data.get('avgtmp_jan', 18.0))
    temp_jul = float(agro_data.get('avgtmp_jul', 30.0))
    
    feat_vector = np.array([[
        state_enc, dist_enc, seas_enc,
        n_val, p_val, k_val, ph_val,
        ann_rf, seasonal_rf, temp_jan, temp_jul
    ]])
    
    # 6. Multi-Label Model Inference
    all_tree_probs = MODEL.predict_proba(feat_vector)
    # Shape of probs: array of shape (1, num_crops)
    crop_probs = np.array([tree_prob[0, 1] if tree_prob.shape[1] > 1 else 0.0 for tree_prob in all_tree_probs])
    
    classes = MLB.classes_
    
    # Check historical baseline crops for this stratum
    stratum_key = f"{st_clean}|{dist_clean}|{seas_clean}"
    historical_strata_crops = set(STRATA_CROPS.get(stratum_key, []))
    
    viable_candidates = []
    for idx, crop_name in enumerate(classes):
        prob = float(crop_probs[idx])
        is_historical = crop_name in historical_strata_crops
        
        # If crop is historically cultivated in this stratum, give baseline boost
        if is_historical and prob < TAU_VIABLE:
            prob = max(prob, 0.45)
            
        if prob >= TAU_VIABLE:
            meta = CROP_METADATA.get(crop_name, {
                'crop_name': crop_name.title(),
                'category': 'FIELD CROP',
                'water_requirement': 'MEDIUM'
            })
            
            # Weather penalty/bonus adjustment
            # If high water crop (e.g. Rice) and upcoming rain is 0 and dry season -> slight penalty
            if meta['water_requirement'] == 'HIGH' and seasonal_rf < 300.0 and upcoming_rain < 5.0:
                prob *= 0.85
                
            conf_percent = min(round(prob * 100.0, 1), 99.5)
            
            # Suitability tier
            if conf_percent >= 80.0:
                tier = "Highly Recommended"
            elif conf_percent >= 60.0:
                tier = "Moderately Recommended"
            else:
                tier = "Feasible with Care"
                
            viable_candidates.append({
                "crop": meta['crop_name'],
                "confidence": conf_percent,
                "category": meta['category'],
                "water_requirement": meta['water_requirement'],
                "suitability": tier,
                "historically_validated": is_historical
            })
            
    # Sort descending by confidence
    viable_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    
    if len(viable_candidates) == 0:
        return {
            "status": "NO_CROPS_POSSIBLE",
            "message": f"No crops possible in {district.title()} under current environmental and seasonal conditions.",
            "reason": "Model evaluated all 84 agricultural crop varieties; none met the minimum viability confidence threshold.",
            "state": st_clean.title(),
            "district": dist_clean.title(),
            "season": seas_clean.title(),
            "weather_report": weather_data,
            "soil_profile": soil_summary,
            "total_crops_possible": 0,
            "crops": []
        }
        
    return {
        "status": "CROPS_AVAILABLE",
        "message": f"Identified {len(viable_candidates)} viable crop(s) for {district.title()}, {state.title()} in {season.title()} season.",
        "state": st_clean.title(),
        "district": dist_clean.title(),
        "season": seas_clean.title(),
        "weather_report": weather_data,
        "soil_profile": soil_summary,
        "total_crops_possible": len(viable_candidates),
        "crops": viable_candidates
    }

if __name__ == "__main__":
    print("Testing Recommender:")
    print("-" * 50)
    
    # Test 1: Ludhiana, Punjab, Kharif
    res1 = recommend_district_crops("Punjab", "Ludhiana", "Kharif")
    print(f"Test 1 (Ludhiana Kharif): Status = {res1['status']}, Crops Found = {res1['total_crops_possible']}")
    print("Top 5 Crops:", [c['crop'] for c in res1['crops'][:5]])
    
    # Test 2: Ludhiana, Punjab, Rabi
    res2 = recommend_district_crops("Punjab", "Ludhiana", "Rabi")
    print(f"Test 2 (Ludhiana Rabi): Status = {res2['status']}, Crops Found = {res2['total_crops_possible']}")
    print("Top 5 Crops:", [c['crop'] for c in res2['crops'][:5]])
    
    # Test 3: Off-season / Leh Winter
    res3 = recommend_district_crops("Jammu and Kashmir", "Leh Ladakh", "Winter")
    print(f"Test 3 (Leh Winter): Status = {res3['status']}, Reason = {res3.get('reason')}")
