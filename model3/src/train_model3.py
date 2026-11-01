import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    f1_score,
    precision_score,
    recall_score
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
EVAL_DIR = os.path.join(BASE_DIR, 'evaluation')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

def train_model3():
    print("=" * 60)
    print("🚀 TRAINING AGRISENSE MODEL 3: DISTRICT-AWARE MULTI-LABEL ENGINE")
    print("=" * 60)
    
    # 1. Load Knowledge Base
    with open(os.path.join(DATA_DIR, 'district_profiles.json'), 'r') as f:
        district_profiles = json.load(f)
        
    with open(os.path.join(DATA_DIR, 'district_season_crops.json'), 'r') as f:
        strata_crops = json.load(f)
        
    with open(os.path.join(DATA_DIR, 'valid_crop_list.json'), 'r') as f:
        valid_crops = json.load(f)
        
    print(f"Loaded {len(district_profiles)} district profiles, {len(strata_crops)} positive strata, and {len(valid_crops)} target crops.")
    
    # 2. Build Multi-Label Dataset (Positive Strata)
    rows = []
    crop_targets = []
    
    for stratum_key, crops in strata_crops.items():
        parts = stratum_key.split('|')
        if len(parts) != 3:
            continue
        state, district, season = parts[0], parts[1], parts[2]
        dist_key = f"{state}_{district}"
        prof = district_profiles.get(dist_key, {})
        
        soil = prof.get('soil', {'NITROGEN': 60.0, 'PHOSPHOROUS': 45.0, 'POTASSIUM': 45.0, 'PH': 6.5})
        rf = prof.get('rainfall', {'ANNUAL': 1100.0, 'Jan-Feb': 35.0, 'Mar-May': 120.0, 'Jun-Sep': 750.0, 'Oct-Dec': 120.0})
        agro = prof.get('agroclimatic', {'avgtmp_jan': 18.0, 'avgtmp_jul': 30.0, 'avgann_rf': 1100.0})
        
        # Calculate seasonal rainfall normal
        s_clean = season.upper()
        if 'KHARIF' in s_clean:
            seasonal_rf = rf.get('Jun-Sep', 750.0)
        elif 'RABI' in s_clean:
            seasonal_rf = rf.get('Jan-Feb', 35.0) + rf.get('Oct-Dec', 90.0)
        elif 'SUMMER' in s_clean:
            seasonal_rf = rf.get('Mar-May', 110.0)
        else:
            seasonal_rf = rf.get('ANNUAL', 1100.0)
            
        rows.append({
            'state': state,
            'district': district,
            'season': season,
            'nitrogen': float(soil.get('NITROGEN', 60.0)),
            'phosphorous': float(soil.get('PHOSPHOROUS', 45.0)),
            'potassium': float(soil.get('POTASSIUM', 45.0)),
            'ph': float(soil.get('PH', 6.5)),
            'annual_rf': float(rf.get('ANNUAL', 1100.0)),
            'seasonal_rf': float(seasonal_rf),
            'temp_jan': float(agro.get('avgtmp_jan', 18.0)),
            'temp_jul': float(agro.get('avgtmp_jul', 30.0))
        })
        crop_targets.append([c for c in crops if c in valid_crops])
        
    pos_count = len(rows)
    print(f"Generated {pos_count} positive district-season training records.")
    
    # 3. Synthesize Negative Samples (Zero-Crop Rejection Boundary)
    # This enables the model to explicitly output NO CROPS when conditions are extreme or off-season
    np.random.seed(42)
    neg_seasons = ['WINTER', 'SUMMER', 'AUTUMN']
    for st_dist_key, prof in district_profiles.items():
        st = prof['state']
        dist = prof['district']
        active_seasons = prof.get('available_seasons', [])
        
        # Find seasons with NO documented cultivation in this district
        for s in neg_seasons:
            if s not in active_seasons and np.random.rand() > 0.4:
                soil = prof.get('soil', {'NITROGEN': 60.0, 'PHOSPHOROUS': 45.0, 'POTASSIUM': 45.0, 'PH': 6.5})
                rf = prof.get('rainfall', {'ANNUAL': 1100.0, 'Jan-Feb': 35.0, 'Mar-May': 120.0, 'Jun-Sep': 750.0, 'Oct-Dec': 120.0})
                agro = prof.get('agroclimatic', {'avgtmp_jan': 18.0, 'avgtmp_jul': 30.0, 'avgann_rf': 1100.0})
                
                rows.append({
                    'state': st,
                    'district': dist,
                    'season': s,
                    'nitrogen': float(soil.get('NITROGEN', 60.0)),
                    'phosphorous': float(soil.get('PHOSPHOROUS', 45.0)),
                    'potassium': float(soil.get('POTASSIUM', 45.0)),
                    'ph': float(soil.get('PH', 6.5)),
                    'annual_rf': float(rf.get('ANNUAL', 1100.0)),
                    'seasonal_rf': 0.0,
                    'temp_jan': float(agro.get('avgtmp_jan', 18.0)),
                    'temp_jul': float(agro.get('avgtmp_jul', 30.0))
                })
                crop_targets.append([]) # EMPTY label vector: No crops possible
                
    # Also inject extreme thermal shock samples (sub-zero or >48C heatwave)
    for _ in range(250):
        rows.append({
            'state': 'EXTREME_CLIMATE',
            'district': 'ARID_DESERT',
            'season': 'SUMMER',
            'nitrogen': 10.0,
            'phosphorous': 5.0,
            'potassium': 10.0,
            'ph': 9.2,
            'annual_rf': 50.0,
            'seasonal_rf': 0.0,
            'temp_jan': 45.0,
            'temp_jul': 50.0
        })
        crop_targets.append([]) # No crops possible
        
    print(f"Total training samples after injecting zero-crop negative boundaries: {len(rows)} (Pos: {pos_count}, Neg: {len(rows)-pos_count})")
    
    df_train = pd.DataFrame(rows)
    
    # 4. MultiLabel Binarizer
    mlb = MultiLabelBinarizer(classes=sorted(list(valid_crops)))
    Y = mlb.fit_transform(crop_targets)
    
    # 5. Categorical Feature Encoders
    encoders = {}
    for col in ['state', 'district', 'season']:
        le = LabelEncoder()
        # Add unknown category
        unique_vals = list(df_train[col].unique()) + ['UNKNOWN']
        le.fit(unique_vals)
        df_train[col + '_enc'] = le.transform(df_train[col])
        encoders[col] = le
        
    feature_cols = [
        'state_enc', 'district_enc', 'season_enc',
        'nitrogen', 'phosphorous', 'potassium', 'ph',
        'annual_rf', 'seasonal_rf', 'temp_jan', 'temp_jul'
    ]
    X = df_train[feature_cols].values
    
    # 6. Train-Test Split (80% Train, 20% Holdout Test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    
    print(f"Training multi-label Random Forest ensemble on {len(X_train)} samples across {Y.shape[1]} crops...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=22,
        min_samples_split=3,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, Y_train)
    
    # 7. Evaluate on Unseen Test Set
    Y_pred_raw = rf.predict(X_test)
    
    # Calculate Probabilities for fine-grained ranking
    # In multi-label RF, predict_proba returns list of length C with (n_samples, 2)
    # We calibrate decision threshold tau = 0.30
    TAU_VIABLE = 0.28
    
    # Compute probabilities across all classes
    probs = np.array([tree_probs[:, 1] for tree_probs in rf.predict_proba(X_test)]).T
    Y_pred_calibrated = (probs >= TAU_VIABLE).astype(int)
    
    # Handle zero-crop detection explicitly: if max prob < TAU_VIABLE, pred is all zeros
    for i in range(len(probs)):
        if np.max(probs[i]) < TAU_VIABLE:
            Y_pred_calibrated[i] = 0
            
    # Compute Metrics
    exact_match = accuracy_score(Y_test, Y_pred_calibrated)
    h_loss = hamming_loss(Y_test, Y_pred_calibrated)
    micro_f1 = f1_score(Y_test, Y_pred_calibrated, average='micro', zero_division=0)
    macro_f1 = f1_score(Y_test, Y_pred_calibrated, average='macro', zero_division=0)
    micro_prec = precision_score(Y_test, Y_pred_calibrated, average='micro', zero_division=0)
    micro_rec = recall_score(Y_test, Y_pred_calibrated, average='micro', zero_division=0)
    
    # Check Negative / Zero-Crop Accuracy (True Negative Rate for infeasible scenarios)
    neg_mask = (Y_test.sum(axis=1) == 0)
    if neg_mask.sum() > 0:
        zero_crop_acc = accuracy_score(Y_test[neg_mask].sum(axis=1) == 0, Y_pred_calibrated[neg_mask].sum(axis=1) == 0)
    else:
        zero_crop_acc = 1.0
        
    print("\n" + "=" * 50)
    print("📊 MODEL 3 EVALUATION METRICS (TEST SET):")
    print("=" * 50)
    print(f"• Exact Match (Subset Accuracy) : {exact_match * 100:.2f}%")
    print(f"• Micro F1-Score                : {micro_f1 * 100:.2f}%")
    print(f"• Macro F1-Score                : {macro_f1 * 100:.2f}%")
    print(f"• Micro Precision               : {micro_prec * 100:.2f}%")
    print(f"• Micro Recall                  : {micro_rec * 100:.2f}%")
    print(f"• Hamming Loss (Lower is better): {h_loss:.4f}")
    print(f"• Zero-Crop Rejection Accuracy  : {zero_crop_acc * 100:.2f}%")
    print("=" * 50)
    
    metrics_summary = {
        "model_name": "AgriSense Model 3 Multi-Label Random Forest",
        "total_crops": int(Y.shape[1]),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "exact_match_ratio": round(float(exact_match), 4),
        "micro_f1": round(float(micro_f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "micro_precision": round(float(micro_prec), 4),
        "micro_recall": round(float(micro_rec), 4),
        "hamming_loss": round(float(h_loss), 4),
        "zero_crop_accuracy": round(float(zero_crop_acc), 4),
        "viability_threshold": TAU_VIABLE,
        "feature_columns": feature_cols
    }
    
    # 8. Export Model Artifacts
    print("\nExporting Model 3 Artifacts...")
    with open(os.path.join(MODELS_DIR, 'crop_recommender_model3.pkl'), 'wb') as f:
        pickle.dump(rf, f)
        
    with open(os.path.join(MODELS_DIR, 'model3_mlb.pkl'), 'wb') as f:
        pickle.dump(mlb, f)
        
    with open(os.path.join(MODELS_DIR, 'model3_encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
        
    with open(os.path.join(MODELS_DIR, 'model3_metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
        
    with open(os.path.join(EVAL_DIR, 'model3_metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
        
    print("✅ All Model 3 artifacts successfully exported to model3/models/!")
    return rf, mlb, encoders, metrics_summary

if __name__ == "__main__":
    train_model3()
