import os
import re
import json
import numpy as np
import pandas as pd
import geopandas as gpd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'model2', 'data', 'raw'))
PROCESSED_M2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'model2', 'data', 'processed'))

os.makedirs(DATA_DIR, exist_ok=True)

def parse_range(val, is_rain=False):
    """Parse range strings like '05 °C To 22 °C' or '075 cm To 150 cm' into mean float values in mm/degC."""
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip()
    nums = [float(x) for x in re.findall(r'\d+', val_str)]
    if not nums:
        return 0.0
    mean_val = sum(nums) / len(nums)
    if is_rain and 'cm' in val_str.lower():
        mean_val *= 10.0  # Convert cm to mm
    return float(mean_val)

def load_agroclimatic_data():
    """Extract and parse agroclimatic temperature and rainfall normals per state."""
    shp_path = os.path.join(RAW_DIR, 'Agroclimatic_regions', 'Agroclimatic_regions.shp')
    state_agro_map = {}
    try:
        gdf = gpd.read_file(shp_path)
        for _, row in gdf.iterrows():
            states_raw = str(row['state']).split(',')
            jan_t = parse_range(row['avgtmp_jan'], is_rain=False)
            jul_t = parse_range(row['avgtmp_jul'], is_rain=False)
            ann_r = parse_range(row['avgann_rf'], is_rain=True)
            
            for s in states_raw:
                st_clean = s.strip().upper()
                if 'ANDMAN' in st_clean:
                    st_clean = 'ANDAMAN AND NICOBAR ISLANDS'
                if st_clean not in state_agro_map:
                    state_agro_map[st_clean] = {
                        'avgtmp_jan': jan_t,
                        'avgtmp_jul': jul_t,
                        'avgann_rf': ann_r
                    }
    except Exception as e:
        print(f"Warning: Could not read agroclimatic shapefile: {e}")
    return state_agro_map

def load_soil_medians():
    """Extract district-level median Nitrogen, Phosphorus, Potassium, and pH."""
    merged_path = os.path.join(PROCESSED_M2_DIR, 'merged_regional_data.csv')
    soil_lookup = {}
    state_fallbacks = {}
    
    if os.path.exists(merged_path):
        df_merged = pd.read_csv(merged_path, usecols=['State', 'District', 'NITROGEN', 'PHOSPHOROUS', 'POTASSIUM', 'PH'], low_memory=False)
        df_merged['State'] = df_merged['State'].astype(str).str.strip().str.upper()
        df_merged['District'] = df_merged['District'].astype(str).str.strip().str.upper()
        
        # Clean numeric
        for col in ['NITROGEN', 'PHOSPHOROUS', 'POTASSIUM', 'PH']:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
        
        # Drop duplicates at district level
        dist_agg = df_merged.groupby(['State', 'District'])[['NITROGEN', 'PHOSPHOROUS', 'POTASSIUM', 'PH']].median().reset_index()
        state_agg = df_merged.groupby('State')[['NITROGEN', 'PHOSPHOROUS', 'POTASSIUM', 'PH']].median().reset_index()
        
        for _, r in state_agg.iterrows():
            state_fallbacks[r['State']] = {
                'NITROGEN': float(r['NITROGEN']) if not np.isnan(r['NITROGEN']) else 60.0,
                'PHOSPHOROUS': float(r['PHOSPHOROUS']) if not np.isnan(r['PHOSPHOROUS']) else 45.0,
                'POTASSIUM': float(r['POTASSIUM']) if not np.isnan(r['POTASSIUM']) else 45.0,
                'PH': float(r['PH']) if not np.isnan(r['PH']) else 6.5
            }
            
        for _, r in dist_agg.iterrows():
            key = f"{r['State']}_{r['District']}"
            fb = state_fallbacks.get(r['State'], {'NITROGEN': 60.0, 'PHOSPHOROUS': 45.0, 'POTASSIUM': 45.0, 'PH': 6.5})
            soil_lookup[key] = {
                'NITROGEN': float(r['NITROGEN']) if not np.isnan(r['NITROGEN']) and r['NITROGEN'] > 0 else fb['NITROGEN'],
                'PHOSPHOROUS': float(r['PHOSPHOROUS']) if not np.isnan(r['PHOSPHOROUS']) and r['PHOSPHOROUS'] > 0 else fb['PHOSPHOROUS'],
                'POTASSIUM': float(r['POTASSIUM']) if not np.isnan(r['POTASSIUM']) and r['POTASSIUM'] > 0 else fb['POTASSIUM'],
                'PH': float(r['PH']) if not np.isnan(r['PH']) and r['PH'] > 0 else fb['PH']
            }
            
    return soil_lookup, state_fallbacks

def load_rainfall_normals():
    """Extract district-level IMD rainfall normals."""
    rf_path = os.path.join(RAW_DIR, 'district wise rainfall normal.csv')
    rain_lookup = {}
    state_rain_fallback = {}
    
    if os.path.exists(rf_path):
        rdf = pd.read_csv(rf_path)
        rdf['State'] = rdf['STATE_UT_NAME'].astype(str).str.strip().str.upper()
        rdf['District'] = rdf['DISTRICT'].astype(str).str.strip().str.upper()
        
        st_agg = rdf.groupby('State')[['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']].median()
        for s, row in st_agg.iterrows():
            state_rain_fallback[s] = {
                'ANNUAL': float(row['ANNUAL']),
                'Jan-Feb': float(row['Jan-Feb']),
                'Mar-May': float(row['Mar-May']),
                'Jun-Sep': float(row['Jun-Sep']),
                'Oct-Dec': float(row['Oct-Dec'])
            }
            
        for _, row in rdf.iterrows():
            key = f"{row['State']}_{row['District']}"
            rain_lookup[key] = {
                'ANNUAL': float(row['ANNUAL']),
                'Jan-Feb': float(row['Jan-Feb']),
                'Mar-May': float(row['Mar-May']),
                'Jun-Sep': float(row['Jun-Sep']),
                'Oct-Dec': float(row['Oct-Dec'])
            }
            
    return rain_lookup, state_rain_fallback

def build_district_season_knowledge_base():
    """
    Build unified district knowledge base with:
    - State -> Districts list
    - (State, District, Season) -> Set of viable crops
    - District coordinates & climate profiles
    """
    print("Building comprehensive Model 3 Knowledge Base...")
    state_agro_map = load_agroclimatic_data()
    soil_lookup, state_soil_fb = load_soil_medians()
    rain_lookup, state_rain_fb = load_rainfall_normals()
    
    crop_csv = os.path.join(RAW_DIR, 'crop_production.csv')
    df = pd.read_csv(crop_csv)
    df.columns = [c.strip() for c in df.columns]
    for col in ['State_Name', 'District_Name', 'Season', 'Crop']:
        df[col] = df[col].astype(str).str.strip().str.upper()
        
    df['Season'] = df['Season'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df.dropna(subset=['Crop', 'Production', 'Area'])
    df = df[df['Area'] > 0]
    
    # Exclude macro-aggregates
    exclude_crops = {'TOTAL FOODGRAIN', 'OILSEEDS TOTAL', 'PULSES TOTAL', 'OTHER MISC. PULSES', 'OTHER CEREALS & MILLETS', 'OTHER OILSEEDS'}
    df = df[~df['Crop'].isin(exclude_crops)]
    
    # Keep crops with at least 50 historical district-year records
    counts = df['Crop'].value_counts()
    valid_crops = set(counts[counts >= 50].index.tolist())
    df = df[df['Crop'].isin(valid_crops)]
    print(f"Retained {len(valid_crops)} granular crops across India with >= 50 records.")
    
    # Group by State, District, Season
    strata = df.groupby(['State_Name', 'District_Name', 'Season'])['Crop'].unique().to_dict()
    
    # Unique States and Districts mapping
    states_dict = {}
    for st, dist_group in df.groupby('State_Name'):
        dists = sorted(dist_group['District_Name'].unique().tolist())
        states_dict[st] = dists
        
    # Standard Indian district coordinates (pre-calculated representative centroids for instant lookup)
    # Allows zero-latency fallback if Open-Meteo geocoding has latency
    district_profiles = {}
    
    all_districts = set()
    for (st, dist), _ in df.groupby(['State_Name', 'District_Name']):
        all_districts.add((st, dist))
        
    for st, dist in all_districts:
        key = f"{st}_{dist}"
        
        # Soil
        soil = soil_lookup.get(key, state_soil_fb.get(st, {'NITROGEN': 60.0, 'PHOSPHOROUS': 45.0, 'POTASSIUM': 45.0, 'PH': 6.5}))
        
        # Rainfall
        rain = rain_lookup.get(key, state_rain_fb.get(st, {'ANNUAL': 1100.0, 'Jan-Feb': 35.0, 'Mar-May': 120.0, 'Jun-Sep': 750.0, 'Oct-Dec': 120.0}))
        
        # Agroclimatic
        agro = state_agro_map.get(st, {'avgtmp_jan': 18.0, 'avgtmp_jul': 30.0, 'avgann_rf': 1100.0})
        
        # List of viable seasons in this district
        seasons_present = sorted(df[(df['State_Name'] == st) & (df['District_Name'] == dist)]['Season'].unique().tolist())
        
        district_profiles[key] = {
            'state': st,
            'district': dist,
            'soil': soil,
            'rainfall': rain,
            'agroclimatic': agro,
            'available_seasons': seasons_present
        }
        
    # Pre-build crop metadata (category, water need, description)
    crop_metadata = {}
    categories = {
        'CEREAL': ['RICE', 'PADDY', 'WHEAT', 'MAIZE', 'BAJRA', 'JOWAR', 'BARLEY', 'RAGI', 'SMALL MILLETS', 'KORRA', 'SAMAI', 'VARAGU'],
        'PULSE': ['MOONG(GREEN GRAM)', 'URAD', 'ARHAR/TUR', 'GRAM', 'PEAS & BEANS (PULSES)', 'MASOOR', 'HORSE-GRAM', 'KHESARI', 'MOTH', 'COWPEA(LOBIA)', 'BLACKGRAM'],
        'OILSEED': ['SESAMUM', 'GROUNDNUT', 'RAPESEED &MUSTARD', 'SUNFLOWER', 'SOYABEAN', 'LINSEED', 'CASTOR SEED', 'NIGER SEED', 'SAFFLOWER'],
        'COMMERCIAL': ['SUGARCANE', 'COTTON(LINT)', 'TOBACCO', 'JUTE', 'SANNHAMP', 'MESTA', 'GUAR SEED'],
        'PLANTATION': ['COCONUT', 'ARECANUT', 'CASHEWNUT', 'TEA', 'BLACK PEPPER', 'CARDAMOM'],
        'VEGETABLE': ['ONION', 'POTATO', 'SWEET POTATO', 'GARLIC', 'TOMATO', 'BRINJAL', 'CABBAGE', 'CAULIFLOWER', 'BHINDI', 'DRUM STICK', 'CUCUMBER', 'BITTER GOURD', 'BOTTLE GOURD', 'PUMP KIN', 'REDISH', 'BEANS & MUTTER(VEGETABLE)'],
        'FRUIT': ['BANANA', 'MANGO', 'PAPAYA', 'CITRUS FRUIT', 'ORANGE', 'PINEAPPLE', 'GRAPES', 'JACK FRUIT', 'WATER MELON', 'POME GRANET'],
        'SPICE': ['DRY CHILLIES', 'TURMERIC', 'DRY GINGER', 'GINGER', 'CORIANDER']
    }
    
    crop_category_map = {}
    for cat, c_list in categories.items():
        for c in c_list:
            crop_category_map[c] = cat
            
    for c in valid_crops:
        cat = crop_category_map.get(c, 'FIELD CROP')
        water_req = 'HIGH' if c in ['RICE', 'PADDY', 'SUGARCANE', 'BANANA', 'JUTE', 'TAPIOCA'] else ('LOW' if c in ['BAJRA', 'JOWAR', 'GRAM', 'SESAMUM', 'HORSE-GRAM', 'CASTOR SEED', 'GUAR SEED', 'MOTH'] else 'MEDIUM')
        crop_metadata[c] = {
            'crop_name': c.title(),
            'category': cat,
            'water_requirement': water_req,
        }
        
    # Serialize Knowledge Base
    with open(os.path.join(DATA_DIR, 'states_districts.json'), 'w') as f:
        json.dump(states_dict, f, indent=2)
        
    with open(os.path.join(DATA_DIR, 'district_profiles.json'), 'w') as f:
        json.dump(district_profiles, f, indent=2)
        
    # Strata mappings: convert tuple keys to string "STATE|DISTRICT|SEASON"
    strata_serializable = {f"{k[0]}|{k[1]}|{k[2]}": list(v) for k, v in strata.items()}
    with open(os.path.join(DATA_DIR, 'district_season_crops.json'), 'w') as f:
        json.dump(strata_serializable, f, indent=2)
        
    with open(os.path.join(DATA_DIR, 'crop_metadata.json'), 'w') as f:
        json.dump(crop_metadata, f, indent=2)
        
    with open(os.path.join(DATA_DIR, 'valid_crop_list.json'), 'w') as f:
        json.dump(sorted(list(valid_crops)), f, indent=2)
        
    print(f"Model 3 Knowledge Base successfully built with {len(district_profiles)} districts and {len(strata)} strata.")
    return states_dict, district_profiles, strata_serializable, valid_crops

if __name__ == "__main__":
    build_district_season_knowledge_base()
