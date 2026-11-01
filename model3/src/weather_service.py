import os
import time
import requests
from typing import Dict, Any, Tuple, Optional

WMO_WEATHER_CODES = {
    0: "Clear Sky",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing Rime Fog",
    51: "Light Drizzle",
    53: "Moderate Drizzle",
    55: "Dense Drizzle",
    61: "Slight Rain",
    63: "Moderate Rain",
    65: "Heavy Rain",
    71: "Slight Snow Fall",
    73: "Moderate Snow Fall",
    75: "Heavy Snow Fall",
    80: "Slight Rain Showers",
    81: "Moderate Rain Showers",
    82: "Violent Rain Showers",
    95: "Thunderstorm",
    96: "Thunderstorm with Slight Hail",
    99: "Thunderstorm with Heavy Hail"
}

# In-memory weather cache: key -> (timestamp, data)
WEATHER_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL = 900  # 15 minutes

def geocode_district(state: str, district: str) -> Tuple[float, float]:
    """
    Geocode district and state to latitude and longitude using Open-Meteo API.
    Provides robust fallback coordinates for Indian states if geocoding fails.
    """
    clean_dist = district.title().replace('And', '&')
    clean_state = state.title().replace('And', '&')
    
    # 1. Try exact district search
    queries = [
        f"{clean_dist}, {clean_state}, India",
        f"{clean_dist}, India",
        f"{clean_state}, India"
    ]
    
    for q in queries:
        try:
            url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=en&format=json"
            resp = requests.get(url, timeout=4).json()
            if resp.get("results") and len(resp["results"]) > 0:
                lat = resp["results"][0]["latitude"]
                lon = resp["results"][0]["longitude"]
                return lat, lon
        except Exception:
            continue
            
    # Default fallback coordinates for central India (Nagpur centroid)
    return 21.1458, 79.0882

def fetch_multi_horizon_weather(state: str, district: str, season: str, historical_normals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fetch comprehensive Multi-Horizon Weather Intelligence:
    1. Current Weather (live telemetry)
    2. Upcoming Weather (7-day daily forecast)
    3. Overall Historical Climate Normals (IMD 50-year normals)
    """
    cache_key = f"{state.upper()}_{district.upper()}"
    now = time.time()
    
    if cache_key in WEATHER_CACHE:
        ts, cached_data = WEATHER_CACHE[cache_key]
        if now - ts < CACHE_TTL:
            return cached_data
            
    lat, lon = geocode_district(state, district)
    
    weather_report = {
        "location": {
            "state": state.title(),
            "district": district.title(),
            "latitude": lat,
            "longitude": lon
        },
        "current": {
            "temperature": 27.0,
            "relative_humidity": 65.0,
            "apparent_temperature": 28.0,
            "precipitation": 0.0,
            "weather_code": 0,
            "weather_condition": "Clear Sky",
            "wind_speed": 5.0,
            "is_live": False
        },
        "upcoming_forecast": {
            "forecast_days": 7,
            "expected_rainfall_sum_mm": 0.0,
            "max_rain_probability": 10,
            "mean_max_temp": 30.0,
            "mean_min_temp": 20.0,
            "rainy_days_count": 0,
            "daily_forecasts": []
        },
        "historical_normals": {
            "annual_rainfall_normal_mm": 1000.0,
            "seasonal_rainfall_normal_mm": 600.0,
            "normal_jan_temp_c": 18.0,
            "normal_jul_temp_c": 30.0,
            "rainfall_status": "Normal"
        }
    }
    
    # 1. Fetch from Open-Meteo API
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"
            "&timezone=auto&forecast_days=7"
        )
        resp = requests.get(url, timeout=5).json()
        
        if "current" in resp:
            curr = resp["current"]
            code = curr.get("weather_code", 0)
            weather_report["current"] = {
                "temperature": float(curr.get("temperature_2m", 27.0)),
                "relative_humidity": float(curr.get("relative_humidity_2m", 65.0)),
                "apparent_temperature": float(curr.get("apparent_temperature", 28.0)),
                "precipitation": float(curr.get("precipitation", 0.0)),
                "weather_code": code,
                "weather_condition": WMO_WEATHER_CODES.get(code, "Clear / Variable"),
                "wind_speed": float(curr.get("wind_speed_10m", 5.0)),
                "is_live": True
            }
            
        if "daily" in resp:
            daily = resp["daily"]
            dates = daily.get("time", [])
            rain_sums = daily.get("precipitation_sum", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            rain_probs = daily.get("precipitation_probability_max", [])
            codes = daily.get("weather_code", [])
            
            clean_rain = [float(r) for r in rain_sums if r is not None]
            total_rain = sum(clean_rain) if clean_rain else 0.0
            
            clean_max_t = [float(t) for t in max_temps if t is not None]
            clean_min_t = [float(t) for t in min_temps if t is not None]
            
            daily_list = []
            rainy_days = 0
            for i in range(len(dates)):
                r_val = float(rain_sums[i]) if i < len(rain_sums) and rain_sums[i] is not None else 0.0
                if r_val >= 1.0:
                    rainy_days += 1
                c_val = codes[i] if i < len(codes) else 0
                daily_list.append({
                    "date": dates[i],
                    "max_temp": float(max_temps[i]) if i < len(max_temps) else 30.0,
                    "min_temp": float(min_temps[i]) if i < len(min_temps) else 20.0,
                    "rainfall_mm": r_val,
                    "rain_probability": float(rain_probs[i]) if i < len(rain_probs) and rain_probs[i] is not None else 0.0,
                    "condition": WMO_WEATHER_CODES.get(c_val, "Fair")
                })
                
            weather_report["upcoming_forecast"] = {
                "forecast_days": len(dates),
                "expected_rainfall_sum_mm": round(total_rain, 1),
                "max_rain_probability": max(rain_probs) if rain_probs and any(p is not None for p in rain_probs) else 0,
                "mean_max_temp": round(sum(clean_max_t) / len(clean_max_t), 1) if clean_max_t else 30.0,
                "mean_min_temp": round(sum(clean_min_t) / len(clean_min_t), 1) if clean_min_t else 20.0,
                "rainy_days_count": rainy_days,
                "daily_forecasts": daily_list
            }
    except Exception as e:
        print(f"Weather API Warning: Could not fetch live weather: {e}. Using calibrated seasonal defaults.")
        
    # 2. Enrich with Historical Normals
    if historical_normals:
        rf_dict = historical_normals.get('rainfall', {})
        agro_dict = historical_normals.get('agroclimatic', {})
        
        ann_rf = float(rf_dict.get('ANNUAL', 1050.0))
        jan_t = float(agro_dict.get('avgtmp_jan', 18.0))
        jul_t = float(agro_dict.get('avgtmp_jul', 30.0))
        
        season_clean = season.upper()
        if 'KHARIF' in season_clean:
            seasonal_rf = float(rf_dict.get('Jun-Sep', 750.0))
        elif 'RABI' in season_clean:
            seasonal_rf = float(rf_dict.get('Jan-Feb', 35.0)) + float(rf_dict.get('Oct-Dec', 90.0))
        elif 'SUMMER' in season_clean:
            seasonal_rf = float(rf_dict.get('Mar-May', 110.0))
        else:
            seasonal_rf = ann_rf
            
        expected_upcoming = weather_report["upcoming_forecast"]["expected_rainfall_sum_mm"]
        if expected_upcoming > 40.0:
            status = "Surplus / Active Rain"
        elif expected_upcoming < 2.0 and 'KHARIF' in season_clean:
            status = "Dry Spell / Moisture Deficit"
        else:
            status = "Normal / Seasonally Aligned"
            
        weather_report["historical_normals"] = {
            "annual_rainfall_normal_mm": round(ann_rf, 1),
            "seasonal_rainfall_normal_mm": round(seasonal_rf, 1),
            "normal_jan_temp_c": round(jan_t, 1),
            "normal_jul_temp_c": round(jul_t, 1),
            "rainfall_status": status
        }
        
    WEATHER_CACHE[cache_key] = (now, weather_report)
    return weather_report

if __name__ == "__main__":
    # Test weather service
    res = fetch_multi_horizon_weather("PUNJAB", "LUDHIANA", "KHARIF")
    print("Test fetch for Ludhiana, Punjab:")
    print("Current:", res["current"])
    print("Upcoming 7-day rain sum:", res["upcoming_forecast"]["expected_rainfall_sum_mm"], "mm")
    print("Historical normals:", res["historical_normals"])
