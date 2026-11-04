"""
Vercel Serverless Production Entrypoint
Devixo Solutions AI/ML Internship Program
Author: Muzamil Ahmad
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AgriPrice AI - Vercel Deployment", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')

# Load models and artifacts lazily/on startup
def load_assets():
    models = {
        'Random Forest (Tuned)': joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib')),
        'Gradient Boosting': joblib.load(os.path.join(MODELS_DIR, 'gradient_boosting.joblib')),
        'Decision Tree': joblib.load(os.path.join(MODELS_DIR, 'decision_tree.joblib')),
        'Ridge Regression': joblib.load(os.path.join(MODELS_DIR, 'ridge_regression.joblib'))
    }
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    encoders = joblib.load(os.path.join(MODELS_DIR, 'encoders.joblib'))
    
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    with open(os.path.join(MODELS_DIR, 'historical_summary.json'), 'r') as f:
        historical = json.load(f)
        
    return models, scaler, encoders, metrics, feature_names, historical

models, scaler, encoders, metrics, feature_names, historical = load_assets()

# Extract feature importances
rf_champion = models['Random Forest (Tuned)']
feature_importances = []
if hasattr(rf_champion, 'feature_importances_'):
    fi_dict = dict(zip(feature_names, rf_champion.feature_importances_))
    sorted_fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)
    feature_importances = [{'feature': k, 'importance': round(float(v), 4)} for k, v in sorted_fi[:10]]

class PredictionRequest(BaseModel):
    commodity: str
    category: str
    unit: str
    province: str
    market: str
    prev_price: float
    year: int
    month: int
    day: int
    model_name: str = 'Random Forest (Tuned)'

@app.get("/api/options")
async def get_options():
    return {
        "commodities": encoders['cmname']['classes'],
        "categories": encoders['category']['classes'],
        "units": encoders['unit']['classes'],
        "provinces": encoders['admname']['classes'],
        "markets": encoders['mktname']['classes'],
        "models": list(models.keys())
    }

@app.get("/api/metrics")
async def get_metrics():
    return metrics

@app.get("/api/features")
async def get_features():
    return feature_importances

@app.get("/api/trends")
async def get_trends():
    return historical

@app.post("/api/predict")
async def predict_price(req: PredictionRequest):
    # Select active model
    selected_model = models.get(req.model_name, models['Random Forest (Tuned)'])
    
    # Feature Engineering
    quarter = ((req.month - 1) // 3) + 1
    sin_month = np.sin(2 * np.pi * req.month / 12)
    cos_month = np.cos(2 * np.pi * req.month / 12)
    is_year_end = 1 if req.month == 12 else 0
    is_year_start = 1 if req.month == 1 else 0
    day_of_week = 3

    cmname_enc = encoders['cmname']['mapping'].get(req.commodity, 0)
    cat_enc = encoders['category']['mapping'].get(req.category, 0)
    unit_enc = encoders['unit']['mapping'].get(req.unit, 0)
    adm_enc = encoders['admname']['mapping'].get(req.province, 0)
    mkt_enc = encoders['mktname']['mapping'].get(req.market, 0)

    input_df = pd.DataFrame([{
        'prev_price': req.prev_price,
        'cmname_encoded': cmname_enc,
        'category_encoded': cat_enc,
        'unit_encoded': unit_enc,
        'admname_encoded': adm_enc,
        'mktname_encoded': mkt_enc,
        'year': req.year,
        'month': req.month,
        'day': req.day,
        'day_of_week': day_of_week,
        'quarter': quarter,
        'sin_month': sin_month,
        'cos_month': cos_month,
        'is_year_end': is_year_end,
        'is_year_start': is_year_start
    }])[feature_names]

    input_scaled = scaler.transform(input_df)
    predicted_val = float(selected_model.predict(input_scaled)[0])

    mae_val = metrics.get(req.model_name, metrics['Random Forest (Tuned)'])['MAE']
    ci_lower = max(0.0, predicted_val - 1.96 * (mae_val / 1.25))
    ci_upper = predicted_val + 1.96 * (mae_val / 1.25)
    delta = predicted_val - req.prev_price
    pct_change = (delta / req.prev_price) * 100 if req.prev_price > 0 else 0.0

    return {
        "status": "success",
        "predicted_price": round(predicted_val, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "price_delta": round(delta, 2),
        "percent_change": round(pct_change, 2),
        "model_used": req.model_name,
        "mae": round(mae_val, 3)
    }

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    html_path = os.path.join(CURRENT_DIR, 'index.html')
    if not os.path.exists(html_path):
        html_path = os.path.join(CURRENT_DIR, '..', 'index.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>AgriPrice AI 3D Platform Active</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)

