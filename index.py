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
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AgriPrice AI | Pakistan Commodity Price Intelligence</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }
    .glass-card { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); }
    .active-tab { border-bottom: 2px solid #38bdf8; color: #38bdf8; }
  </style>
</head>
<body class="min-h-screen flex flex-col">
  <!-- Top Navigation Bar -->
  <header class="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex justify-between items-center">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
          <i class="fa-solid fa-wheat-awn text-white text-lg"></i>
        </div>
        <div>
          <h1 class="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-sky-400 to-indigo-300">AgriPrice AI</h1>
          <p class="text-xs text-slate-400">Devixo Solutions AI/ML Internship | Muzamil Ahmad</p>
        </div>
      </div>
      <div class="flex items-center space-x-3">
        <span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>
          Vercel Serverless Ready
        </span>
        <a href="https://github.com/MuzamilAhmad/Devixo-AIML-Internship" target="_blank" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition flex items-center space-x-1.5">
          <i class="fa-brands fa-github"></i>
          <span>GitHub</span>
        </a>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full">
    <!-- Tab Navigation -->
    <div class="flex border-b border-slate-800 space-x-8 mb-6 overflow-x-auto text-sm font-medium">
      <button onclick="switchTab('predict')" id="tab-btn-predict" class="py-3 px-1 active-tab flex items-center space-x-2">
        <i class="fa-solid fa-wand-magic-sparkles"></i><span>Live Price Predictor</span>
      </button>
      <button onclick="switchTab('models')" id="tab-btn-models" class="py-3 px-1 text-slate-400 hover:text-slate-200 flex items-center space-x-2">
        <i class="fa-solid fa-trophy"></i><span>Model Benchmarks (4 Models)</span>
      </button>
      <button onclick="switchTab('importance')" id="tab-btn-importance" class="py-3 px-1 text-slate-400 hover:text-slate-200 flex items-center space-x-2">
        <i class="fa-solid fa-chart-column"></i><span>Feature Importance</span>
      </button>
      <button onclick="switchTab('trends')" id="tab-btn-trends" class="py-3 px-1 text-slate-400 hover:text-slate-200 flex items-center space-x-2">
        <i class="fa-solid fa-chart-line"></i><span>Market Price Trends</span>
      </button>
      <button onclick="switchTab('ipl')" id="tab-btn-ipl" class="py-3 px-1 text-slate-400 hover:text-slate-200 flex items-center space-x-2">
        <i class="fa-solid fa-baseball-bat-ball"></i><span>Task 01: IPL Analytics</span>
      </button>
    </div>

    <!-- TAB 1: PREDICTION CONSOLE -->
    <div id="tab-predict" class="space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <!-- Input Form -->
        <div class="lg:col-span-7 glass-card rounded-2xl p-6 shadow-xl">
          <div class="flex items-center justify-between mb-5">
            <h2 class="text-lg font-semibold text-slate-100 flex items-center space-x-2">
              <i class="fa-solid fa-sliders text-sky-400"></i>
              <span>Forecast Input Parameters</span>
            </h2>
            <div class="flex items-center space-x-2">
              <label class="text-xs text-slate-400">Inference Engine:</label>
              <select id="model_select" class="text-xs bg-slate-900 border border-slate-700 text-sky-300 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-sky-500">
                <option value="Random Forest (Tuned)">Random Forest (Tuned Champion)</option>
                <option value="Gradient Boosting">Gradient Boosting Regressor</option>
                <option value="Decision Tree">Decision Tree Regressor</option>
                <option value="Ridge Regression">Ridge Linear Regression</option>
              </select>
            </div>
          </div>

          <form id="predictionForm" class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label class="block text-xs font-medium text-slate-400 mb-1.5">Commodity Name</label>
                <select id="commodity" class="w-full bg-slate-900/80 border border-slate-700 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 focus:border-sky-500 focus:ring-1 focus:ring-sky-500"></select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-400 mb-1.5">Food Category</label>
                <select id="category" class="w-full bg-slate-900/80 border border-slate-700 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 focus:border-sky-500"></select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-400 mb-1.5">Unit of Measurement</label>
                <select id="unit" class="w-full bg-slate-900/80 border border-slate-700 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 focus:border-sky-500"></select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-400 mb-1.5">Administrative Province</label>
                <select id="province" class="w-full bg-slate-900/80 border border-slate-700 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 focus:border-sky-500"></select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-400 mb-1.5">Wholesale / Retail Market</label>
                <select id="market" class="w-full bg-slate-900/80 border border-slate-700 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 focus:border-sky-500"></select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-400 mb-1.5">Previous Recorded Price (PKR)</label>
                <input type="number" id="prev_price" value="75.0" step="0.5" min="1" max="2500" class="w-full bg-slate-900/80 border border-slate-700 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 focus:border-sky-500">
              </div>
            </div>

            <div class="pt-3 border-t border-slate-800">
              <span class="text-xs font-medium text-slate-400 uppercase tracking-wider block mb-2.5">Forecast Temporal Setting</span>
              <div class="grid grid-cols-3 gap-3">
                <div>
                  <label class="block text-xs text-slate-500 mb-1">Year</label>
                  <input type="number" id="year" value="2026" min="2004" max="2030" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-center text-slate-100">
                </div>
                <div>
                  <label class="block text-xs text-slate-500 mb-1">Month (1-12)</label>
                  <input type="number" id="month" value="11" min="1" max="12" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-center text-slate-100">
                </div>
                <div>
                  <label class="block text-xs text-slate-500 mb-1">Day (1-31)</label>
                  <input type="number" id="day" value="15" min="1" max="31" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-center text-slate-100">
                </div>
              </div>
            </div>

            <button type="button" onclick="runPrediction()" class="w-full mt-4 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-medium py-3 rounded-xl shadow-lg shadow-sky-500/25 transition duration-200 flex items-center justify-center space-x-2">
              <i class="fa-solid fa-calculator"></i>
              <span>Generate AI Price Forecast</span>
            </button>
          </form>
        </div>

        <!-- Output Cards -->
        <div class="lg:col-span-5 space-y-4">
          <div class="glass-card rounded-2xl p-6 border-l-4 border-sky-400 shadow-xl">
            <span class="text-xs font-semibold uppercase tracking-wider text-sky-400">Forecast Outcome</span>
            <div class="mt-3 flex items-baseline justify-between">
              <div>
                <span id="res_price" class="text-4xl font-black text-white">--.--</span>
                <span class="text-lg text-slate-400 font-medium ml-1">PKR</span>
              </div>
              <span id="res_delta_badge" class="px-2.5 py-1 rounded-full text-xs font-bold bg-slate-800 text-slate-300">0.00%</span>
            </div>
            <p id="res_ci" class="text-xs text-slate-400 mt-2">95% Confidence Span: calculating...</p>

            <div class="mt-6 pt-4 border-t border-slate-800/80 space-y-2 text-xs">
              <div class="flex justify-between py-1 border-b border-slate-800/40">
                <span class="text-slate-400">Baseline Benchmark Price:</span>
                <span id="res_prev" class="font-medium text-slate-200">-- PKR</span>
              </div>
              <div class="flex justify-between py-1 border-b border-slate-800/40">
                <span class="text-slate-400">Estimated Price Delta:</span>
                <span id="res_delta" class="font-medium text-slate-200">-- PKR</span>
              </div>
              <div class="flex justify-between py-1 border-b border-slate-800/40">
                <span class="text-slate-400">Active ML Architecture:</span>
                <span id="res_model" class="font-medium text-sky-300">Random Forest</span>
              </div>
              <div class="flex justify-between py-1">
                <span class="text-slate-400">Model Expected Test MAE:</span>
                <span id="res_mae" class="font-medium text-emerald-400">2.956 PKR</span>
              </div>
            </div>
          </div>

          <!-- Market Summary Widget -->
          <div class="glass-card rounded-2xl p-5 text-xs text-slate-400 space-y-2.5">
            <h3 class="font-semibold text-slate-200 flex items-center space-x-1.5">
              <i class="fa-solid fa-circle-info text-sky-400"></i>
              <span>Model Confidence &amp; Methodology</span>
            </h3>
            <p>Model trained on <b>7,662 observations</b> from the World Food Programme (WFP) Pakistan database across 2004–2019. Cross-validated via 5-Fold K-Fold with R² of <b>0.9971</b>.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: MODEL BENCHMARKS -->
    <div id="tab-models" class="hidden space-y-6">
      <div class="glass-card rounded-2xl p-6">
        <h2 class="text-lg font-bold text-slate-100 mb-2">4-Model Architectural Comparison Matrix</h2>
        <p class="text-xs text-slate-400 mb-5">Models trained on identical 80/20 train/test splits (6,129 training rows, 1,533 holdout evaluation rows).</p>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-sm text-slate-300">
            <thead class="text-xs uppercase bg-slate-800/80 text-slate-400 border-b border-slate-700">
              <tr>
                <th class="py-3 px-4">Architecture</th>
                <th class="py-3 px-4">Test MAE (PKR)</th>
                <th class="py-3 px-4">Test RMSE (PKR)</th>
                <th class="py-3 px-4">Test R² Score</th>
                <th class="py-3 px-4">5-Fold CV R²</th>
                <th class="py-3 px-4">Status</th>
              </tr>
            </thead>
            <tbody id="metrics_table_body" class="divide-y divide-slate-800 text-xs">
              <!-- Populated by JavaScript -->
            </tbody>
          </table>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="glass-card rounded-2xl p-5">
          <h3 class="text-sm font-semibold text-slate-200 mb-4">R² Accuracy Score Comparison (Higher is Better)</h3>
          <div class="h-64"><canvas id="chartR2"></canvas></div>
        </div>
        <div class="glass-card rounded-2xl p-5">
          <h3 class="text-sm font-semibold text-slate-200 mb-4">Root Mean Squared Error (Lower is Better)</h3>
          <div class="h-64"><canvas id="chartRMSE"></canvas></div>
        </div>
      </div>
    </div>

    <!-- TAB 3: FEATURE IMPORTANCE -->
    <div id="tab-importance" class="hidden space-y-6">
      <div class="glass-card rounded-2xl p-6">
        <h2 class="text-lg font-bold text-slate-100 mb-2">Top Features Driving Commodity Prices</h2>
        <p class="text-xs text-slate-400 mb-4">Derived via Gini impurity reduction from the Tuned Random Forest Regressor.</p>
        <div class="h-80"><canvas id="chartFI"></canvas></div>
        <div class="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div class="p-3.5 bg-slate-900/60 rounded-xl border border-slate-800">
            <span class="font-semibold text-sky-400 block mb-1">1. Autoregressive Momentum</span>
            <p class="text-slate-400">Previous price (88.4%) dominates short-term equilibrium, reflecting real-world inventory sticky pricing.</p>
          </div>
          <div class="p-3.5 bg-slate-900/60 rounded-xl border border-slate-800">
            <span class="font-semibold text-sky-400 block mb-1">2. Commodity Categorization</span>
            <p class="text-slate-400">Inherent baseline costs (meat, ghee vs. wheat flour) account for 4.9% of explanatory variance.</p>
          </div>
          <div class="p-3.5 bg-slate-900/60 rounded-xl border border-slate-800">
            <span class="font-semibold text-sky-400 block mb-1">3. Geographical Friction</span>
            <p class="text-slate-400">Provincial transport distances between Karachi port and northern hubs impact per-unit wholesale costs.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: MARKET TRENDS -->
    <div id="tab-trends" class="hidden space-y-6">
      <div class="glass-card rounded-2xl p-6">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5">
          <div>
            <h2 class="text-lg font-bold text-slate-100">Historical Commodity Trajectories (2004–2019)</h2>
            <p class="text-xs text-slate-400">Multi-year retail price curves across Pakistani commercial markets.</p>
          </div>
          <select id="trend_commodity_select" onchange="updateTrendChart()" class="bg-slate-900 border border-slate-700 text-xs text-sky-300 rounded-lg px-3 py-1.5 focus:outline-none">
            <option value="Wheat flour - Retail">Wheat flour - Retail</option>
            <option value="Rice (basmati, broken) - Retail">Rice (basmati, broken) - Retail</option>
            <option value="Oil (cooking) - Retail">Oil (cooking) - Retail</option>
            <option value="Sugar - Retail">Sugar - Retail</option>
            <option value="Poultry - Retail">Poultry - Retail</option>
          </select>
        </div>
        <div class="h-80"><canvas id="chartTrend"></canvas></div>
      </div>
    </div>

    <!-- TAB 5: TASK 01 IPL ANALYTICS -->
    <div id="tab-ipl" class="hidden space-y-6">
      <div class="glass-card rounded-2xl p-6">
        <h2 class="text-lg font-bold text-slate-100 mb-1">Task 01: Indian Premier League (IPL) Historical Analytics</h2>
        <p class="text-xs text-slate-400 mb-6">Exploratory Data Analysis across 636 fixtures (2008–2017) conducted as part of Task 01.</p>
        
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-800 text-center">
            <span class="text-xs text-slate-400 block mb-1">Total Matches</span>
            <span class="text-2xl font-bold text-white">636</span>
          </div>
          <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-800 text-center">
            <span class="text-xs text-slate-400 block mb-1">Chasing Victories</span>
            <span class="text-2xl font-bold text-emerald-400">339 (53.3%)</span>
          </div>
          <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-800 text-center">
            <span class="text-xs text-slate-400 block mb-1">Toss Field Bias</span>
            <span class="text-2xl font-bold text-sky-400">57.1%</span>
          </div>
          <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-800 text-center">
            <span class="text-xs text-slate-400 block mb-1">Top Winning Team</span>
            <span class="text-2xl font-bold text-amber-400">Mumbai (92)</span>
          </div>
        </div>

        <div class="p-4 bg-slate-900/40 rounded-xl border border-slate-800 text-xs text-slate-300 space-y-2">
          <h4 class="font-semibold text-slate-100">Core Findings from Task 01:</h4>
          <p>• <b>Chasing Supremacy:</b> Toss winners elected to field in 57.1% of fixtures, a proportion that grew to &gt;80% in 2016–2017 due to night dew and target-setting metrics.</p>
          <p>• <b>Franchise Dominance:</b> Mumbai Indians (92 wins), Chennai Super Kings (79 wins), and Kolkata Knight Riders (77 wins) represent the most consistent franchises.</p>
          <p>• <b>Clean Data Standard:</b> 100% missing values in umpire3 and city fields resolved cleanly without row loss.</p>
        </div>
      </div>
    </div>
  </main>

  <footer class="border-t border-slate-800 py-4 text-center text-xs text-slate-500">
    Devixo Solutions AI/ML Internship Program | Deployed on Vercel Serverless | Student: Muzamil Ahmad
  </footer>

  <script>
    let globalTrends = null;
    let chartTrendInstance = null;

    function switchTab(tabId) {
      ['predict', 'models', 'importance', 'trends', 'ipl'].forEach(t => {
        document.getElementById('tab-' + t).classList.add('hidden');
        document.getElementById('tab-btn-' + t).classList.remove('active-tab');
        document.getElementById('tab-btn-' + t).classList.add('text-slate-400');
      });
      document.getElementById('tab-' + tabId).classList.remove('hidden');
      document.getElementById('tab-btn-' + tabId).classList.add('active-tab');
      document.getElementById('tab-btn-' + tabId).classList.remove('text-slate-400');
    }

    async function initApp() {
      // 1. Fetch Options
      const optRes = await fetch('/api/options');
      const opts = await optRes.json();
      
      const popSel = (id, arr) => {
        const el = document.getElementById(id);
        el.innerHTML = arr.map(v => `<option value="${v}">${v}</option>`).join('');
      };

      popSel('commodity', opts.commodities);
      popSel('category', opts.categories);
      popSel('unit', opts.units);
      popSel('province', opts.provinces);
      popSel('market', opts.markets);

      // Select default sensible options
      document.getElementById('commodity').value = 'Wheat flour - Retail';
      document.getElementById('province').value = 'Punjab';
      document.getElementById('market').value = 'Lahore';

      // 2. Fetch Metrics & Build Table
      const metRes = await fetch('/api/metrics');
      const metData = await metRes.json();
      buildMetricsView(metData);

      // 3. Fetch Features
      const featRes = await fetch('/api/features');
      const featData = await featRes.json();
      buildFeaturesChart(featData);

      // 4. Fetch Trends
      const trendRes = await fetch('/api/trends');
      globalTrends = await trendRes.json();
      updateTrendChart();

      // Trigger initial prediction
      runPrediction();
    }

    async function runPrediction() {
      const payload = {
        commodity: document.getElementById('commodity').value,
        category: document.getElementById('category').value,
        unit: document.getElementById('unit').value,
        province: document.getElementById('province').value,
        market: document.getElementById('market').value,
        prev_price: parseFloat(document.getElementById('prev_price').value),
        year: parseInt(document.getElementById('year').value),
        month: parseInt(document.getElementById('month').value),
        day: parseInt(document.getElementById('day').value),
        model_name: document.getElementById('model_select').value
      };

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      document.getElementById('res_price').innerText = data.predicted_price.toFixed(2);
      document.getElementById('res_prev').innerText = payload.prev_price.toFixed(2) + ' PKR';
      document.getElementById('res_delta').innerText = (data.price_delta >= 0 ? '+' : '') + data.price_delta.toFixed(2) + ' PKR';
      document.getElementById('res_model').innerText = data.model_used;
      document.getElementById('res_mae').innerText = data.mae.toFixed(3) + ' PKR';
      document.getElementById('res_ci').innerText = `95% Confidence Span: [${data.ci_lower.toFixed(2)} PKR — ${data.ci_upper.toFixed(2)} PKR]`;

      const badge = document.getElementById('res_delta_badge');
      badge.innerText = (data.percent_change >= 0 ? '+' : '') + data.percent_change.toFixed(1) + '%';
      badge.className = data.percent_change >= 0 ? 
        'px-2.5 py-1 rounded-full text-xs font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20' : 
        'px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
    }

    function buildMetricsView(data) {
      const rows = [
        { name: 'Random Forest (Tuned)', r2: data['Random Forest (Tuned)'].R2_Score, mae: data['Random Forest (Tuned)'].MAE, rmse: data['Random Forest (Tuned)'].RMSE, cv: '0.9948 ± 0.0018', status: 'Champion Model' },
        { name: 'Gradient Boosting', r2: data['Gradient Boosting'].R2_Score, mae: data['Gradient Boosting'].MAE, rmse: data['Gradient Boosting'].RMSE, cv: `${data['Gradient Boosting'].CV_R2_Mean.toFixed(4)} ± ${data['Gradient Boosting'].CV_R2_Std.toFixed(4)}`, status: 'Candidate' },
        { name: 'Decision Tree', r2: data['Decision Tree'].R2_Score, mae: data['Decision Tree'].MAE, rmse: data['Decision Tree'].RMSE, cv: `${data['Decision Tree'].CV_R2_Mean.toFixed(4)} ± ${data['Decision Tree'].CV_R2_Std.toFixed(4)}`, status: 'Baseline' },
        { name: 'Ridge Regression', r2: data['Ridge Regression'].R2_Score, mae: data['Ridge Regression'].MAE, rmse: data['Ridge Regression'].RMSE, cv: `${data['Ridge Regression'].CV_R2_Mean.toFixed(4)} ± ${data['Ridge Regression'].CV_R2_Std.toFixed(4)}`, status: 'Linear Reference' },
      ];

      document.getElementById('metrics_table_body').innerHTML = rows.map(r => `
        <tr class="hover:bg-slate-800/40">
          <td class="py-3 px-4 font-semibold text-slate-100">${r.name}</td>
          <td class="py-3 px-4 text-emerald-400">${r.mae.toFixed(3)}</td>
          <td class="py-3 px-4">${r.rmse.toFixed(3)}</td>
          <td class="py-3 px-4 font-bold text-sky-400">${r.r2.toFixed(4)}</td>
          <td class="py-3 px-4 text-slate-400">${r.cv}</td>
          <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-[10px] ${r.status.includes('Champion') ? 'bg-sky-500/20 text-sky-300' : 'bg-slate-800 text-slate-400'}">${r.status}</span></td>
        </tr>
      `).join('');

      // Render Charts
      new Chart(document.getElementById('chartR2'), {
        type: 'bar',
        data: {
          labels: rows.map(r => r.name),
          datasets: [{ data: rows.map(r => r.r2), backgroundColor: '#38bdf8' }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { y: { min: 0.95, max: 1.0, ticks: { color: '#94a3b8' } }, x: { ticks: { color: '#94a3b8' } } }
        }
      });

      new Chart(document.getElementById('chartRMSE'), {
        type: 'bar',
        data: {
          labels: rows.map(r => r.name),
          datasets: [{ data: rows.map(r => r.rmse), backgroundColor: '#f87171' }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { y: { ticks: { color: '#94a3b8' } }, x: { ticks: { color: '#94a3b8' } } }
        }
      });
    }

    function buildFeaturesChart(features) {
      new Chart(document.getElementById('chartFI'), {
        type: 'bar',
        data: {
          labels: features.map(f => f.feature),
          datasets: [{ data: features.map(f => f.importance), backgroundColor: '#818cf8' }]
        },
        options: {
          indexAxis: 'y',
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' } } }
        }
      });
    }

    function updateTrendChart() {
      if (!globalTrends) return;
      const comm = document.getElementById('trend_commodity_select').value;
      const commData = globalTrends.trends[comm];
      if (!commData) return;

      const colors = ['#38bdf8', '#4ade80', '#fbbf24', '#f87171', '#c084fc'];
      let colorIdx = 0;
      const datasets = [];

      for (const [mkt, val] of Object.entries(commData)) {
        datasets.push({
          label: mkt,
          data: val.prices,
          borderColor: colors[colorIdx % colors.length],
          backgroundColor: 'transparent',
          tension: 0.25,
          pointRadius: 3
        });
        colorIdx++;
      }

      // Pick any market years
      const firstMkt = Object.keys(commData)[0];
      const years = commData[firstMkt].years;

      if (chartTrendInstance) chartTrendInstance.destroy();

      chartTrendInstance = new Chart(document.getElementById('chartTrend'), {
        type: 'line',
        data: { labels: years, datasets: datasets },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#cbd5e1' } } },
          scales: { y: { ticks: { color: '#94a3b8' } }, x: { ticks: { color: '#94a3b8' } } }
        }
      });
    }

    window.onload = initApp;
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)

