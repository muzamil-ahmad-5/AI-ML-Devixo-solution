# DEVIXO SOLUTIONS – AI/ML INTERNSHIP PROGRAM
## Task 04: End-to-End ML Application with Vercel Deployment

**Intern Name:** Muzamil Ahmad  
**Domain:** Artificial Intelligence & Machine Learning (AI/ML)  
**Host Organization:** Devixo Solutions  
**Deployment Target:** **Vercel Serverless Platform**  
**Framework:** **FastAPI + Modern Tailwind CSS & Chart.js** (No Streamlit required)  
**Submission Date:** November 2026  
**Status:** Completed & Vercel Deployable  

---

### 1. Project Overview & Vercel Architecture
Task 04 requires delivering a complete, interactive machine learning application deployed online. Rather than using Streamlit (which requires stateful WebSockets and is incompatible with serverless runtimes), this deployment utilizes **FastAPI** with Vercel's official Python runtime (`@vercel/python`).

```
Web Browser
     │
     ▼
Vercel Edge Network
     │
     ▼
api/index.py (FastAPI Serverless Function)
     ├── GET /              -> Responsive Web UI (Tailwind CSS, Chart.js)
     ├── POST /api/predict  -> Live Machine Learning Inference
     ├── GET /api/options   -> Dynamic Selectors (Commodities, Markets)
     ├── GET /api/metrics   -> 4-Model Comparative Benchmark
     ├── GET /api/features  -> Feature Importance Analysis
     └── GET /api/trends    -> Historical Price Trends
```

---

### 2. Model Performance Summary

| Model Architecture | MAE (PKR) | MSE | RMSE (PKR) | R² Score | Deployment Role |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **2.956** | **52.838** | **7.269** | **0.9971** | **Champion Production Engine** |
| Gradient Boosting Regressor | 3.088 | 42.915 | 6.551 | 0.9977 | High-Precision Ensemble |
| Decision Tree Regressor | 3.491 | 72.868 | 8.536 | 0.9960 | High-Speed Tree Baseline |
| Ridge Linear Regression | 7.579 | 185.698 | 13.627 | 0.9899 | Regularized Linear Baseline |

---

### 3. Key Web Features Implemented
1. **Interactive Real-Time Price Forecasting Console**:
   - Dynamic dropdown selectors for Commodities, Categories, Units, Provinces, and Markets.
   - Temporal sliders for target year, month, and day.
   - Instant calculation of predicted price in PKR.
2. **Multi-Model Inference Switcher**:
   - Users can toggle between Tuned Random Forest, Gradient Boosting, Decision Tree, and Ridge Regression in real-time.
3. **95% Confidence Interval**:
   - Parametrically calculated prediction interval bounds.
4. **Model Performance Hub**:
   - Interactive table and Chart.js bar charts for R² and RMSE.
5. **Feature Importance Explorer**:
   - Visualizing Gini impurity reduction across all 15 predictor variables.
6. **Historical Commodity Analytics**:
   - Multi-market time series charting retail price movements from 2004 to 2019.
7. **Task 01 IPL Analytics Explorer**:
   - Embedded summary of Task 01 sports analytics and insights.

---

### 4. How to Deploy on Vercel

```bash
# 1. Install Vercel CLI (if not already installed)
npm i -g vercel

# 2. Deploy from the repository root
vercel
```
Or simply connect your GitHub repository to [Vercel](https://vercel.com) for automated zero-configuration CI/CD deployments.

---

### 5. How to Test Locally

```bash
# In the repository root
python run_local.py
```
Open **`http://localhost:8000`** in your browser!
