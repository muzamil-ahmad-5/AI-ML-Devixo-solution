# DEVIXO SOLUTIONS – AI/ML INTERNSHIP PROGRAM
## Production Deliverables & Vercel Serverless Deployment

**Intern Name:** Muzamil Ahmad  
**Internship Track:** Artificial Intelligence & Machine Learning (AI/ML)  
**Host Organization:** Devixo Solutions  
**Deployment Platform:** **Vercel Serverless (FastAPI + Modern HTML/Tailwind/Chart.js)**  
**Submission Date:** November 2026  
**Status:** Completed & Ready for Instant Vercel Deployment  

---

## 🚀 Live Vercel Deployment Architecture

The application has been engineered to run **directly on Vercel Serverless** using FastAPI, lightweight compressed machine learning pipelines, and a modern reactive frontend with Tailwind CSS and Chart.js.

### Why Serverless instead of Streamlit?
Streamlit requires a persistent, stateful WebSocket connection and long-running server process that is incompatible with Vercel's serverless execution model. By transitioning the application to **FastAPI + Serverless Python (`@vercel/python`)**, the project achieves:
- **Instant Cold Starts:** Under 800ms initial response time.
- **Global Edge Performance:** Served via Vercel Edge Network.
- **Auto-Scaling:** Automatically scales from 0 to thousands of requests with zero infrastructure cost.
- **Rich Interactive UI:** Modern single-page app with dynamic forecasting, multi-model switching, 95% confidence intervals, and Chart.js analytics.

---

## 🗂️ Repository Directory Structure

```
AI ML Devixio solution/
│
├── api/                                 # VERCEL SERVERLESS BACKEND
│   ├── index.py                         # FastAPI Production App + Embedded Modern UI
│   └── models/                          # Production Serialized ML Artifacts (Compressed)
│       ├── best_model.joblib            # Tuned Random Forest (Champion)
│       ├── gradient_boosting.joblib     # Gradient Boosting Regressor
│       ├── decision_tree.joblib         # Decision Tree Regressor
│       ├── ridge_regression.joblib      # Ridge Linear Regression
│       ├── scaler.joblib                # Fitted StandardScaler
│       ├── encoders.joblib              # LabelEncoders Registry
│       ├── metrics.json                 # Benchmarking Metrics Registry
│       ├── feature_names.json           # Input Feature Names Schema
│       └── historical_summary.json      # Pre-computed Historical Time Series
│
├── vercel.json                          # Vercel Serverless Routing Configuration
├── requirements.txt                     # Optimized Vercel Deployment Dependencies
├── run_local.py                         # 1-Click Local Development Server Runner
│
├── Task_01_AI_ML/                       # TASK 01: PYTHON PROGRAMMING, EDA & VISUALIZATION
│   ├── dataset/
│   │   ├── dataset.csv                  # Raw IPL match records (636 rows)
│   │   └── cleaned_dataset.csv          # Cleaned dataset (0 nulls, date-parsed)
│   ├── notebook/
│   │   └── Task_01_Data_Analysis.ipynb  # Executable Jupyter Notebook
│   ├── images/                          # 10 High-Resolution Visualizations (PNG)
│   │   ├── visualization_01.png ... visualization_10.png
│   ├── report/
│   │   └── Task_01_Report.pdf           # Publication-Grade 20-Section Technical Report
│   └── README.md                        # Task 01 Documentation
│
├── Task_03_AI_ML/                       # TASK 03: END-TO-END MACHINE LEARNING PROJECT
│   ├── dataset/
│   │   └── dataset.csv                  # Pakistan Food Prices Dataset (7,662 rows > 3,000)
│   ├── notebook/
│   │   └── Task_03_Machine_Learning.ipynb # Complete ML Pipeline Jupyter Notebook
│   ├── models/                          # Trained Models and Validation Metrics
│   ├── images/                          # Evaluation Charts (R², RMSE, Residuals, Importances)
│   ├── report/
│   │   └── Task_03_Report.pdf           # Technical Machine Learning Report (PDF)
│   └── README.md                        # Task 03 Documentation
│
└── Task_04_AI_ML/                       # TASK 04: END-TO-END ML APPLICATION WITH DEPLOYMENT
    ├── api/
    │   ├── index.py                     # Self-contained Vercel FastAPI Deployment
    │   └── models/                      # Dedicated Model Artifacts
    ├── dataset/
    │   └── dataset.csv
    ├── vercel.json                      # Task 04 Vercel Configuration
    ├── requirements.txt                 # Task 04 Dependencies
    ├── report/
    │   └── Task_04_Report.pdf           # Technical Deployment Report (PDF)
    └── README.md                        # Task 04 Documentation
```

---

## ⚡ How to Deploy Directly to Vercel

### Option 1: Deploy with Vercel CLI (Fastest)
1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```
2. In the project root directory, run:
   ```bash
   vercel
   ```
3. Follow the prompts (use default project settings). Vercel will detect `api/index.py` and `vercel.json` and deploy your application in under 60 seconds!

---

### Option 2: Deploy via GitHub & Vercel Dashboard
1. Push this repository to your GitHub account:
   ```bash
   git init
   git add .
   git commit -m "Devixo Solutions AI/ML Internship Deliverables"
   git remote add origin https://github.com/MuzamilAhmad/Devixo-AIML-Internship.git
   git push -u origin main
   ```
2. Log into [Vercel](https://vercel.com).
3. Click **"Add New..."** ➔ **"Project"**.
4. Select your repository and click **"Deploy"**.
5. Vercel will automatically build the serverless Python environment and provide a live URL (e.g. `https://your-project.vercel.app`).

---

## 💻 How to Test Locally

You can test the exact Vercel FastAPI application on your computer:

```bash
# Run the local server
python run_local.py
```
Open your browser to: **`http://localhost:8000`**

---

## 📊 Summary of Completed Internship Tasks

| Task | Topic | Dataset | Key Outputs | Deliverables |
| :--- | :--- | :--- | :--- | :--- |
| **Task 01** | Python Programming, EDA & Visualization | IPL Records (636 rows) | 10 Visualizations, Central Tendency, Skewness, Insights | `Task_01_Report.pdf`, `Task_01_Data_Analysis.ipynb`, 10 PNGs |
| **Task 03** | End-to-End Machine Learning Pipeline | Commodity Prices (7,662 rows) | 4 Models, 5-Fold CV ($R^2=0.9948$), GridSearchCV | `Task_03_Report.pdf`, `Task_03_Machine_Learning.ipynb`, Joblib models |
| **Task 04** | End-to-End ML App with Deployment | Commodity Prices (7,662 rows) | Vercel Serverless Web App, Multi-Model Switcher, Confidence Intervals | `Task_04_Report.pdf`, `api/index.py`, `vercel.json` |

---

### 🏆 4-Model Comparative Benchmark

| Model Architecture | Test MAE (PKR) | Test RMSE (PKR) | Test R² Score | 5-Fold CV R² | Role |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **2.956** | **7.269** | **0.9971** | **0.9948 ± 0.0018** | **Champion Production Engine** |
| Gradient Boosting Regressor | 3.088 | 6.551 | 0.9977 | 0.9955 ± 0.0015 | Ensemble Alternative |
| Decision Tree Regressor | 3.491 | 8.536 | 0.9960 | 0.9928 ± 0.0024 | Fast Tree Baseline |
| Ridge Linear Regression | 7.579 | 13.627 | 0.9899 | 0.9555 ± 0.0125 | Regularized Reference |

---

<center><b>Devixo Solutions AI/ML Internship Program | Author: Muzamil Ahmad</b></center>
