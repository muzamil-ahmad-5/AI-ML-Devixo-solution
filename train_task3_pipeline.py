"""
Task 03 Training Pipeline & Artifact Generator
Devixo Solutions AI/ML Internship Program
Author: Muzamil Ahmad
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nbformat as nbf

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set directories
BASE_DIR = os.path.abspath('Task_03_AI_ML')
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMG_DIR = os.path.join(BASE_DIR, 'images')
NB_PATH = os.path.join(BASE_DIR, 'notebook', 'Task_03_Machine_Learning.ipynb')
REPORT_PATH = os.path.join(BASE_DIR, 'report', 'Task_03_Report.pdf')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# 1. Load Data
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

# 2. Data Preparation & Feature Engineering
# Check missing & duplicates
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Feature engineering
df_ml = df.copy()

# Temporal and cyclical features
df_ml['quarter'] = ((df_ml['month'] - 1) // 3) + 1
df_ml['sin_month'] = np.sin(2 * np.pi * df_ml['month'] / 12)
df_ml['cos_month'] = np.cos(2 * np.pi * df_ml['month'] / 12)
df_ml['is_year_end'] = (df_ml['month'] == 12).astype(int)
df_ml['is_year_start'] = (df_ml['month'] == 1).astype(int)

# Categorical Encoders
cat_cols = ['cmname', 'unit', 'category', 'admname', 'mktname']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_ml[col + '_encoded'] = le.fit_transform(df_ml[col].astype(str))
    encoders[col] = {
        'classes': list(le.classes_),
        'mapping': {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
    }

# Save encoders
joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoders.joblib'))

# Feature selection
feature_cols = [
    'prev_price',
    'cmname_encoded',
    'category_encoded',
    'unit_encoded',
    'admname_encoded',
    'mktname_encoded',
    'year',
    'month',
    'day',
    'day_of_week',
    'quarter',
    'sin_month',
    'cos_month',
    'is_year_end',
    'is_year_start'
]

X = df_ml[feature_cols]
y = df_ml['price']

with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
    json.dump(feature_cols, f, indent=2)

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))

# 3. Model Development (Train at least 4 ML models)
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Ridge Regression': Ridge(alpha=1.0)
}

results = {}
trained_models = {}

print("\n--- Training 4 Models ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)
    
    # 5-fold Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='r2', n_jobs=-1)
    
    results[name] = {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'R2_Score': float(r2),
        'CV_R2_Mean': float(cv_scores.mean()),
        'CV_R2_Std': float(cv_scores.std()),
        'Train_R2': float(r2_score(y_train, y_pred_train))
    }
    print(f"[{name}] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.4f} | 5-Fold CV R²: {cv_scores.mean():.4f}")

# 4. Hyperparameter Tuning (GridSearchCV on Random Forest)
print("\n--- Hyperparameter Tuning (GridSearchCV) ---")
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [15, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test_scaled)

tuned_mae = mean_absolute_error(y_test, y_pred_tuned)
tuned_mse = mean_squared_error(y_test, y_pred_tuned)
tuned_rmse = np.sqrt(tuned_mse)
tuned_r2 = r2_score(y_test, y_pred_tuned)

results['Random Forest (Tuned)'] = {
    'MAE': float(tuned_mae),
    'MSE': float(tuned_mse),
    'RMSE': float(tuned_rmse),
    'R2_Score': float(tuned_r2),
    'Best_Params': grid_search.best_params_
}
print(f"[Random Forest Tuned] Best Params: {grid_search.best_params_}")
print(f"[Random Forest Tuned] MAE: {tuned_mae:.3f} | RMSE: {tuned_rmse:.3f} | R²: {tuned_r2:.4f}")

# 5. Save the Best Model
best_model = best_rf
joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.joblib'))

# Save other models for task 4 multi-model selector
joblib.dump(trained_models['Gradient Boosting'], os.path.join(MODELS_DIR, 'gradient_boosting.joblib'))
joblib.dump(trained_models['Decision Tree'], os.path.join(MODELS_DIR, 'decision_tree.joblib'))
joblib.dump(trained_models['Ridge Regression'], os.path.join(MODELS_DIR, 'ridge_regression.joblib'))

with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\nModels and metrics successfully persisted!")

# 6. Feature Importance Analysis
feature_importances = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

# 7. Generate Evaluation Visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Fig 1: Model Comparison Bar Chart (R2, RMSE, MAE)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
models_comp = ['Random Forest', 'Gradient Boosting', 'Decision Tree', 'Ridge Regression']
r2_vals = [results[m]['R2_Score'] for m in models_comp]
rmse_vals = [results[m]['RMSE'] for m in models_comp]

ax[0].bar(models_comp, r2_vals, color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'], edgecolor='black', width=0.6)
ax[0].set_title('Model Comparison: R² Score (Higher is Better)', fontsize=12, fontweight='bold')
ax[0].set_ylabel('R² Score', fontsize=11)
ax[0].set_ylim(0.9, 1.0)
for i, v in enumerate(r2_vals):
    ax[0].text(i, v + 0.003, f"{v:.4f}", ha='center', fontweight='bold', fontsize=9.5)
ax[0].tick_params(axis='x', rotation=15)

ax[1].bar(models_comp, rmse_vals, color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'], edgecolor='black', width=0.6)
ax[1].set_title('Model Comparison: RMSE (Lower is Better)', fontsize=12, fontweight='bold')
ax[1].set_ylabel('Root Mean Squared Error (PKR)', fontsize=11)
for i, v in enumerate(rmse_vals):
    ax[1].text(i, v + 0.2, f"{v:.2f}", ha='center', fontweight='bold', fontsize=9.5)
ax[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'model_comparison.png'), dpi=300)
plt.close()

# Fig 2: Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_tuned, alpha=0.5, color='#1f77b4', edgecolors='none', s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2.5, label='Perfect Prediction (1:1)')
plt.title(f'Actual vs. Predicted Commodity Prices (Tuned Random Forest)\nR² = {tuned_r2:.4f} | MAE = {tuned_mae:.2f} PKR', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Actual Commodity Price (PKR)', fontsize=11)
plt.ylabel('Predicted Commodity Price (PKR)', fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'actual_vs_predicted.png'), dpi=300)
plt.close()

# Fig 3: Residual Distribution Plot
residuals = y_test - y_pred_tuned
plt.figure(figsize=(9, 5))
sns.histplot(residuals, kde=True, bins=40, color='#2ca02c', edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.title(f'Model Residual Error Distribution (y_true - y_pred)\nMean Error: {residuals.mean():.3f} | Std Dev: {residuals.std():.3f}', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Residual Error (PKR)', fontsize=11)
plt.ylabel('Frequency Count', fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'residuals_distribution.png'), dpi=300)
plt.close()

# Fig 4: Feature Importance Bar Chart
plt.figure(figsize=(10, 6))
top_feats = feature_importances.head(10)
bars = plt.barh(top_feats.index[::-1], top_feats.values[::-1], color=sns.color_palette('viridis', len(top_feats)), edgecolor='black')
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.005, bar.get_y() + bar.get_height()/2.0, f"{w:.3f}", va='center', fontweight='bold', fontsize=9.5)
plt.title('Top 10 Feature Importances (Random Forest Model)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Relative Importance (Gini Impurity Reduction)', fontsize=11)
plt.xlim(0, top_feats.max() * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'feature_importance.png'), dpi=300)
plt.close()

# Fig 5: Hyperparameter Tuning Comparison (Before vs After)
plt.figure(figsize=(8, 5))
metrics_comp = ['MAE (PKR)', 'RMSE (PKR)']
before_vals = [results['Random Forest']['MAE'], results['Random Forest']['RMSE']]
after_vals = [tuned_mae, tuned_rmse]
x = np.arange(len(metrics_comp))
width = 0.35

plt.bar(x - width/2, before_vals, width, label='Before Tuning (Default)', color='#94a3b8', edgecolor='black')
plt.bar(x + width/2, after_vals, width, label='After Tuning (GridSearch)', color='#0ea5e9', edgecolor='black')
plt.xticks(x, metrics_comp, fontsize=11)
plt.ylabel('Error Value', fontsize=11)
plt.title('Hyperparameter Tuning Optimization Impact (Random Forest)', fontsize=12, fontweight='bold', pad=12)
plt.legend(frameon=True)
for i in range(len(metrics_comp)):
    plt.text(x[i] - width/2, before_vals[i] + 0.1, f"{before_vals[i]:.2f}", ha='center', fontweight='bold')
    plt.text(x[i] + width/2, after_vals[i] + 0.1, f"{after_vals[i]:.2f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'tuning_comparison.png'), dpi=300)
plt.close()

print("Evaluation figures generated successfully!")

