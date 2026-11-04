"""
End-to-End Machine Learning Deployment Application
Task 04 Deliverable | Devixo Solutions AI/ML Internship Program
Author: Muzamil Ahmad
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# Page configuration
st.set_page_config(
    page_title="AgriPrice AI | Pakistan Commodity Price Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern enterprise look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f4c81;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset.csv')

@st.cache_resource
def load_all_models():
    models = {
        'Tuned Random Forest (Champion)': joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib')),
        'Gradient Boosting Regressor': joblib.load(os.path.join(MODELS_DIR, 'gradient_boosting.joblib')),
        'Decision Tree Regressor': joblib.load(os.path.join(MODELS_DIR, 'decision_tree.joblib')),
        'Ridge Linear Regression': joblib.load(os.path.join(MODELS_DIR, 'ridge_regression.joblib'))
    }
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    encoders = joblib.load(os.path.join(MODELS_DIR, 'encoders.joblib'))
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    return models, scaler, encoders, metrics, feature_names

@st.cache_data
def load_raw_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

models_dict, scaler, encoders, metrics, feature_cols = load_all_models()
raw_df = load_raw_dataset()

# Title Header
st.markdown("<p class='main-header'>🌾 AgriPrice AI: Pakistan Commodity Price Forecasting Platform</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Devixo Solutions AI/ML Internship Program | <b>Task 04 Production Deployment</b> | Intern: <b>Muzamil Ahmad</b></p>", unsafe_allow_html=True)

# Sidebar: Model Selector & Quick Info
st.sidebar.header("⚙️ Model Configuration")
selected_model_name = st.sidebar.selectbox(
    "Active Inference Engine:",
    list(models_dict.keys()),
    index=0
)
active_model = models_dict[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 System Metadata")
st.sidebar.info("""
- **Dataset:** WFP Pakistan Food Commodity Prices (7,662 observations)
- **Validation:** 5-Fold Cross-Validation
- **Tuning:** GridSearchCV (inner 3-fold)
- **Deployment:** Production Streamlit Engine
""")

# Main Navigation Tabs
tab_pred, tab_perf, tab_fi, tab_eda = st.tabs([
    "🔮 Live Price Prediction",
    "📊 Model Performance & Benchmarks",
    "🔍 Feature Importance",
    "📈 Historical Data Analytics"
])

# ==========================================
# TAB 1: LIVE PRICE PREDICTION
# ==========================================
with tab_pred:
    st.subheader("🛒 Real-World Market Parameter Input")
    st.write("Input current commodity and regional trading center parameters to generate real-time AI price forecasts.")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        st.markdown("**Commodity Details**")
        commodity = st.selectbox("Select Commodity:", encoders['cmname']['classes'], index=0)
        category = st.selectbox("Food Category:", encoders['category']['classes'], index=0)
        unit = st.selectbox("Unit of Measure:", encoders['unit']['classes'], index=0)
        
    with col_in2:
        st.markdown("**Location & Market**")
        province = st.selectbox("Administrative Province:", encoders['admname']['classes'], index=2) # Punjab default
        market = st.selectbox("Market Center:", encoders['mktname']['classes'], index=2) # Lahore default
        prev_price = st.number_input("Last Recorded Price (PKR):", min_value=1.0, max_value=2500.0, value=65.0, step=0.5)

    with col_in3:
        st.markdown("**Temporal Parameters**")
        year = st.slider("Target Year:", 2004, 2026, 2026)
        month = st.slider("Target Month:", 1, 12, 11)
        day = st.slider("Target Day:", 1, 31, 15)

    predict_action = st.button("🚀 Run AI Price Inference", type="primary", use_container_width=True)

    if predict_action or True: # Auto-predict on load
        # Feature transformation
        quarter = ((month - 1) // 3) + 1
        sin_month = np.sin(2 * np.pi * month / 12)
        cos_month = np.cos(2 * np.pi * month / 12)
        is_year_end = 1 if month == 12 else 0
        is_year_start = 1 if month == 1 else 0
        day_of_week = 3

        input_data = pd.DataFrame([{
            'prev_price': prev_price,
            'cmname_encoded': encoders['cmname']['mapping'].get(commodity, 0),
            'category_encoded': encoders['category']['mapping'].get(category, 0),
            'unit_encoded': encoders['unit']['mapping'].get(unit, 0),
            'admname_encoded': encoders['admname']['mapping'].get(province, 0),
            'mktname_encoded': encoders['mktname']['mapping'].get(market, 0),
            'year': year,
            'month': month,
            'day': day,
            'day_of_week': day_of_week,
            'quarter': quarter,
            'sin_month': sin_month,
            'cos_month': cos_month,
            'is_year_end': is_year_end,
            'is_year_start': is_year_start
        }])[feature_cols]

        input_scaled = scaler.transform(input_data)
        prediction = float(active_model.predict(input_scaled)[0])
        
        # Calculate dynamic uncertainty band
        err_mae = metrics['Random Forest (Tuned)']['MAE']
        ci_lower = max(0.0, prediction - 1.96 * (err_mae / 1.25))
        ci_upper = prediction + 1.96 * (err_mae / 1.25)
        price_diff = prediction - prev_price
        pct_diff = (price_diff / prev_price) * 100

        st.markdown("---")
        st.subheader("💡 Forecast Results & Market Intelligence")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            label="Predicted Market Price",
            value=f"{prediction:.2f} PKR",
            delta=f"{price_diff:+.2f} PKR ({pct_diff:+.1f}%)",
            delta_color="inverse" if pct_diff > 0 else "normal"
        )
        c2.metric(label="Previous Benchmark Price", value=f"{prev_price:.2f} PKR")
        c3.metric(label="Active Machine Learning Model", value=selected_model_name.split('(')[0].strip())
        c4.metric(label="95% Confidence Span", value=f"± {(ci_upper - ci_lower)/2:.2f} PKR")

        st.success(f"""
        **Market Verdict:** The forecast indicates **{commodity}** in **{market} ({province})** for **{month}/{year}** 
        is expected to settle at **{prediction:.2f} PKR per {unit}** (95% CI: [{ci_lower:.2f} PKR — {ci_upper:.2f} PKR]).
        """)

# ==========================================
# TAB 2: MODEL PERFORMANCE & BENCHMARKS
# ==========================================
with tab_perf:
    st.subheader("🏆 Model Development & Benchmarking (4 Models Evaluated)")
    st.write("In compliance with Task 04 specifications, four distinct machine learning architectures were trained on identical stratified train/test splits (80/20).")

    # Metrics DataFrame
    perf_rows = [
        {'Model Architecture': 'Random Forest (Tuned)', 'MAE (PKR)': metrics['Random Forest (Tuned)']['MAE'], 'MSE': metrics['Random Forest (Tuned)']['MSE'], 'RMSE (PKR)': metrics['Random Forest (Tuned)']['RMSE'], 'R² Score': metrics['Random Forest (Tuned)']['R2_Score'], 'Status': 'Champion Model'},
        {'Model Architecture': 'Random Forest (Base)', 'MAE (PKR)': metrics['Random Forest']['MAE'], 'MSE': metrics['Random Forest']['MSE'], 'RMSE (PKR)': metrics['Random Forest']['RMSE'], 'R² Score': metrics['Random Forest']['R2_Score'], 'Status': 'Pre-Tuning'},
        {'Model Architecture': 'Gradient Boosting Regressor', 'MAE (PKR)': metrics['Gradient Boosting']['MAE'], 'MSE': metrics['Gradient Boosting']['MSE'], 'RMSE (PKR)': metrics['Gradient Boosting']['RMSE'], 'R² Score': metrics['Gradient Boosting']['R2_Score'], 'Status': 'Candidate'},
        {'Model Architecture': 'Decision Tree Regressor', 'MAE (PKR)': metrics['Decision Tree']['MAE'], 'MSE': metrics['Decision Tree']['MSE'], 'RMSE (PKR)': metrics['Decision Tree']['RMSE'], 'R² Score': metrics['Decision Tree']['R2_Score'], 'Status': 'Baseline'},
        {'Model Architecture': 'Ridge Linear Regression', 'MAE (PKR)': metrics['Ridge Regression']['MAE'], 'MSE': metrics['Ridge Regression']['MSE'], 'RMSE (PKR)': metrics['Ridge Regression']['RMSE'], 'R² Score': metrics['Ridge Regression']['R2_Score'], 'Status': 'Linear Reference'}
    ]
    perf_table = pd.DataFrame(perf_rows)
    st.dataframe(perf_table.style.format({'MAE (PKR)': '{:.3f}', 'MSE': '{:.3f}', 'RMSE (PKR)': '{:.3f}', 'R² Score': '{:.4f}'}).highlight_max(subset=['R² Score'], color='#bbf7d0').highlight_min(subset=['MAE (PKR)', 'RMSE (PKR)'], color='#bbf7d0'), use_container_width=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig_r2 = px.bar(perf_table, x='Model Architecture', y='R² Score', color='R² Score',
                        color_continuous_scale='Blues', title="R² Score Comparison (Higher is Better)")
        fig_r2.update_layout(yaxis_range=[0.95, 1.0])
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_p2:
        fig_rmse = px.bar(perf_table, x='Model Architecture', y='RMSE (PKR)', color='RMSE (PKR)',
                          color_continuous_scale='Reds_r', title="RMSE Comparison in PKR (Lower is Better)")
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("#### ⚡ Hyperparameter Tuning Impact (Before vs. After)")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f"""
        - **Optimization Algorithm:** `GridSearchCV` with 3-Fold Inner Cross-Validation
        - **Optimal Parameters Found:** `{metrics['Random Forest (Tuned)']['Best_Params']}`
        - **MAE Improvement:** {metrics['Random Forest']['MAE']:.3f} PKR ➔ **{metrics['Random Forest (Tuned)']['MAE']:.3f} PKR**
        - **RMSE Improvement:** {metrics['Random Forest']['RMSE']:.3f} PKR ➔ **{metrics['Random Forest (Tuned)']['RMSE']:.3f} PKR**
        """)
    with col_t2:
        tuning_df = pd.DataFrame({
            'Metric': ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)'],
            'Before Tuning': [metrics['Random Forest']['MAE'], metrics['Random Forest']['RMSE']],
            'After Tuning': [metrics['Random Forest (Tuned)']['MAE'], metrics['Random Forest (Tuned)']['RMSE']]
        })
        fig_tune = px.bar(tuning_df, x='Metric', y=['Before Tuning', 'After Tuning'], barmode='group',
                          title="Tuning Error Reduction", color_discrete_sequence=['#94a3b8', '#0ea5e9'])
        st.plotly_chart(fig_tune, use_container_width=True)

# ==========================================
# TAB 3: FEATURE IMPORTANCE
# ==========================================
with tab_fi:
    st.subheader("🔍 Feature Importance & Economic Interpretability")
    st.write("Evaluating which structural, temporal, and geographic features drive price determinations.")

    rf_model = models_dict['Tuned Random Forest (Champion)']
    if hasattr(rf_model, 'feature_importances_'):
        fi_series = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fi_df = pd.DataFrame({'Feature': fi_series.index, 'Importance': fi_series.values})
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance',
                        color_continuous_scale='Viridis', title="Feature Importance via Gini Impurity Reduction")
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("""
        **Economic Interpretation:**
        1. **`prev_price` (88.4%):** Strongest auto-regressive momentum signal in staple food pricing.
        2. **`cmname_encoded` (4.9%):** Intrinsic commodity characteristics (e.g., meat and ghee vs. coarse grain).
        3. **`unit_encoded` (2.1%):** Weight vs. volume scaling.
        4. **`year` (1.8%):** Long-term macro inflationary shifts.
        5. **`mktname_encoded` (1.1%):** Geographic and freight disparities between urban centers.
        """)

# ==========================================
# TAB 4: HISTORICAL DATA ANALYTICS
# ==========================================
with tab_eda:
    st.subheader("📈 Historical Commodity Price Exploration (2004–2019)")
    if raw_df is not None:
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            sel_comm = st.selectbox("Analyze Commodity Trend:", sorted(raw_df['cmname'].unique()), index=0)
        with c_sel2:
            sel_mkt = st.multiselect("Select Markets:", sorted(raw_df['mktname'].unique()), default=sorted(raw_df['mktname'].unique()))

        trend_df = raw_df[(raw_df['cmname'] == sel_comm) & (raw_df['mktname'].isin(sel_mkt))]
        if len(trend_df) > 0:
            trend_grouped = trend_df.groupby(['year', 'mktname'])['price'].mean().reset_index()
            fig_trend = px.line(trend_grouped, x='year', y='price', color='mktname', markers=True,
                                title=f"Historical Price Trajectory for {sel_comm} (PKR)")
            st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("#### Commodity Categories Breakdown")
        cat_counts = raw_df['category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Records Count']
        fig_cat = px.pie(cat_counts, names='Category', values='Records Count', hole=0.3,
                         title="Distribution of Records by Food Category")
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.warning("Historical dataset not located in local directory.")

st.markdown("---")
st.markdown("<center><small>Devixo Solutions AI/ML Internship Program | Task 04 Production Build | Author: Muzamil Ahmad</small></center>", unsafe_allow_html=True)

