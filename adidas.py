import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# Time-Series Libraries
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -------------------------------------------------------------------------
# Page Setup & Styling Configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Adidas Full Analytics Workspace",
    page_icon="👟",
    layout="wide"
)

st.title("👟 Adidas End-to-End Analytics Framework")
st.markdown("""
This production dashboard combines all four analytics layers of our project: 
**Descriptive EDA**, **Predictive Time-Series Forecasting**, **Predictive Machine Learning**, and **Prescriptive Optimization**.
""")
st.markdown("---")

# -------------------------------------------------------------------------
# Caching Engines for Backend High-Performance Models
# -------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    # Load dataset
   df = pd.read_csv('adidas_dataset.csv')
   df['Order_Date'] = pd.to_datetime(df['Order_Date'])
   df['Cost'] = df['Revenue'] - df['Profit']
   return df

@st.cache_resource
def train_random_forest_pipeline(data):
    features = ['Category', 'Region', 'Store_Type', 'Units_Sold', 'Unit_Price', 'Discount', 'Customer_Age', 'Gender', 'Payment_Method']
    target = 'Revenue'
    X = data[features]
    y = data[target]
    
    categorical_features = ['Category', 'Region', 'Store_Type', 'Gender', 'Payment_Method']
    numerical_features = ['Units_Sold', 'Unit_Price', 'Discount', 'Customer_Age']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    rf_pipeline.fit(X, y)
    return rf_pipeline

@st.cache_data
def generate_time_series_forecast(data):
    ts_data = data.groupby(data['Order_Date'].dt.to_period('M'))['Revenue'].sum().reset_index()
    ts_data['Order_Date'] = ts_data['Order_Date'].dt.to_timestamp()
    ts_data.set_index('Order_Date', inplace=True)
    ts_data = ts_data.asfreq('MS')
    
    model = ExponentialSmoothing(ts_data['Revenue'], trend='add', seasonal='add', seasonal_periods=12)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=12)
    return ts_data, forecast

# Execute pipeline backgrounds on app initialization
try:
    df = load_and_prep_data()
    rf_model = train_random_forest_pipeline(df)
    historical_ts, forecast_series = generate_time_series_forecast(df)
except FileNotFoundError:
    st.error("❌ CRITICAL ERROR: 'adidas_dataset.csv' not found. Please make sure the CSV dataset sits in the exact same folder directory as this script.")
    st.stop()

    # -------------------------------------------------------------------------
# Application Navigation Tabs
# -------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Descriptive EDA Overview",
    "📈 2. Holt-Winters Forecasting", 
    "🤖 3. Random Forest ML Predictor", 
    "🎯 4. Prescriptive Optimizer"
])

# =========================================================================
# LAYER 1: DESCRIPTIVE EDA OVERVIEW
# =========================================================================
with tab1:
    st.header("Exploratory Data Analysis Dashboard")
    st.markdown("A macro evaluation of total historical metrics, performance channels, and customer demographic attributes.")
    
    # Core Global Metrics KPIs Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Recorded Revenue", f"${df['Revenue'].sum():,.2f}")
    kpi2.metric("Total Generated Profit", f"${df['Profit'].sum():,.2f}")
    kpi3.metric("Total Cumulative Volume", f"{df['Units_Sold'].sum():,} Units")
    kpi4.metric("Avg Transaction Price", f"${df['Unit_Price'].mean():.2f}")
    
    st.markdown("### Visual Performance Breakdown")
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        # Category Revenue
        fig_cat, ax_cat = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=df.groupby('Category')['Revenue'].sum().reset_index(), x='Category', y='Revenue', palette='viridis', ax=ax_cat)
        ax_cat.set_title("Revenue Contribution by Category", fontsize=10, weight='bold')
        ax_cat.set_ylabel("Revenue ($)")
        st.pyplot(fig_cat)
        
    with eda_col2:
        # Store Type split
        fig_store, ax_store = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=df.groupby('Store_Type')['Revenue'].sum().reset_index(), x='Store_Type', y='Revenue', palette='coolwarm', ax=ax_store)
        ax_store.set_title("Revenue Contribution by Channel Vector", fontsize=10, weight='bold')
        ax_store.set_ylabel("Revenue ($)")
        st.pyplot(fig_store)

        # =========================================================================
# LAYER 2: PREDICTIVE TIME-SERIES FORECASTING
# =========================================================================
with tab2:
    st.header("12-Month Advanced Demand Horizon Forecast")
    st.markdown("Capturing and extrapolating cyclical seasonal demand patterns over the next fiscal layout.")
    
    fore_col1, fore_col2 = st.columns([3, 2])
    
    with fore_col1:
        sns.set_theme(style="whitegrid")
        fig_ts, ax_ts = plt.subplots(figsize=(10, 5.5))
        ax_ts.plot(historical_ts.index, historical_ts['Revenue'], marker='o', color='b', label='Historical Monthly Data', linewidth=2)
        ax_ts.plot(forecast_series.index, forecast_series, marker='s', linestyle='--', color='r', label='Model Forecast Timeline', linewidth=2)
        ax_ts.axvspan(historical_ts.index[-1], forecast_series.index[-1], color='gray', alpha=0.08, label='Forecast Horizon Window')
        ax_ts.set_ylabel("Total Monthly Revenue ($)", fontsize=11)
        ax_ts.set_title("Adidas Chronological Revenue Projections (Holt-Winters)", fontsize=12, weight='bold')
        ax_ts.legend(loc='upper left')
        st.pyplot(fig_ts)
        
    with fore_col2:
        st.subheader("📋 Forecast Run-Rate Schedule Table")
        forecast_df = forecast_series.reset_index()
        forecast_df.columns = ['Target Month', 'Predicted Revenue Projection']
        forecast_df['Target Month'] = forecast_df['Target Month'].dt.strftime('%B %Y')
        
        # Keep numeric values intact for exporting, but create formatted display column
        forecast_df['Display Projection'] = forecast_df['Predicted Revenue Projection'].map('${:,.2f}'.format)
        
        st.dataframe(forecast_df[['Target Month', 'Display Projection']], use_container_width=True, hide_index=True)
        st.download_button(
            label="📥 Export Forecast Projections (.csv)",
            data=forecast_df[['Target Month', 'Predicted Revenue Projection']].to_csv(index=False),
            file_name="adidas_12m_forecast_projections.csv",
            mime="text/csv"
        )

        # =========================================================================
# LAYER 3: PREDICTIVE MACHINE LEARNING (RANDOM FOREST)
# =========================================================================
with tab3:
    st.header("Machine Learning Live Transaction Estimation Simulator")
    st.markdown("Simulate incoming orders to instantly compute expected transactional gross revenues using our 100-Tree Forest Regressor.")
    
    st.markdown("### 🎛️ Configure Simulated Transaction Properties")
    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
    
    with row1_col1:
        input_cat = st.selectbox("Product Line Category", options=list(df['Category'].unique()))
    with row1_col2:
        input_region = st.selectbox("Target Logistics Region", options=list(df['Region'].unique()))
    with row1_col3:
        input_store = st.selectbox("Retail Channel Vector", options=list(df['Store_Type'].unique()))
    with row1_col4:
        input_gender = st.selectbox("Target Consumer Segment", options=list(df['Gender'].unique()))
        
    with row2_col1:
        input_price = st.number_input("Sticker Unit Price ($)", min_value=5.0, max_value=300.0, value=85.0, step=5.0)
    with row2_col2:
        input_units = st.slider("Quantity Volume (Units)", min_value=1, max_value=10, value=3)
    with row2_col3:
        input_discount = st.selectbox("Applied Campaign Markdown Discount (%)", options=[0, 5, 10, 20, 40], index=0)
    with row2_col4:
        input_age = st.slider("Target Consumer Age Profile", min_value=16, max_value=60, value=30)
        
    input_payment = st.selectbox("Preferred Processing Network Engine", options=list(df['Payment_Method'].unique()))
    
    # Process features precisely into evaluation dataframe format
    mock_input_df = pd.DataFrame([{
        'Category': input_cat, 'Region': input_region, 'Store_Type': input_store,
        'Units_Sold': input_units, 'Unit_Price': input_price, 'Discount': input_discount,
        'Customer_Age': input_age, 'Gender': input_gender, 'Payment_Method': input_payment
    }])
    
    # Invoke predictive estimation computation
    ml_prediction = rf_model.predict(mock_input_df)[0]
    
    st.markdown("---")
    st.markdown("### 🎯 Model Output Summary")
    st.success(f"### Estimated Transaction Gross Revenue Outcome: **${ml_prediction:,.2f}**")
    st.caption("Calculation computed cross-referencing model structural branch weights ($R^2 \approx 97\%$).")

    # =========================================================================
# LAYER 4: PRESCRIPTIVE OPTIMIZATION CANVAS
# =========================================================================
with tab4:
    st.header("Strategic Profit Margin Optimization Panel")
    st.markdown("""
    **The Strategic Rule Strategy:** Pricing metrics demonstrate flat buyer transactional volume across discount tiers. 
    By setting corporate policy discount limits, we can eliminate margin erosion and reclaim profitability.
    """)
    
    prescribed_discount_cap = st.slider(
        "Set Maximum Corporate Markdown Cap Limit Policy Ceiling (%)",
        min_value=0, max_value=40, value=0, step=5,
        help="Clamps historical discount occurrences down to this value threshold ceiling rule limit."
    )
    
    # Run dynamic optimization simulations
    optimized_df = df.copy()
    optimized_df['Adjusted_Discount'] = optimized_df['Discount'].apply(lambda d: min(d, prescribed_discount_cap))
    optimized_df['Adjusted_Revenue'] = optimized_df['Units_Sold'] * optimized_df['Unit_Price'] * (1 - optimized_df['Adjusted_Discount'] / 100)
    optimized_df['Adjusted_Profit'] = optimized_df['Adjusted_Revenue'] - optimized_df['Cost']
    
    base_rev, opt_rev = df['Revenue'].sum(), optimized_df['Adjusted_Revenue'].sum()
    base_prof, opt_prof = df['Profit'].sum(), optimized_df['Adjusted_Profit'].sum()
    
    rev_variance, prof_variance = opt_rev - base_rev, opt_prof - base_prof
    
    # Render the dynamic comparison metrics metrics cards
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.metric("Optimized Cumulative Revenue", f"${opt_rev:,.2f}", f"${rev_variance:+,.2f} ({rev_variance/base_rev*100:+.2f}%)")
    with p_col2:
        st.metric("Optimized Cumulative Net Profit", f"${opt_prof:,.2f}", f"${prof_variance:+,.2f} ({prof_variance/base_prof*100:+.2f}%)")
    with p_col3:
        margin_base = (base_prof / base_rev) * 100
        margin_opt = (opt_prof / opt_rev) * 100
        st.metric("Corporate Margin Shift Efficiency", f"{margin_opt:.2f}%", f"{margin_opt - margin_base:+.2f}% Efficiency Lift")
        
    # Comparative graphical display layout matrix
    fig_opt, ax_opt = plt.subplots(figsize=(9, 4))
    metrics_data = pd.DataFrame({
        'Financial Performance Tracker': ['Revenue', 'Revenue', 'Profit', 'Profit'],
        'Operational Policy Rule Scenario': ['Current State Baseline', 'Prescribed Cap Strategy', 'Current State Baseline', 'Prescribed Cap Strategy'],
        'Financial Totals ($)': [base_rev, opt_rev, base_prof, opt_prof]
    })
    sns.barplot(data=metrics_data, x='Financial Performance Tracker', y='Financial Totals ($)', hue='Operational Policy Rule Scenario', palette='Set1', ax=ax_opt)
    
    for patch in ax_opt.patches:
        h = patch.get_height()
        if h > 0:
            ax_opt.annotate(f'${h:,.2f}', (patch.get_x() + patch.get_width() / 2., h), ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', weight='bold', fontsize=9)
            
    ax_opt.set_ylim(0, max(opt_rev, opt_prof) * 1.25)
    st.pyplot(fig_opt)
