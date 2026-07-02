Adidas Sales Forecasting, Prediction & Prescriptive Analytics
Project Overview

This project is an end-to-end Machine Learning and Time Series Forecasting solution developed using Python in Visual Studio Code (Jupyter Notebook).

The objective is to analyze Adidas sales data, forecast future sales, predict business performance using Machine Learning, generate business recommendations through Prescriptive Analytics, and deploy the solution using Streamlit.

The project follows an industry-standard Data Science lifecycle from data exploration to deployment.

Project Objectives
Perform Exploratory Data Analysis (EDA)
Conduct Exploratory Time Series Analysis (ETSA)
Forecast future sales
Build Machine Learning prediction models
Generate Prescriptive Analytics recommendations
Compare forecasting algorithms
Deploy an interactive dashboard using Streamlit
Dataset

Dataset: Adidas Sales Dataset

The dataset contains historical Adidas sales transactions including:

Order Date
Product
Category
Region
Retailer
Units Sold
Revenue
Operating Profit
Sales Method
Price
Discount
Other business variables
Technologies Used
Technology	Purpose
Python	Data Science
Pandas	Data Cleaning
NumPy	Numerical Computing
Matplotlib	Visualization
Seaborn	Statistical Visualization
Plotly	Interactive Charts
Statsmodels	Time Series Analysis
Prophet	Forecasting
Scikit-Learn	Machine Learning
XGBoost	Gradient Boosting
Streamlit	Deployment
VS Code	Development Environment
Git & GitHub	Version Control
Project Workflow
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Exploratory Time Series Analysis
    │
    ▼
Feature Engineering
    │
    ▼
Forecasting Models
    │
    ▼
Machine Learning Models
    │
    ▼
Model Evaluation
    │
    ▼
Prescriptive Analytics
    │
    ▼
Business Recommendations
    │
    ▼
Streamlit Deployment
Project Structure
Adidas-Sales-Forecasting/

│
├── data/
│     adidas_dataset.csv
│
├── notebooks/
│     01_Data_Loading.ipynb
│     02_EDA.ipynb
│     03_ETSA.ipynb
│     04_Feature_Engineering.ipynb
│     05_Forecasting.ipynb
│     06_Machine_Learning.ipynb
│     07_Prescriptive_Analytics.ipynb
│
├── models/
│     prophet.pkl
│     xgboost.pkl
│     random_forest.pkl
│
├── images/
│
├── streamlit_app/
│     app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
Exploratory Data Analysis (EDA)

The project begins by understanding the business data through:

Missing value analysis
Duplicate detection
Data type validation
Descriptive statistics
Distribution analysis
Correlation analysis
Sales trends
Regional sales analysis
Product performance
Retailer comparison
Monthly revenue trends

Visualizations include:

Histograms
Boxplots
Heatmaps
Countplots
Line charts
Bar charts
Pie charts
Exploratory Time Series Analysis (ETSA)

Time series analysis includes:

Date indexing
Trend analysis
Seasonality analysis
Moving averages
Rolling statistics
Stationarity testing
ACF plots
PACF plots
Seasonal decomposition

Statistical Tests:

Augmented Dickey-Fuller (ADF)
KPSS Test
Feature Engineering

Features created include:

Year
Quarter
Month
Week
Day
Day of Week
Weekend Indicator
Lag Features
Rolling Mean
Rolling Standard Deviation
Percentage Change
Forecasting Models

Several forecasting algorithms are compared.

Statistical Models
ARIMA
SARIMA
SARIMAX
Holt-Winters Exponential Smoothing
Facebook Prophet

Evaluation Metrics

MAE
RMSE
MAPE
Machine Learning Prediction Models

The project also predicts future sales using supervised learning.

Models include:

Linear Regression
Random Forest Regressor
XGBoost Regressor
Gradient Boosting
Extra Trees Regressor

Model comparison is performed using:

MAE
RMSE
R² Score
Prescriptive Analytics

Predictions alone do not drive business value.

This project also provides actionable business recommendations by performing:

What-if Analysis
Sales simulations
Revenue optimization
Discount impact analysis
Product performance recommendations
Inventory planning
Seasonal demand planning

Example recommendations:

Increase inventory before high-demand months.
Reduce discounts on high-performing products.
Increase marketing spend in underperforming regions.
Optimize stock allocation across retailers.
Streamlit Dashboard

The final application allows users to:

Upload a new dataset
View business KPIs
Explore sales trends
Forecast future sales
Compare forecasting models
Generate predictions
Run prescriptive analytics
Download forecast results
Installation

Clone the repository

git clone https://github.com/yourusername/adidas-sales-forecasting.git

Navigate into the project

cd adidas-sales-forecasting

Install dependencies

pip install -r requirements.txt

Run the Streamlit application

streamlit run app.py
Results

The project successfully:

Cleaned and transformed raw sales data
Identified seasonal sales patterns
Built multiple forecasting models
Compared statistical and Machine Learning approaches
Predicted future sales
Generated business recommendations
Delivered an interactive Streamlit dashboard
Future Improvements

Potential enhancements include:

Deep Learning (LSTM)
GRU Networks
Transformer-based forecasting
Real-time forecasting
Azure deployment
Docker containerization
CI/CD pipeline with GitHub Actions
Power BI integration
Automated retraining pipeline
Skills Demonstrated

This project showcases expertise in:

Data Cleaning
Exploratory Data Analysis
Time Series Analysis
Forecasting
Machine Learning
Feature Engineering
Model Evaluation
Prescriptive Analytics
Business Intelligence
Data Visualization
Python
Streamlit
Git
GitHub
Author

Kgobeane Mahlo

Aspiring Data Scientist | Data Analyst | Machine Learning Engineer

Skills

Python
SQL
Power BI
Machine Learning
Forecasting
Time Series Analysis
Streamlit
Git
GitHub
