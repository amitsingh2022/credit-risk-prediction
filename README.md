# 🏦 Credit Risk Prediction App

Predict the probability of a credit card customer defaulting next month using machine learning.

###  🚀 Live Demo

🔗 creditriskpredictionapp.streamlit.app

### 📘 Overview

This project builds a machine learning pipeline to predict credit card default risk based on demographic, behavioral, and financial data.

The app allows users to input customer details (like credit limit, past payments, utilization rate, etc.) and get a real-time prediction of default probability.

### 🧠 Features

✅ Data Cleaning & Preprocessing (handled outliers, missing values)
✅ Feature Engineering (utilization ratio, average payment ratio)
✅ Model Training using XGBoost
✅ Hyperparameter Tuning to improve recall from 33% → 59%
✅ Deployment on Streamlit Cloud for live predictions
✅ Interactive UI with sliders and dropdowns

### ⚙️ Tech Stack
Category	Tools
Language	Python
Libraries	Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
Deployment	Streamlit
Version Control	Git, GitHub

### 🧩 Model Workflow
flowchart LR
A[Data Collection] --> B[EDA & Cleaning]
B --> C[Feature Engineering]
C --> D[XGBoost Model Training]
D --> E[Hyperparameter Tuning]
E --> F[Evaluation - Recall, F1, ROC-AUC]
F --> G[Streamlit Deployment]

### 📊 Model Performance
Metric	Before Tuning	After Tuning
Accuracy	83%	84%
Recall (Defaulters)	33%	59%
ROC-AUC	0.81	0.87

### 🖥 How to Run Locally
# Clone repo
git clone https://github.com/amitsingh2022/credit-risk-prediction-app.git
cd credit-risk-prediction-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

### 🌟 Future Improvements

Add Flask API + Docker deployment version

Implement MLflow for experiment tracking

Improve recall further with SMOTE or class weights

Integrate CI/CD pipeline

## 👨‍💻 Author

### Amit Singh
Machine Learning Engineer | Data Enthusiast
#### 🔗 GitHub
