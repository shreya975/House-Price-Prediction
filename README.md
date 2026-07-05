# 🏡 House Price Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An AI-powered web application that predicts **Bengaluru house prices** using **Machine Learning**. This project leverages **Linear Regression**, **One-Hot Encoding**, and an interactive **Streamlit dashboard** to estimate property prices based on user inputs such as location, total area, bedrooms, and bathrooms.

---

# 🌐 Live Demo

### 🚀 Click below to use the application

## 👉 https://house-price-prediction-cbrveafrrwmzxsneuttr6u.streamlit.app/

---

# 📌 Project Overview

Real estate pricing is influenced by several factors including location, property size, and amenities. Estimating the correct market value manually can be difficult.

This project builds a Machine Learning model trained on the **Bengaluru House Price Dataset** to predict house prices instantly through a modern web interface.

The application follows a complete Data Science workflow:

- Data Collection
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Machine Learning Model Training
- Interactive Visualization
- Web Deployment using Streamlit

---

# ✨ Features

✅ Predict Bengaluru House Prices

✅ AI-powered Property Valuation

✅ Interactive Streamlit Dashboard

✅ Beautiful Modern UI

✅ Real-time Price Prediction

✅ Property Summary

✅ Market Insights

✅ Price Distribution Analysis

✅ Area vs Price Visualization

✅ BHK Analysis

✅ Top Locations Analysis

✅ Download Prediction Report

✅ Prediction History

---

# 📊 Dataset Information

**Dataset Name**

Bengaluru House Price Dataset

The dataset contains thousands of Bengaluru property listings with information such as:

- Property Location
- Total Square Feet
- Number of Bedrooms
- Number of Bathrooms
- Property Price
- Area Information

---

# 🧹 Data Preprocessing

The dataset undergoes several preprocessing steps before training.

### Data Cleaning

- Removed unnecessary columns
- Removed missing values
- Converted area ranges into numeric values
- Extracted Bedrooms from Size column
- Removed invalid records

### Feature Engineering

- Created Bedrooms feature
- Converted total_sqft into numerical values
- Grouped rare locations into "Other"
- Removed outliers using percentile method

### Final Features

- Location
- Total Square Feet
- Bathrooms
- Bedrooms

Target Variable

- Price (Lakhs ₹)

---

# 🤖 Machine Learning Model

The project uses

### Linear Regression

combined with

### One-Hot Encoding

using a Scikit-Learn Pipeline.

Pipeline

```
OneHotEncoder
        ↓
Linear Regression
        ↓
Price Prediction
```

---

# 📈 Project Workflow

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Outlier Removal
   │
   ▼
Train Test Split
   │
   ▼
One Hot Encoding
   │
   ▼
Linear Regression
   │
   ▼
Model Evaluation
   │
   ▼
Prediction
   │
   ▼
Streamlit Deployment
```

---

# 🛠️ Tech Stack

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Scikit-Learn
- Category Encoders
- Plotly
- Streamlit

---

# 📂 Project Structure

```
House-Price-Prediction/

│── app.py
│── Bengaluru_House_Data.csv
│── House Price Prediction Model.ipynb
│── requirements.txt
│── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/shreya975/House-Price-Prediction.git
```

## Move into Project

```bash
cd House-Price-Prediction
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Run Streamlit

```bash
streamlit run app.py
```

---

# 💻 Application Screens

## Home Page

(Add Screenshot Here)

---

## Prediction Dashboard

(Add Screenshot Here)

---

## Analytics Dashboard

(Add Screenshot Here)

---

# 📊 Visualizations

The application includes

- Price Distribution
- Area vs Price Scatter Plot
- BHK Distribution
- Top Locations
- Feature Importance
- Property Score
- AI Confidence Meter
- Affordability Meter

---

# 🎯 Model Inputs

Users provide

- 📍 Location
- 📐 Total Square Feet
- 🛏 Bedrooms
- 🛁 Bathrooms

The application predicts

🏡 Estimated House Price (₹ Lakhs)

---

# 🚀 Future Improvements

- Random Forest Regression
- XGBoost Regression
- LightGBM
- Deep Learning Models
- Google Maps Integration
- Property Recommendation System
- User Login
- Database Support
- API Deployment
- Mobile Responsive Version

---

# 📈 Deployment

The application is deployed using **Streamlit Cloud**

### Live Website

👉 https://house-price-prediction-cbrveafrrwmzxsneuttr6u.streamlit.app/

---

# 👩‍💻 Author

## Shreya Mahajan

### GitHub

https://github.com/shreya975

### LinkedIn

https://www.linkedin.com/in/shreya-mahajan-b38b28385/
