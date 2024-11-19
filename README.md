Stroke Prediction

📋 Overview

This project aims to build a predictive model for stroke detection using machine learning techniques. By analyzing health-related data, the model predicts the likelihood of a stroke occurrence, helping healthcare professionals identify at-risk patients and take preventive measures.

🚀 Features

Data Preprocessing: Handles missing values, encodes categorical data, and normalizes numerical features.
Imbalanced Data Handling: Uses techniques like SMOTE to manage class imbalance.
Model Evaluation: Compares multiple models (e.g., Logistic Regression, Random Forest, Linear Discriminant Analysis) using ROC-AUC as the evaluation metric.
Pipeline Automation: Implements an end-to-end pipeline for data transformation, oversampling, and model training.
Best Model Selection: Saves the best-performing model using pickle for future use.


📊 Dataset

Source: Kaggle Stroke Prediction Dataset
Description: The dataset contains health-related features such as age, BMI, hypertension, and smoking status for stroke prediction.
Target Variable: stroke (0 = No stroke, 1 = Stroke)

📈 Results
Best Model: Linear Discriminant Analysis
ROC-AUC Score: 0.837 


💡 Acknowledgments

data science brian

Thanks to Kaggle for the dataset.

Inspired by healthcare professionals working to prevent strokes.
