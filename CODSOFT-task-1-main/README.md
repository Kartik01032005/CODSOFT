# 💳 Credit Card Fraud Detection

A machine learning project to **identify fraudulent credit card transactions**.  
This project demonstrates how to handle **imbalanced datasets**, apply **feature preprocessing**, train models, and evaluate performance using various classification metrics.

---

## 📌 Project Overview

- Preprocess and normalize transaction data  
- Handle **class imbalance** using **SMOTE** (Synthetic Minority Oversampling Technique) or random oversampling  
- Split the dataset into **training** and **testing** sets  
- Train classification algorithms:
  - Logistic Regression  
  - Random Forest  
- Evaluate models using:
  - **Precision**
  - **Recall**
  - **F1-score**
  - **ROC-AUC**
  - **PR-AUC**  
- Save the best-performing model for future predictions  
- Display top 15 transactions predicted as fraud

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Matplotlib  
- Joblib  

---

## 📂 Dataset

The dataset used is the popular **[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)**, which contains transactions made by European cardholders in September 2013.  

- **Rows:** 284,807 transactions  
- **Features:** 30 (anonymized as `V1, V2, ..., V28`, plus `Time` and `Amount`)  
- **Target:** `Class`  
  - `0` → Genuine transaction  
  - `1` → Fraudulent transaction  

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
