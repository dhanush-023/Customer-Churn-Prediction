# 📊 Telecom Customer Churn Prediction

## 📘 Overview
This project analyzes telecom customer data to understand and predict **customer churn** using data visualization and machine learning.  
It identifies key factors influencing churn such as contract type, tenure, payment method, and monthly charges.

---

## 🎯 Objectives
- Explore customer data to find patterns linked to churn.  
- Visualize relationships between customer features and churn rate.  
- Build and evaluate machine learning models for churn prediction.

---

## 🧠 Dataset
**File:** `chrun.csv`

Key features include:
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`  
- `tenure`, `InternetService`, `OnlineSecurity`, `TechSupport`  
- `Contract`, `PaperlessBilling`, `PaymentMethod`  
- `MonthlyCharges`, `TotalCharges`, and target column `Churn`

---

## 📊 Insights from Analysis
- **Senior Citizens** churn more often than younger customers.  
- **Month-to-Month contracts** show the highest churn rate.  
- **Fiber optic** users churn more than DSL users.  
- **High monthly charges** lead to increased churn.  
- **Electronic check** payments and **paperless billing** have higher churn rates.  
- **Customers with dependents or long tenure** tend to stay longer.

---

## ⚙️ Tools & Libraries
- **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **Plotly**  
- **Scikit-learn** for model training and evaluation  

---

## 🧩 Model Summary
Two models were tested:
| Model | Accuracy |
|--------|-----------|
| Decision Tree | ~78% |
| Random Forest | ~83% |

The **Random Forest** model performed best overall.

---

## 🚀 How to Run
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
jupyter notebook "churn predictions (3).ipynb"
