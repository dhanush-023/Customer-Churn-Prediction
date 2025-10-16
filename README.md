# 📊 Telecom Customer Churn Prediction

## 📘 Project Overview
This project focuses on predicting **customer churn** for a telecom company using **Exploratory Data Analysis (EDA)** and **Machine Learning**.  
The aim is to understand which customer behaviors or service attributes contribute most to churn and to build predictive models to identify customers likely to leave.

The project is implemented entirely in a **Jupyter Notebook** — `churn predictions (3).ipynb` — using the dataset `chrun.csv`.

---

## 🎯 Key Objectives
- Perform **data cleaning and preprocessing** on telecom customer data.  
- Conduct **exploratory data analysis (EDA)** to identify key churn patterns.  
- Visualize trends between customer demographics, service types, and churn rate.  
- Train **machine learning models** (Decision Tree and Random Forest) to predict churn.  
- Evaluate model performance using accuracy, confusion matrix, and classification metrics.

---

## 🧠 Dataset Details
**File used:** `chrun.csv`

Each row represents one customer record with:
- **Demographic info:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`  
- **Account info:** `tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`  
- **Services:** `PhoneService`, `InternetService`, `TechSupport`, `StreamingTV`, etc.  
- **Charges:** `MonthlyCharges`, `TotalCharges`  
- **Target variable:** `Churn` (Yes/No)

---

## 🧩 Steps and Techniques Used

### 1️⃣ Data Loading & Inspection
- Imported data using **Pandas**.
- Inspected shape, column types, and missing values.
- Converted `TotalCharges` to numeric (handled spaces/NaNs).

### 2️⃣ Data Cleaning & Encoding
- Removed unnecessary columns like `customerID`.
- Converted categorical variables to numeric using:
  - **Label Encoding** for binary columns.
  - **One-Hot Encoding** (if required) for multi-category columns.

### 3️⃣ Exploratory Data Analysis (EDA)
Used **Matplotlib**, **Seaborn**, and **Plotly** for visualizations:
- **Pie charts:** Gender distribution, churn percentage, service usage.
- **Bar plots:** Churn vs Internet type, Tech Support, Contract, Payment Method.
- **Histograms:** Tenure and Monthly Charges distribution.
- **Heatmap:** Correlation between numeric features.

📈 **Insights from EDA:**
- Around **26%** of customers churned overall.  
- **Senior Citizens** and **Fiber Optic** users churn more frequently.  
- Customers with **shorter tenure (<1 year)** are more likely to leave.  
- **Month-to-Month** contracts have the highest churn rate (~45%).  
- **Electronic Check** payment users churn more (~35%) compared to card/bank transfers.  
- Lack of **Tech Support** or **Online Security** strongly correlates with churn.  
- Customers with **dependents or long contracts** tend to stay longer.

### 4️⃣ Feature Scaling
- Applied **StandardScaler** from `scikit-learn` to normalize numerical variables for model training.

### 5️⃣ Model Building
Two supervised learning models were implemented:
- **Decision Tree Classifier**
- **Random Forest Classifier**

Steps:
1. Split the dataset into **train** and **test** sets (typically 80/20).  
2. Trained models on training data using `fit()`.  
3. Predicted churn on test data.  

### 6️⃣ Model Evaluation
Evaluated models using:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)

📊 **Results Summary:**
| Model | Accuracy | Remarks |
|--------|-----------|----------|
| Decision Tree | ~78% | Simple but prone to overfitting |
| Random Forest | ~83% | Best performance and good generalization |

---

## ⚙️ Technologies & Libraries Used
- **Python 3.x**
- **NumPy** – Numerical computations  
- **Pandas** – Data manipulation  
- **Matplotlib / Seaborn / Plotly** – Data visualization  
- **Scikit-learn** – Machine learning models, preprocessing, and evaluation  

---

| Visualization             | Interpretation                                                   |
| ------------------------- | ---------------------------------------------------------------- |
| Gender & Churn Pie Chart  | Nearly equal male/female ratio; gender not significant in churn. |
| Tenure Distribution       | Shorter-tenure customers churn more.                             |
| Internet Service vs Churn | Fiber optic customers show higher churn.                         |
| Contract Type vs Churn    | Month-to-month users churn the most.                             |
| Payment Method vs Churn   | Electronic check users show higher churn.                        |
| Correlation Heatmap       | Tenure and total charges correlate negatively with churn.        |
| Confusion Matrix          | Shows that the model predicts non-churn cases better than churn. |

---

🏁 Conclusion

The project successfully demonstrates how data analytics and machine learning can be applied to understand customer churn behavior.
The Random Forest model achieved strong accuracy and highlighted that contract type, service quality, and billing preferences are the major churn drivers.
These insights can help telecom providers develop better retention strategies and improve customer satisfaction.

---

🚀 Future Enhancements

- Apply GridSearchCV for hyperparameter tuning.

- Handle data imbalance using SMOTE or class weighting.

- Deploy model via Streamlit or Flask web app.

- Integrate real-time prediction dashboard.

  ---
  

👤 Author: Dhanush P
💼 Data Science | Machine Learning | Visualization
📧 For collaboration or feedback — feel free to connect via GitHub.
