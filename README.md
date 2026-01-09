# 🏠 House Price Prediction

## 📌 Project Overview
This project aims to predict house prices using supervised machine learning techniques.  
The workflow includes **exploratory data analysis (EDA)**, **data preprocessing**, **training multiple regression models**, and a **detailed comparison** to select the most suitable model.

The goal is not only to achieve good predictive performance, but also to **justify the choice of the final model** based on quantitative metrics and interpretability.


---

## 📊 Exploratory Data Analysis (EDA)
The EDA phase focuses on understanding the dataset and identifying the main drivers of house prices.

Key steps:
- Analysis of the **target variable distribution**
- Detection of **outliers**
- Exploration of **numerical and categorical features**
- Correlation analysis between features and the target variable
- Visualization using histograms, boxplots, and correlation heatmaps

**Key insights:**
- House prices show strong correlations with features such as living area, overall quality, and location
- Some variables exhibit skewed distributions, requiring transformations
- Missing values were identified and handled during preprocessing

---

## 🛠 Data Preprocessing
The following preprocessing steps were applied:
- Handling missing values (imputation)
- Encoding categorical variables
- Feature scaling
- Train-test split

---

## 🤖 Models Trained
Several regression models were implemented and evaluated:

- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

Each model was evaluated using:
- **R² score**
- **Mean Squared Error (MSE)**
- Cross-validation

---

## 📈 Model Comparison & Selection
A comparative analysis was conducted to determine the most appropriate model.

| Model | R² | MSE | Strengths | Limitations |
|------|----|----|-----------|-------------|
| Linear Regression | Low | High | Simple, interpretable | Underfitting |
| Ridge / Lasso | Medium | Medium | Regularization | Sensitive to hyperparameters |
| Random Forest | High | Low | Handles non-linearity | Less interpretable |
| Gradient Boosting | Very High | Very Low | Best performance | Higher complexity |

### ✅ Final Model Choice
The **Gradient Boosting Regressor** was selected as the final model because:
- It achieved the best overall predictive performance
- It effectively captures non-linear relationships
- It provides a good bias–variance trade-off compared to other models

---

## 📌 Results
- Best R² score: **XX.XX**
- Lowest MSE: **XX.XX**
- The model demonstrates strong generalization on unseen data

---

## 🧪 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---



