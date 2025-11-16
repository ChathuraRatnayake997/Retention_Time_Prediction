# Retention Time Prediction with Machine Learning

[![Visit the Live App](https://img.shields.io/badge/Visit_the-Live_App-blue?style=for-the-badge&logo=streamlit)](https://chathuraratnayake997.github.io/Retention_Time_Prediction/)

A machine learning project for predicting Retention times using a **Ridge Regression** model.

---

###  Live Project

You can interact with the live prediction app at the following link:

**[https://chathuraratnayake997.github.io/Retention_Time_Prediction/](https://chathuraratnayake997.github.io/Retention_Time_Prediction/)**

---

### Model Performance

The performance of the trained Ridge Regression model is summarized by the following metrics:

| Metric                | Score   |
| --------------------- | ------- |
| **R² Score**          | `0.633` |
| **MAE** (minutes)     | `1.605` |
| **RMSE** (minutes)    | `2.262` |

-   **R² Score (Coefficient of Determination):** Indicates that the model explains approximately **63.3%** of the variance in the retention times based on the provided features.
-   **MAE (Mean Absolute Error):** On average, the model's predictions are incorrect by approximately **1.6 minutes**.
-   **RMSE (Root Mean Squared Error):** A measure of the error magnitude which gives a higher weight to larger errors.

---

### Tech Stack

-   **Model:** Scikit-learn (Ridge Regression)
-   **Data Manipulation:** Pandas, NumPy
-   **Web App:** Html, css, js
