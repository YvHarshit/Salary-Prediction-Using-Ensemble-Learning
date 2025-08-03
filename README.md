# Salary-Prediction-Using-Ensemble-Learning
Explore AI/ML through Project Making.


## Project Overview

This project aims to build a robust machine learning model that accurately predicts employee salaries based on a variety of features. Leveraging the power of **Ensemble Learning**, we explore different regression techniques to find the most effective approach for salary forecasting.

The goal is to provide insights into how factors like job title, experience level, employment type, company location, and size influence salary, offering a valuable tool for both employers and job seekers.

## Features

* **Data Preprocessing:** Handles missing values, encodes categorical features (One-Hot Encoding), and scales numerical features (StandardScaler) using `ColumnTransformer` and `Pipeline`.
* **Multiple Regression Models:** Explores various individual and ensemble regression algorithms:
    * Linear Regression
    * Decision Tree Regressor
    * Support Vector Regressor (SVR)
    * K-Nearest Neighbors Regressor
    * **Ensemble Models:**
        * Random Forest Regressor
        * Gradient Boosting Regressor
        * XGBoost Regressor
        * LightGBM Regressor
        * **Stacked Regressor** (combining multiple base models with a meta-model)
* **Model Evaluation:** Utilizes key regression metrics (Mean Absolute Error, Root Mean Squared Error, R-squared) and K-Fold Cross-Validation for robust performance assessment.
* **Salary Prediction Function:** Includes a utility function to predict salaries for new, unseen employee data.

## Technologies Used

* **Python**
* **Data Manipulation & Analysis:**
    * `pandas`
    * `numpy`
* **Machine Learning:**
    * `scikit-learn` (for preprocessing, pipelines, base models, and ensemble methods like RandomForest, GradientBoosting, Stacking)
    * `xgboost`
    * `lightgbm`
* **Data Visualization (for EDA, not explicitly in provided notebook, but good practice):**
    * `matplotlib`
    * `seaborn`
* **Environment:**
    * Google Colaboratory (`google.colab.files` for file upload)

## Dataset

The project utilizes the `Latest_Data_Science_Salaries.csv` dataset. This dataset contains various features related to data science roles and their corresponding salaries.

**Key Features in the Dataset:**
* `work_year`: The year the salary was recorded.
* `experience_level`: Level of experience (e.g., Entry-level, Mid-level, Senior, Executive).
* `employment_type`: Type of employment (e.g., Full-time, Part-time).
* `job_title`: The specific job role.
* `salary_currency`: The currency of the salary.
* `salary`: The salary amount in its original currency.
* `salary_in_usd`: The salary converted to USD (target variable).
* `employee_residence`: Country of employee residence.
* `remote_ratio`: Percentage of remote work.
* `company_location`: Country of the company.
* `company_size`: Size of the company (e.g., Small, Medium, Large).


## Methodology

The project follows a standard machine learning pipeline:

1.  **Data Loading:** The dataset is loaded into a pandas DataFrame.
2.  **Data Preprocessing:**
    * Missing values are handled (rows with NaNs are dropped for simplicity).
    * Features are identified as numerical or categorical.
    * A `ColumnTransformer` is used within a `Pipeline` to apply `StandardScaler` to numerical features and `OneHotEncoder` to categorical features.
3.  **Model Definition:** A comprehensive set of regression models, including various ensemble techniques (Random Forest, Gradient Boosting, XGBoost, LightGBM, and a Stacking Regressor), are defined within `scikit-learn` pipelines.
4.  **Model Training & Evaluation:**
    * The data is split into training and testing sets.
    * Each model is trained on the training data.
    * Performance is evaluated on the test set using MAE, RMSE, and R-squared.
    * **K-Fold Cross-Validation** is performed to ensure robust evaluation and reduce bias from a single train-test split.
5.  **Prediction:** A helper function `predict_new_salary` is provided to demonstrate how to use the best-performing model to predict salaries for new, unseen data points.

## Results

The notebook trains and evaluates multiple regression models. While the provided Colab output indicates `NaN` for R-squared on the test set (likely due to the small dummy dataset used when the actual CSV isn't uploaded or very few test samples), the cross-validation metrics provide a more reliable indication of performance.

The model with the best performance (lowest MAE/RMSE and highest R-squared) would typically be selected. In a real-world scenario with a larger dataset, these models would show significant differences in their predictive power.

*Refer to the Google Colab notebook for detailed evaluation metrics of each model.*

## How to Run the Project

This project is designed to be run in Google Colaboratory.

1.  **Open the Notebook:** Click on the Google Colab link:
    [https://colab.research.google.com/drive/11GoV6o8FmMMhUufO1InulwXnaZqUXbkv?usp=sharing](https://colab.research.google.com/drive/11GoV6o8FmMMhUufO1InulwXnaZqUXbkv?usp=sharing)

2.  **Save a Copy:** Go to `File > Save a copy in Drive` to create your editable version.

3.  **Upload Dataset:**
    * In the Colab notebook, navigate to the "Data Loading" section.
    * Run the cell that prompts you to upload the `Latest_Data_Science_Salaries.csv` file. You will need to click "Choose Files" and select your dataset.

4.  **Run All Cells:** Go to `Runtime > Run all` to execute the entire notebook.

5.  **Explore Results:** Review the output cells for data preprocessing steps, model training progress, evaluation metrics, and sample salary predictions.

## Future Enhancements

* **Hyperparameter Tuning:** Implement more extensive hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV` for all ensemble models.
* **Feature Engineering:** Explore creating more sophisticated features (e.g., text embeddings for `job_title`, interaction terms).
* **More Robust Missing Value Imputation:** Implement advanced imputation techniques (e.g., MICE, K-NN Imputer).
* **Outlier Detection and Treatment:** Analyze and handle outliers in salary or numerical features.
* **Model Interpretability:** Use tools like SHAP or LIME to explain individual predictions and feature contributions.
* **Deployment:** Integrate the trained model into a web application (e.g., Flask/Streamlit) for interactive predictions.
* **Larger Dataset:** Obtain and utilize a larger, more diverse salary dataset for improved model generalization.


## Thanks....
