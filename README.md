# Adidas Sales Forecasting: ARIMAX vs. XGBoost
This project focuses on predicting daily sales units for Adidas US retail data using supervised machine learning and statistical time-series models. The goal is to optimize supply chain management by providing accurate demand forecasts.

## Project Overview
In the retail industry, balancing between stockouts and overstocking is a critical challenge. This project compares a classical statistical approach (ARIMAX) with a modern gradient boosting machine learning model (XGBoost) to determine the most effective method for daily sales prediction.

## Dataset
The analysis is based on the Adidas Sales Dataset, which includes sales records across the United States for the years 2020-2021.
The original dataset contained raw transaction-level data with 13 columns including retailers, regions, and product types.
* **Source:** [Adidas Sales Dataset on Kaggle](https://www.kaggle.com/datasets/heemalichaudhari/adidas-sales-dataset)
---
## Project Pipeline & Decision Making
### 1. Data Cleaning & Preprocessing Steps:
1. **Time Aggregation:** The raw transactional data was grouped by `Invoice Date` to create a daily sales time-series.
2. **Outlier Removal 2020(The COVID-19 Factor) :** During initial EDA, it was discovered that 2020 data was highly inconsistent due to the global pandemic's impact on retail. To ensure the model learns stable consumer behavior, **the year 2020 was filtered out**, focusing the analysis on the 2021 recovery period.
<img width="1322" height="418" alt="image" src="https://github.com/user-attachments/assets/f57199e3-c7ec-4359-b85d-175980911162" />
3. **Feature Engineering:** We transformed the date into cyclical features to help the models understand time:
   * `day_of_month` (1-31)
   * `is_holiday` (Binary)
   * `end_of_month` (Binary)
4.**Aggregation:** Data was resampled to a daily level to identify high-frequency patterns.
**Ensure the is_holiday logic is applied to the Invoice Date before training**

* **Key Features:**
Beyond basic sales data, several exogenous features were engineered to improve forecasting:
* `Units Sold`: Daily sum (Target).
* `Price per Unit`: Daily average.
* `day_of_month`: Capturing within-month cyclicality.
* `end_of_month`: Binary flag for salary-driven peaks.
* `is_holiday`: Binary flag for US public holidays.
 ---
## Methodology
The project follows a standard Data Science pipeline:
1. **Exploratory Data Analysis (EDA):** Identifying monthly seasonality.
2. **Feature Engineering:** Creating time-based features (Day of Month, Holidays, Price changes).
3. **Modeling:**
   * **ARIMAX:** A linear parametric model using exogenous variables.
   * **XGBoost:** A non-linear ensemble model designed for high performance.
4. **Evaluation:** Comparing models using $R^2$ and Mean Absolute Error (MAE).

### 3. Visual Analysis
The project follows a visualization-first approach:
* **Initial Trend Analysis:** Identifying seasonality and the decision to remove 2020.
* **Model Comparison:** Plotting predicted vs. actual sales to visualize where XGBoost captures spikes that ARIMAX misses.


---
## Results
### Business Metrics Summary
The model performance was evaluated using standard regression metrics. XGBoost showed a significant improvement in capturing market volatility.

| Model | MAE | MSE | RMSE | $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **ARIMAX** | 3,600.42 | 18,887,476.06 | 4,345.97 | 0.11 |
| **Gradient Boosting** | **2,903.22** | **14,544,641.32** | **3,813.74** | **0.31** |
<img width="1182" height="584" alt="image" src="https://github.com/user-attachments/assets/a5d29b2f-6ced-4e2f-b4ef-775989dba33e" />

XGBoost outperformed ARIMAX by successfully capturing non-linear patterns and complex seasonality, reducing the Mean Absolute Error by approximately 19%.
### Key Insights
* While ARIMAX only reacted to holidays, XGBoost learned the daily "rhythm" of the month, XGBoost **reduced the Mean Absolute Error (MAE) by approximately 19%** compared to ARIMAX.
* **Timing over Price:** The models revealed that **timing - Seasonality** (Day of Month) had a higher predictive power than spot price changes, suggesting Adidas customers are driven by cyclic purchasing habits.
## 🧠 Feature Importance: Statistical vs. Machine Learning Logic
A key finding of this project is how differently each model interprets the features. While ARIMAX relies on explicit binary "cues" (like holidays), XGBoost independently identifies complex patterns within the month.

| Feature | ARIMAX Importance (%) | XGBoost Importance (%) |
| :--- | :---: | :---: |
| **day_of_month** | 1.41 | **88.79** |
| **Price per Unit** | 1.45 | 11.21 |
| **end_of_month** | 44.51 | 0.00 |
| **is_holiday** | **52.63** | 0.00 |

### Analysis:
* **XGBoost (The Strategist):** Focused almost exclusively on `day_of_month` (88.7%). It successfully learned the "rhythm" of sales (e.g., paydays or recurring buying habits) without needing external labels.
* **ARIMAX (The Conservative):** Relied heavily on pre-defined events like `is_holiday` (52.6%) and `end_of_month` (44.5%). As a linear model, it struggled to understand the continuous cyclicality of the month and "waited" for specific flags to adjust the forecast.
<img width="1209" height="582" alt="image" src="https://github.com/user-attachments/assets/6109ec91-37e8-4c6c-a296-c33cf358f865" />

### 📊 The Power of the "Day of Month" Feature
The chart below visualizes why XGBoost assigned nearly 89% importance to the `day_of_month` variable. Unlike the statistical model, the ML model identified clear, recurring sales peaks throughout the month.
<img width="1409" height="418" alt="image" src="https://github.com/user-attachments/assets/3cc98c32-3a2e-4899-abf9-62620b628951" />
> **Insight:** Notice how sales aren't random; there are specific cycles within the 31-day window. This is exactly what the Gradient Boosting model captured to outperform the ARIMAX baseline.
---
## How to Run the Project
### Prerequisites
You can run the analysis directly in your browser without any local setup:
1. **Open in Google Colab:** [Click here to open](https://colab.research.google.com/drive/1XD-PZYz6xk8iXU_vCv-GZManiJDDq2AG)
2. **Upload Data:** Download the dataset from Kaggle and upload the Adidas.xlsx (or CSV version) to the Colab environment when prompted.
3. **Execution:** Run all cells (Runtime -> Run all) to see the data processing, visualizations, and model comparisons.
## Installation (Local)
If you prefer to run it locally, run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost scipy
```
### Libraries Used:
* **Data Manipulation:** pandas, numpy

* **Visualization:** matplotlib, seaborn

* **Statistical Modeling:** statsmodels (for ARIMAX)

* **Machine Learning:** scikit-learn, xgboost (Gradient Boosting)

* **Signal Processing:** scipy (for peak detection in seasonality)


## 👥 Authors & Acknowledgments

This project was developed as part of the **Supervised Machine Learning** course.

* **Project Authors:** Tom Grundland & Ido Armanchik
* **Lecturer:** Dr. Sarah Ita Levitan
* **Academic Institution:** Bar-Ilan University, Department of Information Science

---

### 🙏 Thank You
Thank you for taking the time to explore this project! If you have any questions or suggestions regarding the methodology or the results, feel free to reach out or open an issue in this repository.

**Happy Coding! 🚀**

