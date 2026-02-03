import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_manager import get_clean_data
from visualizer import plot_thought_process, plot_final_comparison
from models import run_arimax_model, run_xgboost_model

# 1. Load data
file_name = 'Adidas.xlsx'
# We need full data for the first graph and filtered for the rest
full_data = pd.read_excel(file_name)
full_data['Invoice Date'] = pd.to_datetime(full_data['Invoice Date'])
full_daily = full_data.groupby('Invoice Date')[['Units Sold']].sum()

df = get_clean_data(file_name)

# --- SHOW FEATURES TABLE ---
print("\n--- Feature Selection: The Data Table for Modeling ---")
# Show the first few rows of the table we built
features_columns = ['Units Sold', 'Price per Unit', 'day_of_month', 'end_of_month', 'is_holiday']
print(df[features_columns].head(10))

# 2. Visualizing Thought Process
plot_thought_process(full_daily, df)

# 3. Split Data
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]
X_train, y_train = train[features_columns[1:]], train['Units Sold']
X_test, y_test = test[features_columns[1:]], test['Units Sold']

# 4. Run Models
p_ari, _ = run_arimax_model(y_train, X_train, X_test)
p_xgb, _ = run_xgboost_model(X_train, y_train, X_test)

# --- SHOW FINAL METRICS TABLE ---
def get_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'Model': model_name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

results = pd.DataFrame([
    get_metrics(y_test, p_ari, 'ARIMAX'),
    get_metrics(y_test, p_xgb, 'XGBoost')
])

pd.options.display.float_format = '{:.2f}'.format

print("\n--- Final Model Comparison Statistics ---")
print(results.to_string(index=False))

# 5. Final Comparison Graph
plot_final_comparison(y_test, p_ari, p_xgb)