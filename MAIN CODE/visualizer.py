import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_thought_process(full_df, df_2021):
    """
    Shows the step-by-step thinking: from all data to 2021 daily analysis.
    """
    print("Generating Thought Process graphs...")

    # 1. Full Data Graph (2020-2021)
    plt.figure(figsize=(12, 5))
    # We use full_df.values because the index is the date
    plt.plot(full_df.index, full_df.values, color='gray', alpha=0.5)
    plt.title("Step 1: All Sales Data (2020-2021) - Identifying Trends")
    plt.show()

    # 2. Filtered 2021 Data
    plt.figure(figsize=(12, 5))
    plt.plot(df_2021.index, df_2021['Units Sold'], color='blue')
    plt.title("Step 2: Focused Analysis - Daily Sales in 2021")
    plt.show()

    # 3. Sales by Day of the Week
    df_2021['day_name'] = df_2021.index.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_2021, x='day_name', y='Units Sold', order=order)
    plt.title("Step 3: Sales Analysis by Day of the Week")
    plt.show()

    # 4. Observed Daily Sales Analysis (Day of Month)
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df_2021, x='day_of_month', y='Units Sold', ci=None)
    plt.title("Step 4: Observed Daily Sales Analysis (By Day of Month)")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_final_comparison(y_test, arimax_pred, xgb_pred):
    """ Final comparison between reality and the two models. """
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.values, label='Observed Reality', color='black', linewidth=2)
    plt.plot(arimax_pred.values, label='ARIMAX Prediction', linestyle='--')
    plt.plot(xgb_pred, label='XGBoost Prediction', color='green')
    plt.title("Final Step: Models vs. Observed Daily Sales")
    plt.legend()
    plt.show()