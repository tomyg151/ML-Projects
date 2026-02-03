import pandas as pd


def get_clean_data(file_path):
    """
    This function loads the data and sets the Date as the Index.
    This is necessary for the monthly graphs to work!
    """
    print("--- Step 1: Loading and cleaning data ---")

    # 1. Load the Excel file
    df = pd.read_excel(file_path)

    # 2. Convert 'Invoice Date' to real date format
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])

    # 3. Group by day AND set the date as the index
    # We use .resample('D').sum() to make sure every day has a row
    daily_df = df.groupby('Invoice Date').agg({
        'Units Sold': 'sum',
        'Price per Unit': 'mean'
    })

    # --- THIS IS THE FIX ---
    # daily_df already has 'Invoice Date' as Index because of groupby
    # but we must ensure it has a frequency
    daily_df = daily_df.asfreq('D').fillna(0)

    # 4. Keep only the year 2021
    print("Filtering data for 2021...")
    df_2021 = daily_df.loc['2021-01-01':'2021-12-31'].copy()

    # 5. Add features (The 'hints' for the models)
    df_2021['day_of_month'] = df_2021.index.day
    df_2021['end_of_month'] = (df_2021.index.day >= 25).astype(int)

    holidays_2021 = ['2021-01-01', '2021-07-04', '2021-09-06', '2021-11-25',
                     '2021-11-26', '2021-12-24', '2021-12-25', '2021-12-31']
    df_2021['is_holiday'] = df_2021.index.isin(pd.to_datetime(holidays_2021)).astype(int)

    print(f"Success! Data is ready with DatetimeIndex.")
    return df_2021