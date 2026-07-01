import pandas as pd

# Load the data
df = pd.read_csv('Customer_Data.csv')

# Calculate missing values and percentages
missing_counts = df.isnull().sum()
missing_percentages = (df.isnull().sum() / len(df)) * 100

missing_audit = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Percentage': missing_percentages
})

# Identify columns with > 0% and < 5% missing values
low_missing_cols = missing_audit[(missing_audit['Percentage'] > 0) &
                                 (missing_audit['Percentage'] < 5.0)].index

# Drop rows containing missing values in ONLY these specific columns
df_clean = df.dropna(subset=low_missing_cols)

# Save the cleaned dataset
df_clean.to_csv('Customer_Data_Cleaned.csv', index=False)
print("Data cleaned and saved to Customer_Data_Cleaned.csv")
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_clean.shape}")
