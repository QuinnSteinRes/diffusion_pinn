import pandas as pd

# Read your CSV file
df = pd.read_csv("/home/qpy/miniconda3/envs/nnmodelling/diffusion_pinn/src/diffusion_pinn/data/intensity_time_series_spatial_temporal.csv")

# Print basic information
print("DataFrame Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nUnique values in 't' column:")
print(df['t'].unique())
