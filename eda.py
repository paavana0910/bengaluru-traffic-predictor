
import pandas as pd

# Load dataset
df = pd.read_csv("bengaluru_traffic.csv")

print("DATASET SHAPE:")
print(df.shape)

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

print("\nCONGESTION DISTRIBUTION:")
print(df["Congestion_Level"].value_counts())
