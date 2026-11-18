
import pandas as pd

# 1. Load dataset
df = pd.read_csv("bengaluru_traffic.csv")

# 2. Convert date
df["Date"] = pd.to_datetime(df["Date"])

# 3. Sort records by location and time
df = df.sort_values(
    by=["Location", "Date", "Hour"]
).reset_index(drop=True)

# 4. Create previous observation features
grouped = df.groupby("Location")

previous_columns = [
    "Total_Vehicles",
    "Cars",
    "Bikes",
    "Buses",
    "Autos",
    "Avg_Speed",
    "Weather"
]

for column in previous_columns:
    df["Previous_" + column] = (
        grouped[column].shift(1)
    )

# 5. # Create target: next observation's congestion
df["Next_Congestion"] = grouped["Congestion_Level"].shift(-1)

# Store the date of the next observation
df["Next_Date"] = grouped["Date"].shift(-1)

# 6. Create information about the next time slot
df["Next_Hour"] = grouped["Hour"].shift(-1)
df["Next_Day_Type"] = grouped["Day_Type"].shift(-1)

# 7. Remove rows without previous or next observations
df = df.dropna().reset_index(drop=True)

# 8. Select useful prediction features
features = [
    "Location",
    "Next_Hour",
    "Next_Day_Type",
    "Previous_Total_Vehicles",
    "Previous_Cars",
    "Previous_Bikes",
    "Previous_Buses",
    "Previous_Autos",
    "Previous_Avg_Speed",
    "Previous_Weather"
]

target = "Next_Congestion"

# 9. Create final dataset
processed_df = df[["Date", "Next_Date"] + features + [target]].copy()

# 10. Save processed dataset
processed_df.to_csv(
    "traffic_prediction_data.csv",
    index=False
)

# 11. Display results
print("PREPROCESSING COMPLETED!")

print("\nProcessed Dataset Shape:")
print(processed_df.shape)

print("\nFirst Five Records:")
print(processed_df.head())

print("\nTarget Distribution:")
print(processed_df[target].value_counts())

print("\nMissing Values:")
print(processed_df.isnull().sum())

print("\nSaved as traffic_prediction_data.csv")
