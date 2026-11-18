
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
df = pd.read_csv("traffic_prediction_data.csv")

print("Dataset Loaded:", df.shape)

# 2. Convert dates
df["Date"] = pd.to_datetime(df["Date"])
df["Next_Date"] = pd.to_datetime(df["Next_Date"])

# 3. Sort by the date we are predicting
df = df.sort_values("Next_Date").reset_index(drop=True)

# 4. Split training and testing data using Next_Date
split_date = df["Next_Date"].quantile(0.8)

train_df = df[df["Next_Date"] <= split_date]
test_df = df[df["Next_Date"] > split_date]

# 5. Separate input features and target
X_train = train_df.drop(
    columns=["Date", "Next_Date", "Next_Congestion"]
)

y_train = train_df["Next_Congestion"]

X_test = test_df.drop(
    columns=["Date", "Next_Date", "Next_Congestion"]
)

y_test = test_df["Next_Congestion"]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Training data through:", split_date.date())
print("Testing data after:", split_date.date())

# 6. Identify categorical and numerical columns
categorical_features = [
    "Location",
    "Next_Day_Type",
    "Previous_Weather"
]

numerical_features = [
    "Next_Hour",
    "Previous_Total_Vehicles",
    "Previous_Cars",
    "Previous_Bikes",
    "Previous_Buses",
    "Previous_Autos",
    "Previous_Avg_Speed"
]

# 7. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

# 8. Define models
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

best_accuracy = 0
best_model = None
best_model_name = ""

# 9. Train and evaluate models
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n----------------------------")
    print(name)
    print("----------------------------")

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_model_name = name

# 10. Save the best model
print("\n===== MODEL COMPARISON =====")

print("Best Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")

joblib.dump(best_model, "best_traffic_model.pkl")

print("\nBest model saved successfully!")
