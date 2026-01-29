import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
# 1. Load the dataset
df = pd.read_csv('housing.csv') 
# 2. Preprocessing
# Fill missing values in 'total_bedrooms' with the median 
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
# --- 3. Random Forest Regression ---
# Goal: Predict 'median_house_value'
print("--- Running Random Forest Regression ---")
# One-hot encode 'ocean_proximity' for regression features
df_reg = pd.get_dummies(df, columns=['ocean_proximity'])
X_reg = df_reg.drop('median_house_value', axis=1)
y_reg = df_reg['median_house_value']
# Split data (80% train, 20% test)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
# Initialize and train Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_reg, y_train_reg)
# Predict and Evaluate
y_pred_reg = regressor.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.4f}\n")
# --- 4. Random Forest Classification ---
# Goal: Predict 'ocean_proximity'
print("--- Running Random Forest Classification ---")
X_clf = df.drop('ocean_proximity', axis=1)
y_clf = df['ocean_proximity']
# Encode the target labels (e.g., 'NEAR BAY' -> 3)
le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)
# Split data (80% train, 20% test)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf_encoded, test_size=0.2, random_state=42
)
# Initialize and train Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_clf, y_train_clf)
# Predict and Evaluate
y_pred_clf = classifier.predict(X_test_clf)
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test_clf, y_pred_clf, target_names=le.classes_))

