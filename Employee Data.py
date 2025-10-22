import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# --- 1. CSV File Loading ---
FILENAME = "employee_data.csv"

try:
    df = pd.read_csv(FILENAME)
    print(f"--- Data Loaded: {FILENAME} ---")
    print(f"Total Employees (Samples): {len(df)}")
    pd.set_option('display.max_columns', None)
    print("Columns (First 10 rows):")
    print(df.head(10))
    print("\n" + "=" * 60 + "\n")
except FileNotFoundError:
    print(f"Error: File '{FILENAME}' Not Found.")
    exit()

# --- 2. Getting data ready(Features aur Target) ---

# Education Encode
education_mapping = {
    'High School': 0,
    'Bachelor': 1,
    'Master': 2,
    'PhD': 3
}
df['Education_Encoded'] = df['Education Level'].map(education_mapping)

# Performance Encode
performance_mapping = {
    'Poor': 0, 'Low': 0,
    'Average': 1, 'Good': 1,
    'Excellent': 2, 'Outstanding': 2
}
df['Performance_Encoded'] = df['Performance Category'].map(performance_mapping)

# Missing Values Remove
df.dropna(subset=['Education_Encoded', 'Performance_Encoded'], inplace=True)

# --- 3. Feature aur Target ---
features = [
    'Age',
    'Experience (Years)',
    'Education_Encoded',
    'Hours Worked per week',
    'Projects Completed'
]
X = df[features]
y = df['Performance_Encoded'].astype(int)

# --- 4. Feature Scaling (necessory for model accuracy) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 5. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print("\n" + "=" * 60 + "\n")

# --- 6. Random Forest Hyperparameter Optimization ---
print("🔍 Optimizing Random Forest Parameters...")

param_grid = {
    'n_estimators': [150],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=0
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"✅ Best Parameters: {grid.best_params_}")
print("\n" + "=" * 60 + "\n")

# --- 7. Model Evaluation ---
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Model Accuracy (Optimized): {accuracy:.4f}")
print("\n--- Classification Report ---")
target_names = ['Low', 'Average', 'High']
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
print("\n" + "=" * 60 + "\n")

# --- 8. Prediction of new Employee ---
print("--- 🔮 New Employee Performance Prediction ---")

new_employee_data = pd.DataFrame({
    'Age': [30],
    'Experience (Years)': [8],
    'Education_Encoded': [2],  # Master's
    'Hours Worked per week': [40],
    'Projects Completed': [10]
})

# Scale New Data
new_employee_scaled = scaler.transform(new_employee_data)

prediction = best_model.predict(new_employee_scaled)
category_map = {0: 'Low', 1: 'Average', 2: 'High'}
predicted_category = category_map[prediction[0]]

print(f"Predicted Category: {predicted_category} ({prediction[0]})")
print("\n✅ Prediction Complete!")
