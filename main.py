###############################################
# EMPLOYEE WELLNESS & CHURN PREDICTION SYSTEM
# Synthetic Data + Multi-Model ML Pipeline
#############################################
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def generate_data(n=20000):
    """Generates a synthetic dataset for employee attrition prediction."""
    print("===== STEP 1: Synthetic Dataset Generation =====")
    np.random.seed(42)
    random.seed(42)

    data = pd.DataFrame({
        "Age": np.random.randint(21, 60, n),
        "Experience": np.random.randint(0, 20, n),
        "Annual_Salary": np.random.randint(250000, 1800000, n),
        "Last_Increment_Pct": np.round(np.random.uniform(0, 20, n), 2),
        "Bonus_Received": np.random.randint(0, 100000, n),
        "Medical_Score": np.random.randint(1, 10, n),
        "Sick_Leaves": np.random.randint(0, 20, n),
        "Skill_Level": np.random.randint(1, 10, n),
        "Skill_Match_With_Project": np.random.randint(1, 10, n),
        "Overtime_Hours": np.random.randint(0, 50, n),
        "Job_Satisfaction": np.random.randint(1, 5, n),
        "Work_Stress": np.random.randint(1, 10, n),
        "Performance_Rating": np.round(np.random.uniform(1, 5, n), 1)
    })

    # Attrition Logic: Higher stress, low satisfaction, low increment -> more chances to leave
    leave_prob = (
        (data["Work_Stress"] * 0.12) +
        ((5 - data["Job_Satisfaction"]) * 0.18) +
        ((5 - data["Skill_Match_With_Project"]) * 0.10) +
        ((10 - data["Medical_Score"]) * 0.05) +
        ((5 - data["Performance_Rating"]) * 0.15)
    )

    data["Attrition"] = (leave_prob > leave_prob.mean()).astype(int)
    print(f"Synthetic dataset with {n} employees created successfully!\n")
    return data

def evaluate(model, model_name, X_test, y_test):
    """Evaluates the performance of a given model."""
    pred = model.predict(X_test)
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred):.4f}")
    print(f"Recall   : {recall_score(y_test, pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, pred))
    return pred

def main():
    # 1. Data Generation
    data = generate_data()

    # 2. Train-Test Split & Scaling
    print("===== STEP 2: Train-Test Split & Scaling =====")
    X = data.drop("Attrition", axis=1)
    y = data["Attrition"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print("Train-test split and scaling complete.\n")

    # 3. Model Training
    print("===== STEP 3: Model Training =====")
    
    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    # K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    print("All models (Random Forest, Logistic Regression, KNN) trained successfully!\n")

    # 4. Model Evaluation
    print("===== STEP 4: Model Evaluation =====")
    results = []
    
    # Evaluate Random Forest
    rf_preds = evaluate(rf, "Random Forest", X_test, y_test)
    results.append({
        "Model": "Random Forest Classifier",
        "Accuracy": accuracy_score(y_test, rf_preds),
        "Precision": precision_score(y_test, rf_preds),
        "Recall": recall_score(y_test, rf_preds),
        "F1-Score": f1_score(y_test, rf_preds)
    })
    
    # Evaluate Logistic Regression
    lr_preds = evaluate(lr, "Logistic Regression", X_test, y_test)
    results.append({
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, lr_preds),
        "Precision": precision_score(y_test, lr_preds),
        "Recall": recall_score(y_test, lr_preds),
        "F1-Score": f1_score(y_test, lr_preds)
    })
    
    # Evaluate KNN
    knn_preds = evaluate(knn, "KNN", X_test, y_test)
    results.append({
        "Model": "KNN",
        "Accuracy": accuracy_score(y_test, knn_preds),
        "Precision": precision_score(y_test, knn_preds),
        "Recall": recall_score(y_test, knn_preds),
        "F1-Score": f1_score(y_test, knn_preds)
    })

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)

    # Identify best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_accuracy = comparison_df.iloc[0]['Accuracy']

    print(f"\nBEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

    # 5. Prediction for New Employee
    print("\n===== STEP 5: Prediction for New Employee =====")
    
    # Generate random values for a new employee
    new_employee_dict = {
        "Age": random.randint(21, 60),
        "Experience": random.randint(0, 20),
        "Annual_Salary": random.randint(250000, 1800000),
        "Last_Increment_Pct": round(random.uniform(0, 20), 2),
        "Bonus_Received": random.randint(0, 100000),
        "Medical_Score": random.randint(1, 10),
        "Sick_Leaves": random.randint(0, 20),
        "Skill_Level": random.randint(1, 10),
        "Skill_Match_With_Project": random.randint(1, 10),
        "Overtime_Hours": random.randint(0, 50),
        "Job_Satisfaction": random.randint(1, 5),
        "Work_Stress": random.randint(1, 10),
        "Performance_Rating": round(random.uniform(1, 5), 1)
    }
    
    new_employee_df = pd.DataFrame([new_employee_dict])
    scaled_new = scaler.transform(new_employee_df)

    rf_prob = rf.predict_proba(scaled_new)[0][1] * 100
    prediction = rf.predict(scaled_new)[0]

    print("\n--- New Employee Prediction (Random Forest) ---")
    print(f"Data      : {new_employee_dict}")
    print(f"Prediction: {'Will Leave' if prediction == 1 else 'Will Stay'}")
    print(f"Leave Probability: {rf_prob:.2f}%")
    print(f"Stay Probability : {100 - rf_prob:.2f}%")

    # 6. Display Sample Results
    print("\n--- Sample Test Results (Actual vs Predicted) ---")
    full_results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": rf_preds
    })
    print(full_results.head(20))

    # Identify best model for future use if needed
    if best_model_name == "Random Forest Classifier":
        best_model = rf
    elif best_model_name == "Logistic Regression":
        best_model = lr
    else:
        best_model = knn
    
    print(f"Best model selected: {best_model_name}")
    print("\nProject Completed Successfully!")

if __name__ == "__main__":
    main()
