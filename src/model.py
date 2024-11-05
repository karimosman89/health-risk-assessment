import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import preprocess_data, load_and_clean_data

# Load and preprocess data
data_path = 'data/diabetes.csv'  
df = load_and_clean_data(data_path)
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

# Set up the Gradient Boosting model with hyperparameter tuning
model = GradientBoostingClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)
search.fit(X_train, y_train)
best_model = search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")

# Save model and preprocessor
joblib.dump(best_model, 'models/health_risk_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
print("Model and preprocessor saved to 'models/' directory")
