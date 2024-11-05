import pandas as pd

def save_model(model, filepath):
    import joblib
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    import joblib
    return joblib.load(filepath)

def generate_report(accuracy, precision, recall, f1):
    print("Model Evaluation Report")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

