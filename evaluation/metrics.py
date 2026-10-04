from sklearn.metrics import accuracy_score, f1_score, classification_report

def get_performance_report(y_true, y_pred, model_name, feature_name):
    # Calculate basic accuracy (the % of correct guesses)
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate Macro F1: This is important because it treats all classes 
    # equally, even if some labels appear much less often than others.
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"\n--- {model_name} on {feature_name} ---")
    
    # This prints a breakdown of Precision, Recall, and F1 for each class
    # so we can see exactly where the model is getting confused.
    print(classification_report(y_true, y_pred))
    
    # Returning a dictionary makes it easy to convert these results 
    # into a pandas DataFrame later for side-by-side comparison.
    return {
        "Feature Space": feature_name,
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Macro F1": round(f1, 4)
    }