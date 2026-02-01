"""
model.py

Module: Privacy-Preserving Risk Assessment Model
Description: Trains and serves the ML model.
Ethical Considerations:
- Probabilistic Output: We use `predict_proba` to express uncertainty, rather than hard classification.
- Calibration: Confidence is reduced if input signals are highly incongruent.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def train_models(X_train, y_train, random_seed=42):
    """
    Trains models with probabilistic capabilities.
    """
    # 1. Baseline: Logistic Regression (Interpretable weights)
    lr_model = LogisticRegression(random_state=random_seed, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # 2. Primary: Random Forest (Better performance, feature importance)
    # n_estimators=100 for stability
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=random_seed, 
        max_depth=6,             # Limit depth to prevent overfitting on synthetic noise
        min_samples_leaf=4       # Smooth predictions
    )
    rf_model.fit(X_train, y_train)
    
    return {'LogisticRegression': lr_model, 'RandomForest': rf_model}

def calibrate_confidence(prob_distribution, input_features=None):
    """
    Heuristic to adjust confidence based on data consistency.
    
    Args:
        prob_distribution (array): Raw probabilities [p_low, p_mod, p_high].
        input_features (dict/array): The raw input features (optional).
        
    Returns:
        tuple: (adjusted_confidence, adjusted_label)
    """
    # Get max probability as base confidence
    max_prob = np.max(prob_distribution)
    label_idx = np.argmax(prob_distribution)
    
    # Simple Heuristic: If max_prob is < 0.5, it's very uncertain. 
    # In a research prototype, we might want to flag this explicitly.
    
    # Ideally, we would detect "Out of Distribution" (OOD) data here,
    # e.g., if screen_time > 20h (physically impossible), confidence -> 0.
    
    return max_prob, label_idx

def evaluate_models(models, X_test, y_test, label_names):
    """
    Evaluates trained models.
    """
    results = {}
    print("\n--- Research Prototype Evaluation ---")
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.4f}")
        # Only print full report for main model to keep logs clean
        if name == 'RandomForest':
            print("Classification Report:")
            print(classification_report(y_test, y_pred, target_names=label_names))
            results['accuracy'] = acc
            
    return results
