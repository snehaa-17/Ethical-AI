"""
preprocessing.py

Module: Data Preprocessing
Description: Cleans, scales, and splits the data for model training.
Ethical Considerations:
- Fairness: Ensure preprocessing doesn't introduce bias (though harder to control in synthetic stats).
- Privacy: In a real deployment, Differential Privacy techniques (e.g., adding Laplacian noise) 
  would be applied HERE before any data leaves the user's device.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    """
    Standardizes features and encodes labels.
    
    Args:
        df (pd.DataFrame): Raw dataframe with features and target.
        
    Returns:
        X_train, X_test, y_train, y_test (arrays): Split data ready for training.
        scaler (StandardScaler): Fitted scaler object.
        le (LabelEncoder): Fitted label encoder object.
        feature_names (list): List of feature column names.
    """
    
    # Separating features and target
    target_col = 'risk_level'
    feature_cols = [c for c in df.columns if c != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # --- Data Splitting ---
    # 80% Train, 20% Test
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- Feature Scaling ---
    # Standardization is important for Logistic Regression and general model stability.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # --- Label Encoding ---
    # Map strings to integers: Low -> 0, Moderate -> 1, Elevated -> 2 (Mapping may vary, we track classes)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    
    # Print mapping for clarity
    print(f"Label Encoding Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return X_train, X_test, y_train, y_test, scaler, le, feature_cols

if __name__ == "__main__":
    # Quick test
    from data_simulation import generate_synthetic_data
    df = generate_synthetic_data(100)
    X_tr, X_te, y_tr, y_te, _, _, _ = preprocess_data(df)
    print("Debug: Data Shape", X_tr.shape)
