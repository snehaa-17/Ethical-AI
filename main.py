"""
main.py

Project: Human-Centric AI: Early Mental Health Risk Detection
Description: Main orchestration script.
Author: Ethical AI Researcher (Simulated)
"""

import sys
import numpy as np
import pandas as pd

# Import local modules
from data_simulation import generate_synthetic_data
from preprocessing import preprocess_data
from model import train_models, evaluate_models
from explainability import generate_explanation, visualize_feature_importance, visualize_risk_trends

def perform_ethical_check():
    """Prints mandatory ethical disclaimers found in research protocols."""
    print("\n" + "="*80)
    print(" ETHICAL AI SYSTEM: PRE-RUN CHECK")
    print("="*80)
    print(" [✓] DATA PRIVACY:  No real user data is used. All input is synthetic.")
    print(" [✓] NO DIAGNOSIS:  This tool predicts *risk levels* only, not medical conditions.")
    print(" [✓] HUMAN-IN-LOOP: Output is for clinical support, not automated decision making.")
    print("="*80 + "\n")

def main():
    perform_ethical_check()
    
    # 1. Pipeline: Data Generation
    print("Step 1: generating synthetic digital behavior data...")
    df = generate_synthetic_data(n_samples=2000)
    print(f" -> Generated {len(df)} samples.")
    
    # Save a visualization of the population distribution
    visualize_risk_trends(df)
    
    # 2. Pipeline: Preprocessing
    print("Step 2: Preprocessing and splitting data...")
    X_train, X_test, y_train, y_test, scaler, le, feature_cols = preprocess_data(df)
    
    # 3. Pipeline: Model Training
    print("Step 3: Training machine learning models...")
    models = train_models(X_train, y_train)
    
    # 4. Pipeline: Evaluation
    label_names = le.classes_
    results = evaluate_models(models, X_test, y_test, label_names)
    
    # 5. Explainability & Inference Demo
    print("\n" + "="*80)
    print(" DEMONSTRATION: PREDICTION & EXPLAINABILITY")
    print("="*80)
    
    # Pick a random sample from the test set that is PREDICTED as 'Elevated' or 'Moderate' to show interesting explanation
    # If none found easily, just pick random.
    
    rf_model = models['RandomForest']
    
    # Let's verify feature importance visually
    visualize_feature_importance(rf_model, feature_cols)
    
    # Mock Inference Sample
    sample_idx = np.random.choice(len(X_test))
    sample_input = X_test[sample_idx]
    true_label_idx = y_test[sample_idx]
    
    prediction_idx = rf_model.predict([sample_input])[0]
    prediction_label = le.inverse_transform([prediction_idx])[0]
    true_label = le.inverse_transform([true_label_idx])[0]
    
    print(f"\nTest Sample Index: {sample_idx}")
    print(f"True Risk Level:      {true_label}")
    print(f"Predicted Risk Level: {prediction_label}")
    
    # Generate Explanation
    explanation = generate_explanation(rf_model, sample_input, feature_cols, prediction_label)
    print("\n--- AI Explanation ---")
    print(explanation)
    print("----------------------")
    
    print("\n[SUCCESS] Pipeline Completed. Check folder for 'feature_importance.png' and 'risk_trend_visualization.png'.")

if __name__ == "__main__":
    main()
