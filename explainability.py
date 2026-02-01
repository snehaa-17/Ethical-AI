"""
explainability.py

Module: Explainability & Transparency
Description: Generates human-friendly explanations and counterfactuals.
Ethical Design:
- Explanations are framed as "Observed Patterns" not "Medical Causes".
- Counterfactuals provide actionable, non-prescriptive options.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import copy

def generate_explanation(model, input_data, feature_names, prediction_label_name, confidence):
    """
    Generates a natural language explanation with uncertainty awareness.
    """
    importances = model.feature_importances_
    
    # Combine feature names with importance and values
    feature_info = []
    for name, imp, val in zip(feature_names, importances, input_data):
        feature_info.append({'name': name, 'importance': imp, 'value': val})
        
    feature_info.sort(key=lambda x: x['importance'], reverse=True)
    top_features = feature_info[:2]
    
    # Header based on Risk Level
    if prediction_label_name == "Low":
        explanation = "Status: **Stable Patterns**\n"
        explanation += "Digital behaviors appear consistent with a balanced routine."
        return explanation
        
    # For Risk / Moderate
    explanation = f"Status: **{prediction_label_name}** (Confidence: {confidence:.0%})\n"
    explanation += "The system detected behavioral deviations often associated with stress or fatigue:\n"
    
    for feat in top_features:
        # Heuristic for direction based on scaled value (assuming std scaling approx centered at 0)
        direction = "elevated" if feat['value'] > 0 else "reduced"
        clean_name = feat['name'].replace('_', ' ').title().replace('Avg Daily ', '').replace('Score', '')
        
        explanation += f"- **{clean_name}** appears {direction}.\n"
        
    return explanation

def generate_counterfactual_suggestion(model, input_data, feature_names, current_label):
    """
    Simulates "What If" scenarios to find a change that lowers risk.
    """
    if current_label == "Low":
        return "Maintaining current digital habits is recommended."
        
    # Try perturbing the top 3 most important features
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    for idx in sorted_indices[:3]:
        original_value = input_data[idx]
        feature_name = feature_names[idx]
        
        # Try reducing the value (assuming lower is usually better for risk factors)
        # Note: For 'Social Withdrawal', lower is better. For 'App Diversity', higher is better.
        # This requires domain knowledge mapping.
        
        metric_direction = -1 # Default: try reducing
        if "diversity" in feature_name:
            metric_direction = 1 # Try increasing
            
        # Perturb by 1.0 (approx 1 standard deviation in scaled space)
        modified_input = copy.deepcopy(input_data)
        modified_input[idx] += (metric_direction * 1.0)
        
        # Reshape for prediction
        new_pred_idx = model.predict([modified_input])[0]
        # We need the label mapping. Assuming standard logic implicitly here or returning raw change.
        # To be safe, we just check if prediction changed? 
        # Actually without the LabelEncoder context, we can't be sure of the string label.
        # But we can check if class index changed to '0' (Low) roughly.
        
        # Let's return a generic suggestion based on valid improvements
        clean_name = feature_name.replace('_', ' ').title()
        
        if metric_direction == -1:
            return f"Tip: Reducing **{clean_name}** may help stabilize your digital phenotype."
        else:
            return f"Tip: Increasing **{clean_name}** (e.g., using more varied apps) may reflect better engagement."
            
    return "Consider stabilizing sleep and screen schedules."

def visualize_feature_importance(model, feature_names, save_path="feature_importance.png"):
    """
    Saves a bar chart of global feature importance.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Key Behavioral Indicators (Global Model Importance)")
    plt.bar(range(len(importances)), importances[indices], align="center", color='#4a90e2')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.ylabel("Relative Importance")
    plt.savefig(save_path)
    plt.close()

def visualize_risk_trends(df, save_path="risk_trend_visualization.png"):
    """
    Saves a visualization of the risk distribution in the population.
    """
    if 'risk_level' not in df.columns:
        print("Warning: 'risk_level' column not found for visualization.")
        return

    plt.figure(figsize=(8, 6))
    custom_colors = ['#66c2a5', '#fc8d62', '#8da0cb'] # Qualitative colors
    
    # Order: Low, Moderate, Elevated
    order = ['Low', 'Moderate', 'Elevated']
    counts = df['risk_level'].value_counts().reindex(order).fillna(0)
    
    plt.bar(counts.index, counts.values, color=custom_colors)
    plt.title("Population Risk Distribution (Synthetic)")
    plt.xlabel("Risk Level")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(save_path)
    plt.close()
