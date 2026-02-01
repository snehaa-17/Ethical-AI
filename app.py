"""
app.py

Project: Human-Centric AI Research Prototype
Description: Flask backend for the Privacy-Preserving Digital Phenotype system.
Modes:
- Auto-Simulation: Generates synthetic streams of passive data.
- Manual Override: Allows research-grade testing of specific inputs.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import random

# Local Modules
from data_simulation import generate_synthetic_data, generate_digital_phenotype_stream
from preprocessing import preprocess_data
from model import train_models, calibrate_confidence
from risk_tracker import RiskTracker
from explainability import generate_explanation, generate_counterfactual_suggestion

app = Flask(__name__)

# --- System State (Simulation Memory) ---
# In a real deployed system, this would be encrypted on-device storage.
STATE = {
    'models': None,
    'scaler': None,
    'label_encoder': None,
    'feature_cols': None,
    'risk_tracker': RiskTracker(history_size=10),
    'current_day': 0,
    'simulation_stream': None # Will hold dataframe of pre-generated days
}

def initialize_system():
    """Trains models and prepares the simulation engine."""
    print(" * [System] Initializing Human-Centric AI Engine...")
    
    # 1. Train on synthetic population data
    df = generate_synthetic_data(n_samples=2000)
    X_train, X_test, y_train, y_test, scaler, le, feats = preprocess_data(df)
    models = train_models(X_train, y_train)
    
    STATE['models'] = models
    STATE['scaler'] = scaler
    STATE['label_encoder'] = le
    STATE['feature_cols'] = feats
    
    # 2. Pre-generate a simulation stream for "Auto Mode"
    # Scenario: User starts stable, then drifts into risk, then recovers?
    # Let's do a simple Increasing Risk scenario for the demo.
    STATE['simulation_stream'] = generate_digital_phenotype_stream(n_days=30, risk_scenario="increasing_risk")
    
    print(" * [System] Ready. Privacy constraints active.")

initialize_system()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint. 
    Handles both Auto-Simulated data (fetched by ID) and Manual Overrides.
    """
    try:
        data = request.json
        mode = data.get('mode', 'manual') # 'manual' or 'auto'
        
        # --- 1. Data Source Selection ---
        if mode == 'auto':
            # Fetch next day from stream
            day_idx = STATE['current_day'] % len(STATE['simulation_stream'])
            day_data = STATE['simulation_stream'].iloc[day_idx]
            
            # Extract features (exclude day_id)
            input_features = [
                float(day_data['avg_daily_screen_time']),
                float(day_data['night_usage_ratio']),
                float(day_data['app_usage_diversity']),
                float(day_data['typing_speed_variance']),
                float(day_data['sleep_irregularity_score']),
                float(day_data['social_app_withdrawal_score'])
            ]
            STATE['current_day'] += 1 # Advance time
            
        else:
            # Manual Override Input
            input_features = [
                float(data.get('avg_daily_screen_time')),
                float(data.get('night_usage_ratio')),
                float(data.get('app_usage_diversity')),
                float(data.get('typing_speed_variance')),
                float(data.get('sleep_irregularity_score')),
                float(data.get('social_app_withdrawal_score'))
            ]
            
        # --- 2. Preprocessing ---
        scaler = STATE['scaler']
        # Reshape for single sample
        input_scaled = scaler.transform([input_features])
        
        # --- 3. Inference & Uncertainty ---
        model = STATE['models']['RandomForest']
        
        # Get probabilities
        probs = model.predict_proba(input_scaled)[0]
        
        # Get raw confidence and label
        raw_conf, label_idx = calibrate_confidence(probs)
        label_str = STATE['label_encoder'].inverse_transform([label_idx])[0]
        
        # Apply "Simulated Penalty" to confidence if Manual Mode
        # (Because manual sliders essentially fake the data structure)
        final_confidence = raw_conf
        if mode == 'manual':
            final_confidence = max(0.4, raw_conf - 0.15) 
            
        # --- 4. Risk Tracking ---
        tracker = STATE['risk_tracker']
        # Only track history in Auto mode (to show longitudinal trends)
        # Manual mode is for "What-If" testing, shouldn't pollute history?
        # Actually, let's track everything but flag it. 
        # For this prototype, we'll only track Auto updates to show the clean trend line.
        
        trend = "N/A (Manual)"
        if mode == 'auto':
            tracker.add_prediction(label_str, final_confidence, probs)
            trend = tracker.get_trend()
            
        # --- 5. Explainability ---
        explanation = generate_explanation(model, input_scaled[0], STATE['feature_cols'], label_str, final_confidence)
        
        counterfactual = generate_counterfactual_suggestion(model, input_features, STATE['feature_cols'], label_str)
        
        # Prepare Feature Data for UI
        importances = model.feature_importances_.tolist()
        feature_data = []
        for name, imp in zip(STATE['feature_cols'], importances):
            feature_data.append({'name': name, 'importance': imp})
            
        return jsonify({
            'status': 'success',
            'mode': mode,
            'day_index': STATE['current_day'] if mode == 'auto' else -1,
            'input_echo': input_features,
            'risk_level': label_str,
            'confidence': final_confidence,
            'trend': trend,
            'explanation': explanation,
            'counterfactual': counterfactual,
            'feature_data': feature_data
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    STATE['risk_tracker'].reset()
    STATE['current_day'] = 0
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
