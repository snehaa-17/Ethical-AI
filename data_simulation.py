"""
data_simulation.py

Module: Privacy-Preserving Data Simulation
Description: Generates SYNTHETIC passive digital behavior signals. 
Ethical Constraint: No real user data is ever collected or stored. This module strictly simulates
what a passive sensing system *would* see (e.g., timestamps, acceleration variance) converted into
privacy-preserving aggregate features.

Design:
- "Phenotype Stream": Simulates longitudinal data (days of behavior) to show trends.
- "Personas": Generates data drift to simulate changing mental states (Stable -> At Risk).
"""

import pandas as pd
import numpy as np
import random

def generate_digital_phenotype_stream(n_days=30, risk_scenario="stable"):
    """
    Generates a longitudinal stream of daily behavior for a single simulated user.
    
    Args:
        n_days (int): Number of days to simulate.
        risk_scenario (str): 'stable', 'increasing_risk', 'improving'.
        
    Returns:
        pd.DataFrame: DataFrame where each row is a day.
    """
    days = []
    
    # Base baseline for this user
    baseline_screen = np.random.normal(5.0, 1.0)
    baseline_night = 0.1
    baseline_sleep_var = 0.2
    
    current_risk_factor = 0.0 # 0 to 1
    
    for day in range(n_days):
        # Drift logic based on scenario
        if risk_scenario == "increasing_risk":
            current_risk_factor += (0.8 / n_days) # Slowly drift to high risk
        elif risk_scenario == "improving":
            current_risk_factor = max(0, 0.8 - (day * (0.8 / n_days)))
            
        current_risk_factor = min(current_risk_factor, 1.0)
        
        # Apply risk factor to behaviors (Simulating depression/anxiety correlates)
        # Risk -> Higher night usage, more erratic sleep, social withdrawal
        
        # 1. Screen Time: Often increases with withdrawal, or decreases significantly. Let's say increases.
        daily_screen = np.random.normal(baseline_screen + (2.0 * current_risk_factor), 1.0)
        daily_screen = np.clip(daily_screen, 0.5, 17.0)
        
        # 2. Night Usage: Strongly correlated with risk
        daily_night = np.random.beta(2 + (5 * current_risk_factor), 5) 
        # roughly: low risk -> beta(2,5) ~0.28, high risk -> beta(7,5) ~0.58
        
        # 3. Sleep Irregularity: Increases with risk
        daily_sleep_var = np.clip(np.random.normal(baseline_sleep_var + (0.5 * current_risk_factor), 0.1), 0, 1)
        
        # 4. Typing Speed Variance: Higher stress -> higher variance
        daily_typing_var = np.random.gamma(shape=2.0, scale=30.0 + (30.0 * current_risk_factor))
        
        # 5. Social Withdrawal: Increases with risk
        daily_withdrawal = np.clip(np.random.normal(0.2 + (0.6 * current_risk_factor), 0.15), 0, 1)
        
        # 6. App Diversity: Drops with risk (relying on fewer apps, doomscrolling)
        daily_diversity = int(np.random.poisson(12 - (5 * current_risk_factor)))
        daily_diversity = max(1, daily_diversity)
        
        days.append({
             'day_id': day,
             'avg_daily_screen_time': daily_screen,
             'night_usage_ratio': daily_night,
             'app_usage_diversity': daily_diversity,
             'typing_speed_variance': daily_typing_var,
             'sleep_irregularity_score': daily_sleep_var,
             'social_app_withdrawal_score': daily_withdrawal
        })
        
    return pd.DataFrame(days)

def generate_synthetic_data(n_samples=2000, random_seed=42):
    """
    Generates a static synthetic dataset for MODEL TRAINING.
    Includes labels based on a hidden ground-truth derived from literature.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # 1. Avg Daily Screen Time (hours)
    avg_daily_screen_time = np.random.normal(loc=6.0, scale=2.5, size=n_samples)
    avg_daily_screen_time = np.clip(avg_daily_screen_time, 0.5, 18.0)
    
    # 2. Night Usage Ratio (0.0 to 1.0)
    night_usage_ratio = np.random.beta(a=2, b=5, size=n_samples)
    
    # 3. App Usage Diversity (count)
    app_usage_diversity = np.random.poisson(lam=12, size=n_samples)
    
    # 4. Typing Speed Variance (ms)
    typing_speed_variance = np.random.gamma(shape=2.0, scale=30.0, size=n_samples)
    
    # 5. Sleep Irregularity Score (0.0=Consistent, 1.0=Erratic)
    sleep_irregularity_score = np.random.uniform(0.0, 1.0, size=n_samples)
    
    # 6. Social App Withdrawal Score (0.0=Active, 1.0=Withdrawn)
    social_app_withdrawal_score = np.random.beta(a=1.5, b=1.5, size=n_samples)

    # --- Hidden Risk Logic (Ground Truth Generation) ---
    # Based on "Digital Phenotyping" literature correlations.
    
    raw_risk_score = (
        0.35 * (night_usage_ratio) + 
        0.30 * (sleep_irregularity_score) + 
        0.25 * (social_app_withdrawal_score) + 
        0.15 * (typing_speed_variance / 200.0) - 
        0.10 * (app_usage_diversity / 30.0)
    )
    
    # Add noise for realism
    noise = np.random.normal(0, 0.08, n_samples)
    final_risk_score = raw_risk_score + noise
    
    threshold_low = np.percentile(final_risk_score, 50)
    threshold_mod = np.percentile(final_risk_score, 80)
    
    risk_labels = []
    for score in final_risk_score:
        if score < threshold_low:
            risk_labels.append("Low")
        elif score < threshold_mod:
            risk_labels.append("Moderate")
        else:
            risk_labels.append("Elevated")
            
    df = pd.DataFrame({
        'avg_daily_screen_time': avg_daily_screen_time,
        'night_usage_ratio': night_usage_ratio,
        'app_usage_diversity': app_usage_diversity,
        'typing_speed_variance': typing_speed_variance,
        'sleep_irregularity_score': sleep_irregularity_score,
        'social_app_withdrawal_score': social_app_withdrawal_score,
        'risk_level': risk_labels
    })
    
    return df

if __name__ == "__main__":
    print("Testing Stream Generation...")
    stream = generate_digital_phenotype_stream(n_days=5, risk_scenario="increasing_risk")
    print(stream)
    print("\nTesting Training Data Generation...")
    df = generate_synthetic_data(10)
    print(df.head())
