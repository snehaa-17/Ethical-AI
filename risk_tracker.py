"""
risk_tracker.py

Module: Longitudinal Risk Tracking
Description: Maintains a short-term history of risk predictions to identify trends.
Privacy: This is a session-based tracker. In a real system, this would be encrypted on-device.
"""

from collections import deque
import numpy as np

class RiskTracker:
    def __init__(self, history_size=7):
        # Rolling window of the last N days
        self.history = deque(maxlen=history_size)
    
    def add_prediction(self, risk_label, confidence, proba_distribution):
        """
        Adds a new daily prediction to the history.
        """
        self.history.append({
            'label': risk_label,
            'confidence': confidence,
            'probs': proba_distribution # Dictionary or array
        })
        
    def get_trend(self):
        """
        Determines the trend based on the last few entries.
        Returns: 'Stable', 'Increasing Risk', 'Decreasing Risk', 'Insufficient Data'
        """
        if len(self.history) < 2:
            return "Insufficient Data"
            
        # Map labels to numeric severity
        severity_map = {'Low': 0, 'Moderate': 1, 'Elevated': 2}
        
        # Get last 3 predictions (or fewer)
        recent = list(self.history)[-3:]
        scores = [severity_map.get(item['label'], 0) for item in recent]
        
        # Simple slope check
        if all(x == scores[0] for x in scores):
            return "Stable"
            
        # Check for strict increase
        if scores[-1] > scores[0]:
            return "Increasing Trend"
            
        if scores[-1] < scores[0]:
            return "Improving Trend"
            
        return "Fluctuating"

    def reset(self):
        self.history.clear()
