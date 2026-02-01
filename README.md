# Human-Centric AI: Early Mental Health Risk Detection

**An Ethical AI Research Prototype**

## 🎯 Project Goal
To design a non-intrusive AI system that predicts *early* mental health risk levels (Low, Moderate, Elevated) using **passive digital behavior patterns** (screen time, sleep irregularity, typing variance) without accessing raw content or diagnoses.

## 🛡️ Ethical AI Principles implemented
1.  **Privacy First**: The project uses synthetic data. In deployment, it supports local processing and differential privacy.
2.  **No Diagnosis**: The system outputs "Risk Levels" for triage, explicitly avoiding medical labels like "Depression" or "Anxiety".
3.  **Transparency**: Every prediction follows the "Right to Explanation" principle, providing human-readable reasons (e.g., "Elevated risk due to sleep irregularity").
4.  **Bias Mitigation**: Code structure allows for fairness audits (though data here is synthetic).

## 📂 Project Structure
- `data_generation.py`: Creates synthetic user behavior data.
- `preprocessing.py`: Handles data scaling and encoding.
- `model_training.py`: Trains Logistic Regression (baseline) and Random Forest (main).
- `explainability.py`: Generates text explanations and importance plots.
- `main.py`: Runs the full pipeline.

## 🚀 How to Run
1.  Ensure you have Python installed.
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```
3.  Run the main script:
    ```bash
    python main.py
    ```

## 📊 Output
- The script prints model accuracy and classification reports.
- It demonstrates a prediction case with a clear explanation.
- Generates `risk_trend_visualization.png` and `feature_importance.png`.

---
*Disclaimer: This is a technical demonstration for educational purposes. It is not a medical device.*
