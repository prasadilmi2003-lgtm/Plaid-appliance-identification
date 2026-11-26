"""Predict the appliance type from one raw waveform CSV (column 0 = current, column 1 = voltage, no header)."""
import joblib
import pandas as pd
from plaid_features import load_wave, extract_features

bundle = joblib.load("plaid_model.joblib")
model, FEATURES = bundle["pipeline"], bundle["features"]


def predict_csv(path):
    i, v = load_wave(path)                         # 1. raw CSV -> current and voltage arrays
    feats = extract_features(i, v)                 # 2. arrays -> all 30 features + 3 flags (same code as 02)
    X = pd.DataFrame([feats])[FEATURES]            # 3. one-row table, only the chosen features, same order
    probs = model.predict_proba(X)[0]              # 4. impute -> random forest
    best = probs.argmax()
    warnings = [name for name in ["stable_found", "window_ok"] if not feats[name]]
    return {"type": model.classes_[best], "confidence": round(float(probs[best]), 2), "warnings": warnings}


if __name__ == "__main__":
    import sys
    print(predict_csv(sys.argv[1]))
