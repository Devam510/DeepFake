import joblib
import numpy as np
import traceback

try:
    print("Loading model...")
    clf = joblib.load('models/trained/audio_lgbm_ensemble.pkl')
    print("Model loaded. Predicting...")
    X = np.zeros((1, 10))
    res = clf.predict_proba(X)
    print(f"Success! Result: {res}")
except Exception as e:
    print("FAILED!")
    traceback.print_exc()
