import pickle
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

def main():
    print("Loading cached audio features...")
    with open("models/audio_features_cache.pkl", "rb") as f:
        ckpt = pickle.load(f)
    
    X = np.array(ckpt["X"])
    y = np.array(ckpt["y_labels"])
    
    # We will use ALL 10 features so the model can rely on Spectral Flatness, ZCR, etc.
    # which are highly effective at catching ElevenLabs.
    
    valid_mask = np.all(np.isfinite(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Training on {X.shape[0]} samples with ALL {X.shape[1]} features...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    base_clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    base_clf.fit(X_train, y_train)
    
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='isotonic', cv=5)
    calibrated_clf.fit(X_train, y_train)
    
    y_pred = base_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy (Full Features): {acc * 100:.2f}%")
    
    for out_path in [Path("models/audio_lgbm_ensemble.pkl"), Path("models/trained/audio_lgbm_ensemble.pkl")]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated_clf, out_path)
    
    print("Saved restored 10-feature ensemble model!")

if __name__ == "__main__":
    main()
