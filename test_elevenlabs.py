import sys
import numpy as np
import joblib
import json

def test():
    try:
        from audio_worker import extract_features, load_models
        load_models()
        
        # We need to find the ElevenLabs file in the directory
        import os
        import glob
        
        target = glob.glob("ElevenLabs*")
        if not target:
            # Let's check uploads
            target = glob.glob("uploads/ElevenLabs*")
            if not target:
                print("Could not find the ElevenLabs file.")
                return
                
        file_path = target[0]
        print(f"Testing file: {file_path}")
        
        res = extract_features(file_path)
        print(f"Extraction result:\n{json.dumps(res, indent=2)}")
        
        if 'inst_phase_variance' in res:
            feature_vector = [
                res.get("inst_phase_variance", 0),
                res.get("rt60_estimate", 0),
                res.get("mfcc_variance", 0),
                res.get("spectral_flatness_var", 0),
                res.get("zcr_variance", 0),
                res.get("codec_banding_score", 0),
                res.get("pause_ratio", 0),
                res.get("pitch_drift_over_time", 0)
            ]
            X_np = np.array([feature_vector])
            
            # Load the lightgbm model
            model = joblib.load("models/audio_lgbm_ensemble.pkl")
            prob = model.predict_proba(X_np)[0][1]
            print(f"\nFinal LightGBM Probability (Acoustic Only): {prob*100:.2f}%")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test()
