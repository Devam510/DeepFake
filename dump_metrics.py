from audio_worker import extract_features, load_models
import glob

load_models()
files = glob.glob('ElevenLabs*.mp3')
if files:
    res = extract_features(files[0])
    print('\n--- ELEVENLABS METRICS ---')
    print(f'MFCC Var: {res.get("mfcc_variance")}')
    print(f'Flatness Var: {res.get("spectral_flatness_var")}')
    print(f'ZCR Var: {res.get("zcr_variance")}')
    print('--------------------------')
else:
    print('No ElevenLabs file found.')
