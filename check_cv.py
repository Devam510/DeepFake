import pickle
try:
    with open('models/trained/video_meta_voter.pkl', 'rb') as f:
        data = pickle.load(f)
        accuracy = data.get('cv_accuracy', 0)
        print(f"New CV Accuracy: {accuracy * 100:.2f}%")
except Exception as e:
    print(f"Error: {e}")
