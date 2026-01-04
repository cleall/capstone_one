import requests
import time
import os
import numpy as np

from preprocess.preprocess import GTZAN_TEST_ONLY

URL = "http://localhost:4444/predict"
GENRE = "metal"
NAME = "metal_testnpy.npy"

#load spectrogram data
def load_test_spectrogram_data(genre, name):
    spectrogram = os.path.join(GTZAN_TEST_ONLY, genre, name)
    spec_np_arr = None
    
    try:
        spec_np_arr = np.load(spectrogram, allow_pickle=False)
    except Exception as e:
        print(f"Error processing spectrogram: {spectrogram} \n error: {e}")

    return np.array(spec_np_arr, dtype=np.float32)

if __name__ == "__main__":
    spectrogram = load_test_spectrogram_data(GENRE, NAME)
    request = {
        "data": spectrogram.tolist()
    }

    start_time = time.time()
    response = requests.post(URL, json=request)
    result = response.json()
    end_time = time.time()

    duration = end_time - start_time
    print(f"Duration: {duration:.3f} seconds")
    print(f"\nTop genre: {result['top_genre']} ({result['top_probability']:.3f})\n")

    print("Predictions summary")
    for cls, prob in result["predictions"].items():
        print(f"{cls}: {prob:.3f}")