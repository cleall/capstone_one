import requests
import time
import os
import numpy as np

from preprocess.preprocess import GTZAN_TEST_ONLY
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://localhost:4444/predict"
GENRE = "metal"
NAME = "metal_testnpy.npy"
REQUESTS = 400  #increasing takes more time to complete
WORKERS = 4

def load_test_spectrogram_data(genre, name):
    spectrogram = os.path.join(GTZAN_TEST_ONLY, genre, name)
    spec_np_arr = None
    
    try:
        spec_np_arr = np.load(spectrogram, allow_pickle=False)
    except Exception as e:
        print(f"Error processing spectrogram: {spectrogram} \n error: {e}")

    return np.array(spec_np_arr, dtype=np.float32)

spectrogram = load_test_spectrogram_data(GENRE, NAME)
request = {
    "data": spectrogram.tolist()
}

def send_request(_):
    try:
        response = requests.post(URL, json=request, timeout=8)
        return response.status_code
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("====\t Starting load test (╯°□°）╯︵ ┻━┻\t====")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(send_request, r) for r in range(REQUESTS)]
        results = [future.result() for future in as_completed(futures)]
    end_time = time.time()
    duration = end_time - start_time

    success_count = sum(1 for result in results if result == 200)
    error_count = len(results) - success_count
    print(f"Duration: {duration:.3f} seconds")
    print(f"Requests per second: {len(results)/duration:.3f}")
    print(f"Successful requests: {success_count}")
    print(f"Failed requests: {error_count}")
    print("====\t Load test complete ¯\_(ಠ_ಠ)_/¯\t====")