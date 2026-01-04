import numpy as np
from fastapi import FastAPI
import onnxruntime as ort
import uvicorn
import os

from modeling.model import Model
from preprocess.preprocess import GENRES
from train_pipeline.train import ONNX_MODEL_NAME
from response.predict_response import PredictResponse
from response.spectrogram import Spectrogram

MODELS = "models/"
PORT = 4444
music_classifier = FastAPI(title="music_classifier")

def pre_process_test_data(modeling: Model, test_data):
    norm_test_data = modeling.get_normalized_data(test_data)
    cast_test_data = np.array(norm_test_data, dtype=np.float32)
    batch_test_data = np.expand_dims(cast_test_data, axis=0)
    
    return batch_test_data

def predict_single(spectrogram_data):
    model = Model()
    onnx_session = ort.InferenceSession(
        "models/"+ONNX_MODEL_NAME,
        providers=["CPUExecutionProvider"]
    )

    data = pre_process_test_data(model, spectrogram_data)
    inputs = onnx_session.get_inputs()
    outputs = onnx_session.get_outputs()
    input_name = inputs[0].name
    output_name = outputs[0].name
    outputs = onnx_session.run([output_name], {input_name: data})

    predictions = outputs[0][0].tolist()
    results = dict(zip(GENRES, predictions))
    top_genre = max(results, key=results.get)
    top_probability = results.get(top_genre)

    return results, top_genre, top_probability

@music_classifier.get("/")
def root():
    return {"message": "Music Classification Service"}

@music_classifier.get("/health")
def health():
    return {"status": "healthy"}

@music_classifier.post("/predict", response_model=PredictResponse)
def predict(spectrogram: Spectrogram):
    spectrogram_data = np.array(spectrogram.data, dtype=np.float32)
    predictions, top_genre, top_prob = predict_single(spectrogram_data)
    
    return PredictResponse(
        predictions = predictions,
        top_genre = top_genre,
        top_probability = top_prob
    )

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 4444))
    uvicorn.run("music_classifier:music_classifier", host="0.0.0.0", port=PORT)
