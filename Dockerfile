FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./
RUN uv sync --locked

COPY "music_classifier.py" ./
COPY "models/v6_onnx.onnx" ./models/
COPY "modeling/model.py" "modeling/custom_callback.py" ./modeling/
COPY "train_pipeline/train.py" ./train_pipeline/
COPY "preprocess/preprocess.py" ./preprocess/
COPY "response/spectrogram.py" "response/predict_response.py" ./response/

COPY "entrypoint.sh" /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 4444

ENTRYPOINT ["/entrypoint.sh"]