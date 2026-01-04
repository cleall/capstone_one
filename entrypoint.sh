#!/bin/sh
exec uvicorn music_classifier:music_classifier --host 0.0.0.0 --port "${PORT}"