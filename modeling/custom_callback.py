from tensorflow import keras

#now accepts time window to identify best model after checkpoint
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, tm_win):
        self.time_window = int(tm_win)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["tm_win"] = self.time_window