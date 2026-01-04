import numpy as np

from preprocess.preprocess import Preprocess, GTZAN_DS, GENRES, GTZAN_NP_ARR
from keras._tf_keras.keras.utils import to_categorical
from modeling.model import Model

KERAS_MODEL_NAME = "v6_tw_25_35_val_acc_0.706.keras"
ONNX_MODEL_NAME = "v6_onnx.onnx"

def main():
    pre_proc = Preprocess()
    model = Model()
    #create spectrogram from audio samples
    print("Creating and saving mel spectrograms, may take a few minutes")
    pre_proc.create_genre_folders()
    #After first execution comment this line and only load saved data
    pre_proc.save_spectrograms(GTZAN_DS, GENRES, GTZAN_NP_ARR)
    #load spectrogram data
    print("Loading spectrogram data")
    spec_data, genres = pre_proc.load_spectrograms(GENRES, GTZAN_NP_ARR)
    #set homogeneous shape
    hmgns_spec_data = pre_proc.set_homogeneous_shape(spec_data)
    print(f"spec data shape: {hmgns_spec_data.shape}")
    genre = np.array(genres)
    genre = to_categorical(genre, num_classes=len(GENRES))
    print(f"spec data target shape: {genre.shape}")
    #remove duplicates basic eda
    print("EDA, removing duplicates using unique spectrogram indexes")
    unique_indexes = pre_proc.get_spectrogram_unique_indexes(hmgns_spec_data)
    unique_spec_data = [hmgns_spec_data[idx] for idx in sorted(unique_indexes)]
    unique_spec_data = pre_proc.set_homogeneous_shape(unique_spec_data)
    #use mfcc to remove more duplicates
    print("EDA, removing duplicates using spectrogram mfccs")
    mfcc_spec_data = pre_proc.get_mfcc_data_from_spectrogram(unique_spec_data)
    spec_data_for_dtw = pre_proc.mfcc_ready_for_dtw(mfcc_spec_data)
    #compute dtw distances
    dtw_distances = pre_proc.get_dtw_distances(spec_data_for_dtw)
    #remove duplicates obtained from dtw from train and target
    x_load, y_load = pre_proc.remove_dtw_duplicates(dtw_distances, spec_data_for_dtw, unique_spec_data, genre)
    x_load = pre_proc.set_homogeneous_shape(x_load)
    print(f"unique spec data shape: {x_load.shape}")
    print(f"unique spec data target shape: {y_load.shape}")
    #split data 70,20,20 train,validation,test respectively
    print("Splitting data to train, validation and test")
    x_train, y_train, x_vldtn, y_vldtn, x_test, y_test = pre_proc.get_data_splits(x_load, y_load)
    print(len(x_train), len(x_vldtn), len(x_test))
    print(len(y_train), len(y_vldtn), len(y_test))
    #save test data explicitly
    print("Saving test data to use after deployment")
    pre_proc.create_test_folders()
    pre_proc.save_test_data(x_test)
    #normalize train data
    print("Preparing data to train models")
    norm_x_train = model.get_normalized_data(x_train)
    norm_x_vldtn = model.get_normalized_data(x_vldtn)
    #cast to float32 to apply time window during training
    cast_x_train = model.data_to_f32(norm_x_train)
    #train models use augmented data, check models folder for output
    print("Beginning mode training using defined time windows")
    model.train_model(cast_x_train, y_train, norm_x_vldtn, y_vldtn)
    print("Completed training")
    #replace value with the model name you selected
    print("Loading model for conversion")
    #load selected model
    keras_model = model.load_keras_model(KERAS_MODEL_NAME)
    #export to onnx, replace value with desired name to export
    print("Converting loaded model to onnx format")
    model.keras_to_onnx(keras_model, ONNX_MODEL_NAME)

if __name__ == "__main__":
    main()