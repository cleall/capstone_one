import librosa
import os
import numpy as np

from sklearn.model_selection import train_test_split
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

#data constants
GTZAN_DS = "data/gtzan_ds/"
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]
WAV=".wav"
GTZAN_NP_ARR="data/gtzan_np_arr/"
NPY="npy"
GTZAN_TEST_ONLY = "data/gtzan_test_data/"
TEST = "test"

#spectrogram parameters
MS = 1000
OFF_SZ = 3
DUR = 30
SMPL_RT = 16_000
WIN_TM = 128
WIN_SZ = 2048
WIN_OFF = 256
MEL_BANDS = 64
MIN_FRQ = 0
MAX_FRQ = 8_000
PWR = 2
CHANNELS = 3

#dtw threshold
DIST_TRESH = 120

#base model parameters
REG=0.01
METRIC="accuracy"
ACTIVATE="relu"
EPOCHS=50
BATCH=32
LEARN_RT=0.01
INN_SZ=100
DROP=0.0

#random test indices
TEST_IDXS = [5, 12, 3, 10, 17, 4, 1, 23, 0, 13]

class Preprocess():
    #create folders to store spectrogram data
    def create_genre_folders(self):
        for genre in GENRES:
            os.makedirs(os.path.join(GTZAN_NP_ARR, genre), exist_ok=True)

    #create mel spectrogram from audio sample using librosa
    def to_mel_spectrogram(self, source, duration=DUR, sample_rate=SMPL_RT, window_size=WIN_SZ,
            window_offset=WIN_OFF, mel_bands=MEL_BANDS, min_frq=MIN_FRQ, max_frq=MAX_FRQ, power=PWR):
        #load audio sample
        audio_waveform, sampling_rate = librosa.load(source, duration=duration, sr=sample_rate)
        #create spectrogram from audio sample
        spectrogram = librosa.feature.melspectrogram(
            y=audio_waveform, sr=sampling_rate, n_fft=window_size,
            hop_length=window_offset, n_mels=mel_bands, fmin=min_frq, fmax=max_frq, power=power
        )
        #scale amplitude relative to max value in spectrogram
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        #represent as rgb img, shape goes from (Height, Width) to (Height, Width, channels)
        rgb_log_spectrogram = np.stack([log_spectrogram] * CHANNELS, axis=-1)

        return rgb_log_spectrogram

    #create spectrograms and store spectrogram arrays of each genre audio sample
    def save_spectrograms(self, source_folder, classes, arr_folder):
        #create spectrograms
        for idx, genre in enumerate(classes):
            #get each genre path e.g. ../data/gtzan_ds/blues
            genre_folder = os.path.join(source_folder, genre)
            #go through the audio samples 100
            for audio in os.listdir(genre_folder)[:100]:
                if audio.endswith(WAV):
                    audio_sample = os.path.join(genre_folder, audio)
                    try:
                        #save spectrogram arr to respective np arr genre folder
                        arr_name = audio[:-3]
                        arr_name = arr_name+NPY
                        save_arr_to = os.path.join(arr_folder, genre, arr_name)
                        
                        spectrogram_arr = self.to_mel_spectrogram(audio_sample)
                        np.save(save_arr_to, spectrogram_arr)
                    except Exception as e:
                        print(f"Error processing audio sample: {audio} \n error: {e}")
                        continue
    
    #load spectrogram data from arrays
    def load_spectrograms(self, classes, arr_folder):
        spectrograms, genres = [], []
        #load spectrogram array data
        for idx, genre in enumerate(classes):
            genre_folder = os.path.join(arr_folder, genre)
            for arr in os.listdir(genre_folder)[:100]:
                if arr.endswith(NPY):
                    spectrogram = os.path.join(genre_folder, arr)
                    try:
                        spectrogram_arr = np.load(spectrogram)
                        spectrograms.append(spectrogram_arr)
                        genres.append(idx)
                    except Exception as e:
                        print(f"Error processing spectrogram array: {arr} \n error: {e}")
                        continue
        return spectrograms, genres
    
    #set spectrogram homogeneous shape for pre-processing
    def set_homogeneous_shape(self, spectrogram_data):
        max_len = max(spectrogram.shape[1] for spectrogram in spectrogram_data)
        spec_homogeneous = np.zeros((len(spectrogram_data), 64, max_len, 3))

        for idx, spectrogram in enumerate(spectrogram_data):
            spec_homogeneous[idx, :, :spectrogram.shape[1], :] = spectrogram
        
        return spec_homogeneous
    
    #get indexes of unique spectrogram data
    def get_spectrogram_unique_indexes(self, spectrogram_data):
        spec_data_reshaped = np.array([arr.flatten() for arr in spectrogram_data])
        #ignore unique elements use unique index instead
        _, unique_idxs = np.unique(spec_data_reshaped, axis=0, return_index=True)
        return unique_idxs
    
    #obtain mfcc data from spectrogram
    def get_mfcc_data_from_spectrogram(self, spectrogram_data):
        spec_data_mfcc = np.array([librosa.feature.mfcc(S=arr, n_mfcc=13) for arr in spectrogram_data])
        return spec_data_mfcc
    
    #get mfcc data ready for dynamic time warping (DTW)
    def mfcc_ready_for_dtw(self, mfcc_data):
        mfcc_data_ready_dtw = [mfcc_data[mfcc, 0, ...].T for mfcc in range(len(mfcc_data))]
        return mfcc_data_ready_dtw
    
    #compute pairwise dtw distances
    def get_dtw_distances(self, dtw_data):
        n_mfccs = len(dtw_data)
        distances = []

        for mfcc_idx in range(n_mfccs):
            for next_mfcc in range(mfcc_idx+1, n_mfccs):
                distance, _ = fastdtw(dtw_data[mfcc_idx], dtw_data[next_mfcc], dist=euclidean)
                distances.append((mfcc_idx, next_mfcc, distance))
        
        return distances
    
    #remove dtw duplicates
    def remove_dtw_duplicates(self, dtw_distances, dtw_data, spec_data, genre_data):
        duplicate_pairs = [(i, j) for i, j, d in dtw_distances if d < DIST_TRESH]
        
        to_remove = set()
        for _, j in duplicate_pairs:
            to_remove.add(j)

        unique_idxs = [idx for idx in range(len(dtw_data)) if idx not in to_remove]
        spec_data_unique = [spec_data[idx] for idx in sorted(unique_idxs)]
        genre_data_unique = np.array([genre_data[idx] for idx in sorted(unique_idxs)])

        return spec_data_unique, genre_data_unique
    
    #split data
    def get_data_splits(self, spectrogram_data, genre_data):
        #split to 70% train and 30% test
        x_train, x_val_test, y_train, y_val_test = train_test_split(
            spectrogram_data, genre_data, test_size=0.3, stratify=genre_data, random_state=42
        )

        #split the 30% test to 15% train and 15% validation
        x_test, x_vldtn, y_test, y_vldtn = train_test_split(
            x_val_test, y_val_test, test_size=0.5,
            stratify=y_val_test, random_state=42
        )

        return x_train, y_train, x_vldtn, y_vldtn, x_test, y_test
    
    #create test data folders
    def create_test_folders(self):
        for genre in GENRES:
            os.makedirs(os.path.join(GTZAN_TEST_ONLY, genre), exist_ok=True)

    #explicitly save test data to call model after deployment
    def save_test_data(self, test_data):
        test_spectrograms = test_data[TEST_IDXS]
        for idx in range(len(GENRES)):
            arr_name = GENRES[idx]+"_"+TEST
            arr_name = arr_name+NPY
            save_arr_to = os.path.join(os.path.join(GTZAN_TEST_ONLY, GENRES[idx]), arr_name)
            np.save(save_arr_to, test_spectrograms[idx])
        print(f"Saved test data to {GTZAN_TEST_ONLY} folder")