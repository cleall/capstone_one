import tensorflow as tf
import tf2onnx

from keras._tf_keras.keras.applications.mobilenet import preprocess_input
from spec_augment import SpecAugment

from preprocess.preprocess import GENRES
from tensorflow import keras
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.layers import (Dense, Reshape)
from autopool.tf import AutoPool1D
from keras._tf_keras.keras.constraints import non_neg
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.layers import Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.losses import CategoricalCrossentropy
from modeling.custom_callback import CustomCallback
from keras._tf_keras.keras.callbacks import ModelCheckpoint
from keras._tf_keras.keras.models import load_model

#models folder
MODELS = "models/"

#model parameters
REG = 0.01
METRIC = "accuracy"
ACTIVATE = "relu"
EPOCHS = 50
BATCH = 32
LEARN_RT = 0.01
INN_SZ = 100
DROP = 0.0
IN_SHAPE = (64, 1876, 3)

#time masks, use ints in [0-100]
TM_MASKS = [25, 75]

class Model():
    #normalize data for modeling
    def get_normalized_data(self, split):
        normalized_split =  preprocess_input(split)
        return normalized_split

    #cast data to float32
    def data_to_f32(self, train_split):
        train_split_cast = tf.constant(train_split, dtype=tf.float32)
        return train_split_cast

    def get_time_mask(self, mask_value):
        spec_aug = SpecAugment(
            time_mask_param=mask_value,
            n_time_mask=1,
            mask_value=0,
            freq_mask_param=0,
            n_freq_mask=0
        )

        return spec_aug
    
    #train model with defined parameters
    def make_model(self):
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=IN_SHAPE
        )
        
        base_model.trainable = False
        inputs = keras.Input(shape=IN_SHAPE)
        base = base_model(inputs, training=False)

        height, width, channels = base.shape[1], base.shape[2], base.shape[3]
        base_reshaped = Reshape((height * width, channels))(base)
        vectors = AutoPool1D(
            axis=1, 
            kernel_constraint=non_neg(), 
            kernel_regularizer=l2(REG)
        )(base_reshaped)

        inner_layer = Dense(INN_SZ, activation=ACTIVATE)(vectors)
        drop = Dropout(rate=DROP)(inner_layer)

        outputs = Dense(len(GENRES))(drop)
        model = keras.Model(inputs, outputs)

        optimizer = Adam(learning_rate=LEARN_RT)
        loss = CategoricalCrossentropy(from_logits=True)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[METRIC]
        )

        return model

    def get_checkpoint(self):
        checkpoint = ModelCheckpoint(
            "models/mdl_tw_{tm_win}_{epoch:02d}_val_acc_{val_accuracy:.3f}.keras",
            save_best_only=True,
            monitor="val_accuracy",
            mode="max"
        )

        return checkpoint

    def train_model(self, cast_train_split, y_train, normalized_vldtn_split, y_vldtn):
        scores = {}

        #train a model for each time mask, save scores
        for mask_value in TM_MASKS:
            print(f"Time mask: {mask_value}")

            time_mask = self.get_time_mask(mask_value)
            masked_train_split = time_mask(cast_train_split)

            model_aug_tm = self.make_model()
            history = model_aug_tm.fit(
                masked_train_split,
                y_train,
                epochs=EPOCHS,
                batch_size=BATCH,
                validation_data=(normalized_vldtn_split, y_vldtn),
                callbacks=[CustomCallback(mask_value), self.get_checkpoint()]
            )

            scores[str(mask_value)] = history.history
            print("====")
        print("Check models folder to see stored models.")
    
    def load_keras_model(self, keras_model_name):
        keras_model = load_model(
            MODELS+keras_model_name,
            custom_objects={"AutoPool1D": AutoPool1D}
        )
        return keras_model
    
    def keras_to_onnx(self, keras_model, onnx_model_name):
        spec = tf.TensorSpec(keras_model.input_shape, keras_model.input.dtype, name="input")
        onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature=[spec], opset=13)
        
        with open(MODELS+onnx_model_name, "wb") as f_out:
            f_out.write(onnx_model.SerializeToString())
        
        print(f"Saved {onnx_model_name} to {MODELS} folder")