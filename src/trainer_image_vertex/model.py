import pprint
import os
import sys
import pickle
import subprocess
import datetime
import hypertune
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from io import BytesIO
from tensorflow.python.lib.io import file_io
from tensorflow.keras.layers import (
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling3D,
    Softmax
)

pp = pprint.PrettyPrinter(depth=8)
AIP_MODEL_DIR = os.environ["AIP_MODEL_DIR"]
MODEL_FILENAME = "model.pkl"

labels_to_numeric = {
    "Arterial": 0,
    "Late": 1,
    "Non-Contrast": 2,
    "Venous": 3
}

numeric_to_labels = {
    0:   "Arterial",
    1:   "Late",
    2:   "Non-Contrast",
    3:  "Venous"
}


def reshape_and_normalize(images):

    ### START CODE HERE

    # Reshape the images to add an extra dimension
    images = images.reshape((images.shape[0], images.shape[1], images.shape[2], images.shape[3], 1))

    # Normalize pixel values
    max_value = np.max(images)
    images = images/max_value

    ### END CODE HERE

    return images, max_value# Reload the images in case you run this cell multiple times


def load_and_format_data_from_gcs(sample_dir):
    # sample_dir="gs://capstone-datasets/train_3d.csv"
    print(f"sample_dir: {sample_dir}")
    file_list = file_io.read_file_to_string(sample_dir).split("\n")
    images = np.array([np.load(BytesIO(file_io.read_file_to_string(file, binary_mode=True)))
                       for file in file_list if file])
    labels = np.array([os.path.basename(file).split("_")[4] for file in file_list if file])
    labels = np.array([labels_to_numeric[label] for label in labels])
    one_hots = to_categorical(labels)

    images_tranformed, max_value = reshape_and_normalize(images)
    return images_tranformed, one_hots


class myCallback(tf.keras.callbacks.Callback):
    # Define the method that checks the accuracy at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.995:
            print("Reached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True
            

def convolutional_model(dropout_rate=0.2, l2_regularization_lambda=0.1, training_images_shape=(32, 128, 128, 1)):
    ### START CODE HERE

    # Define the model
    model = tf.keras.models.Sequential([
        # hub.KerasLayer("https://tfhub.dev/google/HRNet/scannet-hrnetv2-w48/1", trainable=False),
        # tf.keras.layers.Dropout(rate=0.2)
        tf.keras.layers.Conv3D(16, 3, activation='relu', input_shape=training_images_shape),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid'),
        tf.keras.layers.Conv3D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu,
                              kernel_regularizer=keras.regularizers.l2(l=l2_regularization_lambda)),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu,
                              kernel_regularizer=keras.regularizers.l2(l=l2_regularization_lambda)),
        tf.keras.layers.Dropout(rate=dropout_rate),
        tf.keras.layers.Dense(4),
        tf.keras.layers.Softmax()
    ])
    ### END CODE HERE

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Instantiate the HyperTune reporting object
hpt = hypertune.HyperTune()


# Reporting callback
class HPTCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        global hpt
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='val_loss',
            metric_value=logs['val_accuracy'],
            global_step=epoch)


def train_and_evaluate(args):

    training_images, one_hots_train = load_and_format_data_from_gcs(args["train_data_path"])
    print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
    print(f"Shape of training set after reshaping: {training_images.shape}\n")
    print(f"Shape of one image after reshaping: {training_images[0].shape}")

    valid_images, one_hots_valid = load_and_format_data_from_gcs(args["eval_data_path"])
    print(f"Maximum pixel value after normalization: {np.max(valid_images)}\n")
    print(f"Shape of training set after reshaping: {valid_images.shape}\n")
    print(f"Shape of one image after reshaping: {valid_images[0].shape}")

    model = convolutional_model(args["dropout_rate"], args["l2_regularization_lambda"],
                                training_images_shape=training_images.shape[1:])
    print("Here is our model so far:\n")
    print(model.summary())

    # checkpoint_path = os.path.join(args["output_dir"], "checkpoints/phase_contrast")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=AIP_MODEL_DIR, verbose=1, save_weights_only=True)

    callbacks = myCallback()

    history = model.fit(
        x=training_images,
        y=one_hots_train,
        validation_data=(valid_images, one_hots_valid),
        epochs=args["num_epochs"],
        callbacks=[callbacks, cp_callback, HPTCallback()])

    # history.history
    # {'loss': [0.3386789858341217, 0.1543138176202774],
    #  'sparse_categorical_accuracy': [0.9050400257110596, 0.9548400044441223],
    #  'val_loss': [0.19569723308086395, 0.14253544807434082],
    #  'val_sparse_categorical_accuracy': [0.9426000118255615, 0.9592999815940857]}
    hptune = args["hptune"]
    history_history = history.history
    print(f"history_history: {pp.pformat(history_history)}")
    final_val_accuracy = history_history.get("val_accuracy")[-1]
    print(f"Final Validation accuracy: {final_val_accuracy}")
    if hptune:
        # Log it with hypertune
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="final_val_accuracy", metric_value=final_val_accuracy
        )

    # Save the model
    if not hptune:
        # with open(MODEL_FILENAME, "wb") as model_file:
        #     pickle.dump(model, model_file)
        # subprocess.check_call(
        #     ["gsutil", "cp", MODEL_FILENAME, AIP_MODEL_DIR], stderr=sys.stdout
        # )
        # print(f"Saved model in: {AIP_MODEL_DIR}")

        tf.saved_model.save(
            obj=model, export_dir=AIP_MODEL_DIR)  # with default serving function

    print("Exported trained model to {}".format(AIP_MODEL_DIR))