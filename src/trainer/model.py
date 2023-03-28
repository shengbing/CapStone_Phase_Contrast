import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import (
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling3D,
    Softmax
)

training_sample_dir = "/home/jupyter/asl-ml-immersion/notebooks/capstone_project/train-3d-npy"
training_images = np.array([np.load(training_sample_dir + "/" + file) for file in  os.listdir(training_sample_dir)])

training_labels = np.array([file.split("_")[4]  for file in  os.listdir(training_sample_dir)])

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

training_labels = np.array([labels_to_numeric[label]  for label in training_labels])
one_hots_train = to_categorical(training_labels)
def reshape_and_normalize(images):

    ### START CODE HERE

    # Reshape the images to add an extra dimension
    images = images.reshape((images.shape[0], images.shape[1], images.shape[2], images.shape[3], 1))

    # Normalize pixel values
    max_value = np.max(images)
    images = images/max_value

    ### END CODE HERE

    return images, max_value# Reload the images in case you run this cell multiple times

# Reload the images in case you run this cell multiple times
training_sample_dir = "/home/jupyter/asl-ml-immersion/notebooks/capstone_project/train-3d-npy"
training_images = np.array([np.load(training_sample_dir + "/" + file) for file in os.listdir(training_sample_dir)])

# Apply your function
training_images, max_value = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")


class myCallback(tf.keras.callbacks.Callback):
    # Define the method that checks the accuracy at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.995:
            print("Reached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True
            

callbacks = myCallback()


def convolutional_model():
    ### START CODE HERE

    # Define the model
    model = tf.keras.models.Sequential([
        # hub.KerasLayer("https://tfhub.dev/google/HRNet/scannet-hrnetv2-w48/1", trainable=False),
        # tf.keras.layers.Dropout(rate=0.2)
        tf.keras.layers.Conv3D(16, 3, activation='relu',input_shape=training_images.shape[1:]),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2,2), strides=(2, 2,2), padding='valid'),
        tf.keras.layers.Conv3D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2,2), strides=(2, 2,2), padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, kernel_regularizer=keras.regularizers.l2(l=0.1)),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, kernel_regularizer=keras.regularizers.l2(l=0.1)),
        tf.keras.layers.Dropout(rate=0.20),
        tf.keras.layers.Dense(4),
        tf.keras.layers.Softmax()
    ])
    ### END CODE HERE

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


model = convolutional_model()

model.summary()


# Reload the images in case you run this cell multiple times
valid_sample_dir = "/home/jupyter/asl-ml-immersion/notebooks/capstone_project/valid-3d-npy"
valid_images = np.array([np.load(valid_sample_dir + "/" + file)  for file in  os.listdir(valid_sample_dir)])

# Apply your function
valid_images, max_value = reshape_and_normalize(valid_images)

print(f"Maximum pixel value after normalization: {np.max(valid_images)}\n")
print(f"Shape of training set after reshaping: {valid_images.shape}\n")
print(f"Shape of one image after reshaping: {valid_images[0].shape}")

valid_labels = np.array([file.split("_")[4] for file in os.listdir(valid_sample_dir)])
valid_labels = np.array([labels_to_numeric[label] for label in valid_labels])
one_hots_valid = to_categorical(valid_labels)

history = model.fit(x=training_images, y=one_hots_train, validation_data=(valid_images, one_hots_valid),
                    epochs=40, callbacks=[callbacks])





def create_input_layers():
    """Creates dictionary of input layers for each feature.

    Returns:
        Dictionary of `tf.Keras.layers.Input` layers for each feature.
    """
    deep_inputs = {
        colname: tf.keras.layers.Input(
            name=colname, shape=(1,), dtype="float32"
        )
        for colname in NUMERICAL_COLUMNS
    }

    wide_inputs = {
        colname: tf.keras.layers.Input(name=colname, shape=(1,), dtype="string")
        for colname in CATEGORICAL_COLUMNS
    }

    inputs = {**wide_inputs, **deep_inputs}

    return inputs









def build_wide_deep_model(dnn_hidden_units=[64, 32]):
    """Builds wide and deep model using Keras Functional API.

    Returns:
        `tf.keras.models.Model` object.
    """
    # Create input layers
    inputs = create_input_layers()

    # transform raw features for both wide and deep
    wide, deep = transform(inputs, nembeds)

    # The Functional API in Keras requires: LayerConstructor()(inputs)
    wide_inputs = tf.keras.layers.Concatenate()(wide.values())
    deep_inputs = tf.keras.layers.Concatenate()(deep.values())

    # Get output of model given inputs
    output = get_model_outputs(wide_inputs, deep_inputs, dnn_hidden_units)

    # Build model and compile it all together
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=[rmse, "mse"])

    return model


# Instantiate the HyperTune reporting object
hpt = hypertune.HyperTune()

# Reporting callback
class HPTCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        global hpt
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='val_rmse',
            metric_value=logs['val_rmse'],
            global_step=epoch)


def train_and_evaluate(args):
    model = build_wide_deep_model(args["nnsize"], args["nembeds"])
    print("Here is our Wide-and-Deep architecture so far:\n")
    print(model.summary())

    trainds = load_dataset(
        args["train_data_path"],
        args["batch_size"],
        tf.estimator.ModeKeys.TRAIN)

    evalds = load_dataset(
        args["eval_data_path"], 1000, tf.estimator.ModeKeys.EVAL)
    if args["eval_steps"]:
        evalds = evalds.take(count=args["eval_steps"])

    num_batches = args["batch_size"] * args["num_epochs"]
    steps_per_epoch = args["train_examples"] // num_batches

    checkpoint_path = os.path.join(args["output_dir"], "checkpoints/babyweight")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_weights_only=True)

    history = model.fit(
        trainds,
        validation_data=evalds,
        epochs=args["num_epochs"],
        steps_per_epoch=steps_per_epoch,
        verbose=2,  # 0=silent, 1=progress bar, 2=one line per epoch
        callbacks=[cp_callback, HPTCallback()])

    EXPORT_PATH = os.path.join(
        args["output_dir"], datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    tf.saved_model.save(
        obj=model, export_dir=EXPORT_PATH)  # with default serving function

    print("Exported trained model to {}".format(EXPORT_PATH))