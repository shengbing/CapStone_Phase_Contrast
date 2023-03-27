import datetime
import os
import shutil
import numpy as np
import tensorflow as tf
import hypertune

# Determine CSV, label, and key columns
CSV_COLUMNS = [
    "weight_pounds",
    "is_male",
    "mother_age",
    "plurality",
    "gestation_weeks",
]
LABEL_COLUMN = "weight_pounds"

NUMERICAL_COLUMNS = ["mother_age", "gestation_weeks"]
CATEGORICAL_COLUMNS = ["is_male", "plurality"]

# Set default values for each CSV column.
# Treat is_male and plurality as strings.
DEFAULTS = [[0.0], ["null"], [0.0], ["null"], [0.0]]


def features_and_labels(row_data):
    """Splits features and labels from feature dictionary.

    Args:
        row_data: Dictionary of CSV column names and tensor values.
    Returns:
        Dictionary of feature tensors and label tensor.
    """
    label = row_data.pop(LABEL_COLUMN)

    return row_data, label  # features, label


def load_dataset(pattern, batch_size=1, mode=tf.estimator.ModeKeys.EVAL):
    """Loads dataset using the tf.data API from CSV files.

    Args:
        pattern: str, file pattern to glob into list of files.
        batch_size: int, the number of examples per batch.
        mode: tf.estimator.ModeKeys to determine if training or evaluating.
    Returns:
        `Dataset` object.
    """
    # Make a CSV dataset
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=pattern,
        batch_size=batch_size,
        column_names=CSV_COLUMNS,
        column_defaults=DEFAULTS,
    )

    # Map dataset to features and label
    dataset = dataset.map(map_func=features_and_labels)  # features, label

    # Shuffle and repeat for training
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=1000).repeat()

    # Take advantage of multi-threading; 1=AUTOTUNE
    dataset = dataset.prefetch(buffer_size=1)

    return dataset


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


def transform(inputs, nembeds):
    """Creates dictionary of transformed inputs.

    Returns:
        Dictionary of transformed Tensors
    """

    deep = {}
    wide = {}

    buckets = {
        "mother_age": np.arange(15, 45, 1).tolist(),
        "gestation_weeks": np.arange(17, 47, 1).tolist(),
    }
    bucketized = {}

    for numerical_column in NUMERICAL_COLUMNS:
        deep[numerical_column] = inputs[numerical_column]
        bucketized[numerical_column] = tf.keras.layers.Discretization(buckets[numerical_column])(inputs[numerical_column])
        wide[f"btk_{numerical_column}"] = tf.keras.layers.CategoryEncoding(
            num_tokens=len(buckets[numerical_column]) + 1, output_mode="one_hot"
        )(bucketized[numerical_column])

    crossed = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        num_bins=len(buckets["mother_age"]) * len(buckets["gestation_weeks"])
    )((bucketized["mother_age"], bucketized["gestation_weeks"]))

    deep["age_gestation_embeds"] = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(
            input_dim=len(buckets["mother_age"])
                      * len(buckets["gestation_weeks"]),
            output_dim=nembeds,
        )(crossed)
    )

    vocab = {
        "is_male": ["True", "False", "Unknown"],
        "plurality": [
            "Single(1)",
            "Twins(2)",
            "Triplets(3)",
            "Quadruplets(4)",
            "Quintuplets(5)",
            "Multiple(2+)",
        ],
    }

    for categorical_column in CATEGORICAL_COLUMNS:
        wide[categorical_column] = tf.keras.layers.StringLookup(
            vocabulary=vocab[categorical_column], output_mode="one_hot"
        )(inputs[categorical_column])

    return wide, deep

def get_model_outputs(wide_inputs, deep_inputs, dnn_hidden_units):
    """Creates model architecture and returns outputs.

    Args:
        wide_inputs: Dense tensor used as inputs to wide side of model.
        deep_inputs: Dense tensor used as inputs to deep side of model.
        dnn_hidden_units: List of integers where length is number of hidden
            layers and ith element is the number of neurons at ith layer.
    Returns:
        Dense tensor output from the model.
    """
    # Hidden layers for the deep side
    layers = [int(x) for x in dnn_hidden_units.split()]
    deep = deep_inputs
    for layerno, numnodes in enumerate(layers):
        deep = tf.keras.layers.Dense(
            units=numnodes, activation="relu", name=f"dnn_{layerno + 1}"
        )(deep)
    deep_out = deep

    # Linear model for the wide side
    wide_out = tf.keras.layers.Dense(
        units=10, activation="relu", name="linear"
    )(wide_inputs)

    # Concatenate the two sides
    both = tf.keras.layers.Concatenate(name="both")([deep_out, wide_out])

    # Final output is a linear activation because this is regression
    output = tf.keras.layers.Dense(units=1, activation="linear", name="weight")(
        both
    )

    return output


def rmse(y_true, y_pred):
    """Calculates RMSE evaluation metric.

    Args:
        y_true: tensor, true labels.
        y_pred: tensor, predicted labels.
    Returns:
        Tensor with value of RMSE between true and predicted labels.
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


def build_wide_deep_model(dnn_hidden_units=[64, 32], nembeds=3):
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