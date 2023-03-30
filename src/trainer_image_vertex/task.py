import argparse
import json
import os
import sys
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)
import model
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        help="GCS location of training data, e.g., gs://capstone-datasets/train_3d.csv",
        required=True
    )
    parser.add_argument(
        "--eval_data_path",
        help="GCS location of evaluation data, e.g., gs://capstone-datasets/valid_3d.csv",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        help="GCS location to write checkpoints and export models",
        default=os.getenv("AIP_MODEL_DIR")
    )
    # parser.add_argument(
    #     "--batch_size",
    #     help="Number of examples to compute gradient over.",
    #     type=int,
    #     default=512
    # )
    # parser.add_argument(
    #     "--nnsize",
    #     help="Hidden layer sizes for DNN -- provide space-separated layers",
    #     default="64 64"
    # )
    parser.add_argument(
        "--num_epochs",
        help="Number of epochs to train the model.",
        type=int,
        default=40
    )
    # parser.add_argument(
    #     "--train_examples",
    #     help="""Number of examples (in thousands) to run the training job over.
    #     If this is more than actual # of examples available, it cycles through
    #     them. So specifying 1000 here when you have only 100k examples makes
    #     this 10 epochs.""",
    #     type=int,
    #     default=5000
    # )
    # parser.add_argument(
    #     "--eval_steps",
    #     help="""Positive number of steps for which to evaluate model. Default
    #     to None, which means to evaluate until input_fn raises an end-of-input
    #     exception""",
    #     type=int,
    #     default=None
    # )

    parser.add_argument(
        "--dropout_rate",
        help="""dropout_rate""",
        type=float,
        default=0.2)

    parser.add_argument(
        "--l2_regularization_lambda",
        help="""l2_regularization_lambda""",
        type=float,
        default=0.1)
    parser.add_argument("--hptune", action='store_true')  # if hptune is not set, it is False in default

    # Parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # Run the training job
    model.train_and_evaluate(arguments)