"""
This module provides functions for training the prediction models k-fold-cross validation manner
"""

import argparse
import tensorflow as tf

from train_and_predict_scripts.utilities import TrainModelSpec, train_main

from OT_deep_score_src.general_utilities import Model_type, Data_type, Encoding_type, Model_task

# TODO: Note that read threshold is shared to both CHANGE-seq and
# GUIDE-seq model which is not something we wish in general


def train_on_guide_seq_or_change_seq_specs(
        model_version, model_parameters, include_distance_feature, sample_weight,
        encoding_type, read_threshold, aligned, data_type):
    """
    Generates training model specifications for CHANGE-seq or GUIDE-seq datasets.

    Args:
        model_version (str):  A string identifier for the model version.
        model_parameters (dict): A dictionary containing model-specific parameters.
        include_distance_feature (bool): If True, includes the distance feature in training.
        sample_weight (bool): If True, uses sample weights during training.
        encoding_type (Encoding_type):  The encoding type.
        read_threshold (int): Threshold for filtering off-targets by read count.
        aligned (bool): Indicates whether the sequences are aligned.
        data_type (Data_type):  The dataset type (CHANGE-seq or GUIDE-seq).

    Returns:
        list: A list of `TrainModelSpec` objects, each representing a model configuration.
    """
    train_model_spec_list = [
        TrainModelSpec(
            model_type=model_type, predict_distance=False, model_version=model_version,
            model_parameters=model_parameters,
            include_distance_feature=include_distance_feature, sample_weight=sample_weight,
            encoding_type=encoding_type, flat_encoding=False,
            read_threshold=read_threshold, aligned=aligned,
            data_type=data_type) for model_type in [
                Model_type.C_1, Model_type.C_2, Model_type.C_3]]
    train_model_spec_list.append(
        TrainModelSpec(
            model_type=Model_type.XGBOOST, predict_distance=False, model_version=model_version, model_parameters=None,
            include_distance_feature=include_distance_feature, sample_weight=sample_weight,
            encoding_type=encoding_type, flat_encoding=True,
            read_threshold=read_threshold, aligned=aligned,
            data_type=data_type))

    return train_model_spec_list


def train_tl_specs(model_version, model_parameters, include_distance_feature, sample_weight,
                   encoding_type, read_threshold, aligned):
    """
    Generates training specifications for transfer learning models.

    Args:
        model_version (str): A string identifier for the model version.
        model_parameters (dict): A dictionary containing model-specific parameters.
        include_distance_feature (bool): If True, includes the distance feature in training.
        sample_weight (bool): If True, uses sample weights during training.
        encoding_type (Encoding_type):  The encoding type.
        read_threshold (int): Threshold for filtering off-targets by read count.
        aligned (bool): Indicates whether the sequences are aligned.

    Returns:
        list: A list of `TrainModelSpec` objects, each representing a transfer learning model configuration.
    """
    # generate the models specs for training
    c_model_types, c_models_spec_dict = [Model_type.C_1, Model_type.C_2, Model_type.C_3], {}
    # Train model specs for the TL models
    for c_model_type in c_model_types:
        c_models_spec_dict[c_model_type] = TrainModelSpec(
            model_type=c_model_type, predict_distance=False,
            model_version="{}_continue_from_change_seq".format(model_version),
            model_parameters=model_parameters,
            include_distance_feature=include_distance_feature, sample_weight=sample_weight,
            encoding_type=encoding_type, flat_encoding=False, data_type=Data_type.GUIDE_SEQ,
            read_threshold=read_threshold, aligned=aligned,
            continue_train=True)

    # Train model specs of regular trained on CHANGE-seq (CH) data. They get as tf_models_spec the TL models specs
    train_model_spec_list = [
        TrainModelSpec(
            model_type=c_model_type, predict_distance=False, model_version=model_version,
            model_parameters=model_parameters,
            include_distance_feature=include_distance_feature, sample_weight=sample_weight,
            encoding_type=encoding_type, flat_encoding=False,
            read_threshold=read_threshold, aligned=aligned,
            tf_models_spec_list=[c_models_spec_dict[c_model_type]]) for c_model_type in c_model_types
            ]

    return train_model_spec_list


def validate_args_params(model_version, data_type):
    """
    Validates the model version and data type arguments, and sets appropriate model parameters.

    Args:
        model_version (str): A string identifier for the model version.
        data_type (Data_type):  The dataset type (CHANGE-seq, GUIDE-seq, or "TL").

    Returns:
        dict: A dictionary containing model parameters.

    Raises:
        ValueError: If the data type or model version is unsupported.
    """
    # validate model version
    if model_version.startswith("5_revision"):
        model_parameters = {"batch_size": 512, "learning_rate": 0.0005}
    elif model_version.startswith("4_revision"):
        model_parameters = {"batch_size": 512}
    else:
        raise ValueError("Model version is not supported. Add new version parameters to code to support.")

    if data_type not in [Data_type.CHANGE_SEQ, Data_type.GUIDE_SEQ, "TL"]:
        raise ValueError("Got unsupported data type for training.")

    return model_parameters


def train_handler(
        read_threshold, sample_weight, num_ensembles, model_version, data_type, model_tasks):
    """
    Handles k-fold cross-validation training.

    Args:
        read_threshold (int):  Threshold for filtering off-targets by read count.
        sample_weight (bool):  If True, uses sample weights during training.
        num_ensembles (int): The number of model ensembles to train.
        model_version (str): A string identifier for the model version.
        data_type (str): Dataset and training type ("CHANGEseq", "GUIDEseq", or "TL" for transfer learning).
        model_tasks (tuple of Model_task): Specifies the model task(s) to train,
                                           either classification, regression, or both.
    """
    # validate model version and data_type
    model_parameters = validate_args_params(model_version, data_type)

    # train parameters
    aligned = True
    include_distance_feature = False
    encoding_type = Encoding_type.ONE_HOT  # can be changed to other encoding
    k_fold_number = 10
    load_predefined_folds = True

    if num_ensembles > 1:
        model_versions = ["{}_ensemble_{}".format(model_version, i) for i in range(num_ensembles)]
    else:
        model_versions = [model_version]

    for model_version in model_versions:
        if data_type == "TL":
            train_model_spec_list = train_tl_specs(
                model_version, model_parameters, include_distance_feature,
                sample_weight, encoding_type, read_threshold, aligned)
        else:
            train_model_spec_list = train_on_guide_seq_or_change_seq_specs(
                model_version, model_parameters, include_distance_feature, sample_weight,
                encoding_type, read_threshold, aligned, data_type)

        train_main(train_model_spec_list, model_tasks=model_tasks,
                   k_fold_number=k_fold_number, load_predefined_folds=load_predefined_folds)


def main():
    """
    Parses command-line arguments, initializes GPU settings, and calls the main training function.
    """
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    parser = argparse.ArgumentParser(
        description="10 folds train on CHANGE-seq/GUIDE-seq (CH/GU) or"
        " continue training from CHANGE-seq to GUIDE-seq data (Transfer Learning models)."
        " Training options are not flexible, see train 1 fold for model felxible train ")
    parser.add_argument("-th", "--read_threshold", type=int, default=0, help="Reads threshold.")
    parser.add_argument("-sw", "--sample_weight", action="store_true",
                        default=False, help="Use Sample weight in training.")
    parser.add_argument("-ens", "--num_ensembles", type=int,
                        default=1, help="Number of ensembles. Default is 1 - no ensembles.")
    parser.add_argument("-ver", "--model_version", type=str,
                        help="String representing the model version, used to load parameters and save model.")
    parser.add_argument("-d_type", "--data_type", type=str, default=Data_type.CHANGE_SEQ,
                        help="Train data type. Can be either CHANGEseq, GUIDEseq, or TL for transfer learning model")
    parser.add_argument("-c", "--train_classification", action="store_true",
                        default=False, help="Train classification task models.")
    parser.add_argument("-r", "--train_regression", action="store_true",
                        default=False, help="Train regression task models.")
    args = parser.parse_args()

    read_threshold = args.read_threshold
    sample_weight = args.sample_weight
    num_ensembles = args.num_ensembles
    model_version = args.model_version
    data_type = args.data_type
    train_classification = args.train_classification
    train_regression = args.train_regression
    print("**********************************************")
    print("Read threshold:", read_threshold)
    print("Sample weight:", sample_weight)
    print("Number of ensembles:", num_ensembles)
    print("Model version:", model_version)
    print("Data type:", data_type)
    print("Train classification:", train_classification)
    print("Train regression:", train_regression)
    print("**********************************************")
    if train_classification and train_regression:
        model_tasks = (Model_task.CLASSIFICATION_TASK, Model_task.REGRESSION_TASK)
    elif train_classification:
        model_tasks = (Model_task.CLASSIFICATION_TASK,)
    elif train_regression:
        model_tasks = (Model_task.REGRESSION_TASK,)
    else:
        raise ValueError("Model task needs to be defined")
    train_handler(read_threshold, sample_weight, num_ensembles, model_version, data_type, model_tasks)


if __name__ == "__main__":
    main()
