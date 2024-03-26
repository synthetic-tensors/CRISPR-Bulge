"""
This module provides functions for training the prediction models on entire dataset
"""

import argparse
import tensorflow as tf

from OT_deep_score_src.general_utilities import Model_type, Data_type, Encoding_type, Model_task
from train_and_predict_scripts.utilities import TrainModelSpec, train_main


# TODO: Note that read threshold is shared to both CHANGE-seq and
# GUIDE-seq model which is not something we wish in general


def train_on_one_dataset(
        model_version, model_parameters, include_distance_feature, sample_weight,
        encoding_type, read_threshold, aligned, data_types_to_exclude, data_type, model_type):
    """
    Generates training model specifications for model trained on a single dataset.

    Args:
        model_version (str):  String representing the model version.
        model_parameters (dict): A dictionary containing model-specific parameters.
        include_distance_feature (bool): If True, includes the distance feature in training.
        sample_weight (bool): If True, uses sample weights during training.
        encoding_type (Encoding_type): Type of encoding.
        read_threshold (int): Threshold for filtering off-targets by read count.
        aligned (bool): Indicates whether the sequences are aligned.
        data_types_to_exclude (list of Data_type or None):  Dataset types to exclude sgRNAs from (e.g., "RHAMP_SEQ").
        data_type (Data_type):  The dataset type to train on.
        model_type (Model_type or None): If specified, trains only this model type.
                                         Otherwise, trains all supported types.

    Returns:
        list: A list of `TrainModelSpec` objects, each representing a model configuration.
    """
    if model_type is None:
        c_model_types = [Model_type.C_1, Model_type.C_2, Model_type.C_3]
    elif model_type != Model_type.XGBOOST:
        c_model_types = [model_type]
    else:
        c_model_types = []
    train_model_spec_list = []

    train_model_spec_list.extend([
        TrainModelSpec(
            model_type=c_model_type, predict_distance=False, model_version=model_version,
            model_parameters=model_parameters,
            include_distance_feature=include_distance_feature, sample_weight=sample_weight,
            encoding_type=encoding_type, flat_encoding=False,
            data_type=data_type, data_types_to_exclude=data_types_to_exclude,
            read_threshold=read_threshold, aligned=aligned) for c_model_type in c_model_types
            ])
    if model_type is None or model_type == Model_type.XGBOOST:
        train_model_spec_list.append(
                TrainModelSpec(
                    model_type=Model_type.XGBOOST, predict_distance=False,
                    model_version=model_version, model_parameters=None,
                    include_distance_feature=include_distance_feature, sample_weight=sample_weight,
                    encoding_type=encoding_type, flat_encoding=True,
                    data_type=data_type, data_types_to_exclude=data_types_to_exclude,
                    read_threshold=read_threshold, aligned=aligned)
            )

    return train_model_spec_list


def train_tl_specs(model_version, model_parameters, include_distance_feature, sample_weight,
                   encoding_type, read_threshold, aligned, data_types_to_exclude,
                   data_type, model_type):
    """
    Generates training specifications for transfer learning models.

    Args:
        model_version (str): String representing the model version.
        model_parameters (dict): A dictionary containing model-specific parameters.
        include_distance_feature (bool): If True, includes the distance feature in training.
        sample_weight (bool): If True, uses sample weights during training.
        encoding_type (Encoding_type):  Type of encoding.
        read_threshold (int): Threshold for filtering off-targets by read count.
        aligned (bool): Indicates whether the sequences are aligned.
        data_types_to_exclude (list of Data_type or None): Dataset types to exclude sgRNAs from (e.g., "RHAMP_SEQ").
        data_type (Data_type): The dataset type used for continued transfer learning.
        model_type (Model_type or None): If specified, trains only this model type.
                                         Otherwise, trains all supported types.

    Returns:
        list: A list of `TrainModelSpec` objects for transfer learning model configurations.
    """
    # generate the models specs for training
    if model_type is None:
        c_model_types, c_models_spec_dict = [Model_type.C_1, Model_type.C_2, Model_type.C_3], {}
    else:
        c_model_types, c_models_spec_dict = [model_type], {}
    # Train model specs for the TL models
    for c_model_type in c_model_types:
        c_models_spec_dict[c_model_type] = TrainModelSpec(
            model_type=c_model_type, predict_distance=False,
            model_version="{}_continue_from_change_seq".format(model_version),
            model_parameters=model_parameters,
            include_distance_feature=include_distance_feature, sample_weight=sample_weight,
            encoding_type=encoding_type, flat_encoding=False, data_type=data_type,
            data_types_to_exclude=data_types_to_exclude, continue_train=True,
            read_threshold=read_threshold, aligned=aligned)

    # Train model specs of regular trained on CHANGE-seq (CH) data. They get as tf_models_spec the TL models specs
    train_model_spec_list = [
        TrainModelSpec(
            model_type=c_model_type, predict_distance=False, model_version=model_version,
            model_parameters=model_parameters,
            include_distance_feature=include_distance_feature, sample_weight=sample_weight,
            encoding_type=encoding_type, flat_encoding=False,
            data_type=Data_type.CHANGE_SEQ, data_types_to_exclude=data_types_to_exclude,
            read_threshold=read_threshold, aligned=aligned,
            tf_models_spec_list=[c_models_spec_dict[c_model_type]]) for c_model_type in c_model_types
            ]

    return train_model_spec_list


def validate_args_params(model_version, data_type, data_types_to_exclude, transfer_learning, model_type):
    """
    Validates arguments, sets model parameters based on the model version,
    and updates the model version if data exclusion is specified.

    Args:
        model_version (str): String representing the model version.
        data_type (Data_Type): The dataset type.
        data_types_to_exclude (Data_type or None): Dataset types to exclude sgRNAs from (e.g., "RHAMP_SEQ").
                                                   This function supports only one dataset to exclude.
        transfer_learning (bool):  Indicates whether to conduct transfer learning.
        model_type (Model_type or None): If specified, trains only this model type.
                                         Otherwise, trains all supported types.

    Returns:
        tuple:
            * model_parameters (dict): Model-specific parameters.
            * model_version (str): Potentially updated model version if data exclusion is used.
            * data_types_to_exclude (list of Data_type): List of datasets to exclude.

    Raises:
       ValueError: If invalid combinations of data type, transfer learning, or excluded data types are provided.
    """
    # validate model version
    if model_version.startswith("5_revision"):
        model_parameters = {"batch_size": 512, "learning_rate": 0.0005}
    elif model_version.startswith("4_revision"):
        model_parameters = {"batch_size": 512}
    else:
        raise ValueError("Model version is not supported. Add new version parameters to code to support.")

    if transfer_learning:
        if data_type not in [Data_type.GUIDE_SEQ, Data_type.FULL_GUIDE_SEQ]:
            raise ValueError("Got unsupported data type to continue training on.")
    else:
        if data_type not in [Data_type.CHANGE_SEQ, Data_type.GUIDE_SEQ, Data_type.FULL_GUIDE_SEQ, Data_type.CRISPR_NET]:
            raise ValueError("Got unsupported data type for training.")

    if data_types_to_exclude is not None:
        if data_types_to_exclude not in [Data_type.NEW_GUIDE_SEQ, Data_type.RHAMP_SEQ]:
            raise ValueError("Support exclude sgRNAs of RHAMP_SEQ (of CHANGE-seq) and NEW_GUIDE_SEQ")
        data_types_to_exclude = [data_types_to_exclude]
        model_version = "{}_exclude_{}".format(model_version, data_types_to_exclude[0])

    if model_type not in [None, Model_type.C_1, Model_type.C_2, Model_type.C_3, Model_type.XGBOOST]:
        raise ValueError("Got unsupported data type for training.")
    if model_type == Model_type.XGBOOST and transfer_learning:
        raise ValueError("XGBoost is not supporting transfer learning.")

    return model_parameters, model_version, data_types_to_exclude


def train_handler(
        read_threshold, sample_weight, data_type,
        data_types_to_exclude, model_version, num_ensembles,
        transfer_learning, model_tasks, model_type):
    """
    Handles model training, including model parameter validation, train specification generation, and train execution.

    Args:
        read_threshold (int): Threshold for filtering off-targets by read count.
        sample_weight (bool): If True, uses sample weights during training.
        data_type (Data_type): Dataset type for training or transfer learning continuation.
        data_types_to_exclude (Data_type or None): Dataset types to exclude sgRNAs from (e.g., "RHAMP_SEQ").
                                                   This function supports only one dataset to exclude.
        model_version (str): A string representing the model version.
        num_ensembles (int): The number of model ensembles to train.
        transfer_learning (bool): If True, performs transfer learning.
        model_tasks (tuple of Model_task): Specifies the model task(s) to train,
                                           either classification, regression, or both.
        model_type (Model_type or None): If specified, trains only this model type.
                                         Otherwise, trains all supported types.
    """
    # validate model version, data_type, and data type to exclude
    model_parameters, model_version, data_types_to_exclude = validate_args_params(
        model_version, data_type, data_types_to_exclude, transfer_learning, model_type)

    # train parameters
    aligned = True
    include_distance_feature = False
    encoding_type = Encoding_type.ONE_HOT  # can be changed to other encoding
    k_fold_number = 1

    if num_ensembles > 1:
        model_versions = ["{}_ensemble_{}".format(model_version, i) for i in range(num_ensembles)]
    else:
        model_versions = [model_version]

    for model_version in model_versions:
        if transfer_learning:
            train_model_spec_list = train_tl_specs(
                model_version, model_parameters, include_distance_feature, sample_weight,
                encoding_type, read_threshold, aligned, data_types_to_exclude, data_type, model_type)
        else:
            train_model_spec_list = train_on_one_dataset(
                model_version, model_parameters, include_distance_feature, sample_weight,
                encoding_type, read_threshold, aligned, data_types_to_exclude, data_type, model_type)

        train_main(train_model_spec_list, model_tasks=model_tasks, k_fold_number=k_fold_number)


def main():
    """
    Parses command-line arguments, initializes GPU settings for TensorFlow,
    and orchestrates the model training process.
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
        description="1 fold on dataset or on CHANGE-seq and then continue on GUIDE-seq data.")
    parser.add_argument("-th", "--read_threshold", type=int, default=0, help="Reads threshold.")
    parser.add_argument("-sw", "--sample_weight", action="store_true",
                        default=False, help="Use Sample weight in training.")
    parser.add_argument("-ens", "--num_ensembles", type=int,
                        default=1, help="Number of ensembles. Default is 1 - no ensembles.")
    parser.add_argument("-ver", "--model_version", type=str,
                        help="String representing the model version, used to load parameters and save model.")
    parser.add_argument("-tl", "--transfer_learning", action="store_true",
                        default=False, help="Continue training from CHANGE-seq to GUIDE-seq. XGBoost is not supported.")
    parser.add_argument(
        "-d_type", "--data_type", type=str, default=Data_type.FULL_GUIDE_SEQ,
        help="If tl=True, data type to continue on, else, data type to train on. See code for supported options.")
    parser.add_argument(
        "-exc_type", "--data_types_to_exclude",
        type=str, default=None, help="sgRNAs of data type to exclude. See code for supported options.")
    parser.add_argument("-c", "--train_classification", action="store_true",
                        default=False, help="Train classification task models.")
    parser.add_argument("-r", "--train_regression", action="store_true",
                        default=False, help="Train regression task models.")
    parser.add_argument(
        "-m_type", "--model_type", type=str, default=None,
        help="train spesific model type can be either 'c_1', 'c_2', 'c_3', or 'xgboost'."
             "Default is None, and all models types are trained")
    args = parser.parse_args()

    read_threshold = args.read_threshold
    sample_weight = args.sample_weight
    num_ensembles = args.num_ensembles
    model_version = args.model_version
    data_type = args.data_type
    data_types_to_exclude = args.data_types_to_exclude
    transfer_learning = args.transfer_learning
    train_classification = args.train_classification
    train_regression = args.train_regression
    model_type = args.model_type
    print("**********************************************")
    print("read threshold:", read_threshold)
    print("Sample weight:", sample_weight)
    print("Number of ensembles:", num_ensembles)
    print("Model version:", model_version)
    print("Transfer learning:", transfer_learning)
    if transfer_learning:
        print("Continue on dataset:", data_type)
    else:
        print("Dataset to train on:", data_type)
    print("Dataset to exclude:", data_types_to_exclude)
    print("Train classification:", train_classification)
    print("Train regression:", train_regression)
    if model_type is not None:
        print("Spesific model type:", model_type)
    print("**********************************************")
    if train_classification and train_regression:
        model_tasks = (Model_task.CLASSIFICATION_TASK, Model_task.REGRESSION_TASK)
    elif train_classification:
        model_tasks = (Model_task.CLASSIFICATION_TASK,)
    elif train_regression:
        model_tasks = (Model_task.REGRESSION_TASK,)
    else:
        raise ValueError("Model task needs to be defined")
    train_handler(read_threshold, sample_weight, data_type,
                  data_types_to_exclude, model_version, num_ensembles,
                  transfer_learning, model_tasks, model_type)


if __name__ == "__main__":
    main()
