"""
    This module contains the functions for predicting
"""
import gc
import random
import time

import numpy as np
import pandas as pd

from OT_deep_score_src.data_processing_utilities import build_sequence_features
from OT_deep_score_src.models_utilities import load_model_from_pkl
from OT_deep_score_src.dataset_utilities import create_fold_sets, split_sg_rnas_into_folds
from OT_deep_score_src.general_utilities import SEED, Encoding_type, Padding_type
from OT_deep_score_src.naming_utilities import extract_model_name


random.seed(SEED)


def k_fold_predict(
        test_dataset_df, test_data_type, test_sg_rnas, model_task, train_model_spec,
        padding_type, pred_type_path, k_fold_number, load_predefined_folds, include_gmt_score,
        include_nuclea_seq_score):
    """
    Conducts k-fold cross-validation prediction.

    Args:
        test_dataset_df (pd.DataFrame): Dataset on which predictions are made.
        test_data_type (Data_type):  The dataset type.
        test_sg_rnas (list): List of sgRNA sequences to use for testing.
        model_task (Model_task): Specifies whether the task is classification or regression.
        train_model_spec (TrainModelSpec):  Contains specifications of the trained model (features, parameters, etc.).
        padding_type (Padding_type): Type of padding to use for sequence alignment.
        pred_type_path (str):  Path segment specifying prediction type.
        k_fold_number (int): Number of folds for cross-validation.
        load_predefined_folds (bool): If True, loads sgRNAs of the folds from a file.
        include_gmt_score (bool): If True, includes the GMT score as a feature.
        include_nuclea_seq_score (bool): If True, includes the Nuclea-seq score as a feature.

    Returns:
        pd.DataFrame: DataFrame containing predictions for each fold, along with fold indices.
    """
    if test_sg_rnas is None:
        raise ValueError("K-fold cannot accept sgRNAs list that is None.")
    if k_fold_number is None:
        raise ValueError("K-fold test cannot accept k_fold_number that is None.")
    elif k_fold_number == 1:
        # in this case, we want all sgRNAs in the test set.
        sg_rna_folds_list = np.array_split(test_sg_rnas, k_fold_number)
    else:
        sg_rna_folds_list = split_sg_rnas_into_folds(
            k_fold_number=k_fold_number, sg_rnas=test_sg_rnas,
            load_predefined_folds=load_predefined_folds, data_type=test_data_type)

    test_fold_datasets = []
    for fold_index, sg_rna_fold in enumerate(sg_rna_folds_list):
        print("test fold ", fold_index)
        test_fold_dataset_df, _ = create_fold_sets(
            sg_rna_fold, test_sg_rnas, test_dataset_df, exclude_sg_rnas_without_positives=False)

        test_fold_dataset_df = load_and_predict(
            test_dataset_df=test_fold_dataset_df, model_task=model_task, train_model_spec=train_model_spec,
            padding_type=padding_type, pred_type_path=pred_type_path, k_fold_number=k_fold_number,
            fold_index=fold_index, include_gmt_score=include_gmt_score,
            include_nuclea_seq_score=include_nuclea_seq_score)
        test_fold_dataset_df["fold"] = fold_index
        test_fold_datasets.append(test_fold_dataset_df)

    return pd.concat(test_fold_datasets, ignore_index=True)


def load_and_predict(
        test_dataset_df, model_task, train_model_spec, padding_type, pred_type_path,
        k_fold_number, fold_index, include_gmt_score, include_nuclea_seq_score):
    """
    Loads a pre-trained model and generates predictions for a test set.

    Args:
        test_dataset_df (pd.DataFrame): Test dataset
        model_task (Model_task): Specifies whether the task is classification or regression.
        train_model_spec (TrainModelSpec): Contains specifications of the trained model.
        padding_type (Padding_type):  Type of padding for sequence alignment.
        pred_type_path (str):  Path segment specifying prediction type.
        k_fold_number (int or None): Number of folds for cross-validation (relevant for naming).
        fold_index (int or None): Current fold index (relevant for naming).
        include_gmt_score (bool): If True, includes the GMT score as a feature.
        include_nuclea_seq_score (bool): If True, includes the Nuclea-seq score as a feature.

    Returns:
        pd.DataFrame: The test dataset DataFrame with model predictions added as a column.
    """
    aligned_str = "aligned" if train_model_spec.aligned else "non_aligned"
    model = load_model_from_pkl(
        model_task=model_task, data_type=train_model_spec.data_type,
        include_distance_feature=train_model_spec.include_distance_feature, include_sequence_features=True,
        include_gmt_score=include_gmt_score, include_nuclea_seq_score=include_nuclea_seq_score,
        sample_weight=train_model_spec.sample_weight, model_type=train_model_spec.model_type, bulges=True,
        encoding_type=train_model_spec.encoding_type,
        path_prefix="{}/{}/{}/{}/".format(train_model_spec.model_version,
                                          "read_ts_{}".format(train_model_spec.read_threshold),
                                          pred_type_path, aligned_str),
        k_fold_number=k_fold_number, fold_index=fold_index)

    y_prediction = predict(
        test_dataset_df=test_dataset_df, model=model,
        include_distance_feature=train_model_spec.include_distance_feature, include_sequence_features=True,
        include_gmt_score=include_gmt_score, include_nuclea_seq_score=include_nuclea_seq_score,
        padding_type=padding_type, aligned=train_model_spec.aligned,
        bulges=True, encoding_type=train_model_spec.encoding_type,
        flat_encoding=train_model_spec.flat_encoding
        )
    if y_prediction.ndim == 2:
        y_prediction = y_prediction[:, 1]

    model_name = extract_model_name(
        model_type=train_model_spec.model_type, model_task=model_task,
        include_distance_feature=train_model_spec.include_distance_feature,
        include_sequence_features=True, include_gmt_score=include_gmt_score,
        include_nuclea_seq_score=include_nuclea_seq_score, sample_weight=train_model_spec.sample_weight,
        read_threshold=train_model_spec.read_threshold, encoding_type=train_model_spec.encoding_type,
        aligned=train_model_spec.aligned)
    test_dataset_df[model_name] = y_prediction

    return test_dataset_df


def predict(test_dataset_df, model,
            include_distance_feature=False, include_sequence_features=True,
            include_gmt_score=False, include_nuclea_seq_score=False,
            padding_type=Padding_type.NONE, aligned=True, bulges=False,
            encoding_type=Encoding_type.ONE_HOT, flat_encoding=True, seq_len=None):
    """
    Generates predictions.

    Args:
        test_dataset_df (pd.DataFrame): DataFrame containing off-targets and sgRNAs sequences,
                                        and potentially "DISTANCE" columns
        model (Model): The pre-trained model instance.
        include_gmt_score (bool): If True, includes the GMT score as a feature.
        include_nuclea_seq_score (bool): If True, includes the Nuclea-seq score as a feature.
        padding_type (Padding_type):  Type of padding for sequence alignment.
        aligned (bool): Indicates whether sequences are aligned.
        bulges (bool): Indicates whether bulges are considered.
        encoding_type (Encoding_type):  The type of encoding used.
        flat_encoding (bool): If True, uses flattened encoding.
        seq_len (int, optional): Sequences length.

    Returns:
        np.ndarray: Array of model predictions.
    """
    # build features
    start = time.time()

    test_dataset_df = build_sequence_features(
        test_dataset_df,
        include_distance_feature=include_distance_feature, include_sequence_features=include_sequence_features,
        include_gmt_score=include_gmt_score, include_nuclea_seq_score=include_nuclea_seq_score,
        bulges=bulges, padding_type=padding_type, aligned=aligned,
        encoding_type=encoding_type, flat_encoding=flat_encoding, seq_len=seq_len)
    end = time.time()
    print("************** features build time:", end - start, "**************")

    # Force call to the garbage collector
    gc.collect()

    return model.predict(test_dataset_df)
