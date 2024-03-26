"""
    This module contains the functions for training
"""
import gc
import random
import time
import json
from pathlib import Path

import numpy as np
from sklearn.utils import shuffle

from OT_deep_score_src.models_utilities import model_selection
from OT_deep_score_src.dataset_utilities import create_fold_sets, split_sg_rnas_into_folds
from OT_deep_score_src.data_processing_utilities import build_sequence_features, \
    build_sampleweight, transformer_generator, transform
from OT_deep_score_src.naming_utilities import extract_model_path
from OT_deep_score_src.models_inter import NN_model

from OT_deep_score_src.general_utilities import SEED, SG_RNA, SG_RNA_SEQ, OFF_TARGET, LABEL, READS, \
    DISTANCE, Model_task, Model_type, Data_type, Data_trans_type, Padding_type, Encoding_type


random.seed(SEED)


def data_preprocessing(
        dataset_df, trans_type, trans_all_fold, trans_only_positive):
    """
    Preprocesses data for model training.

    Args:
        dataset_df (pd.DataFrame): DataFrame containing off-targets, sgRNAs, labels, and reads.
        trans_type (Data_trans_type): The type of data transformation to apply.
        trans_all_fold (bool): If True, applies the transformation across all folds.
        trans_only_positive (bool): If True, applies the transformation only to positive examples.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    labels_df = dataset_df[[SG_RNA, SG_RNA_SEQ, OFF_TARGET, LABEL, READS]]
    labels_df.loc[dataset_df[LABEL] == 0, READS] = 0
    if trans_only_positive:
        labels_df = labels_df[labels_df[LABEL] == 1]

    if trans_all_fold:
        labels = labels_df[READS].values
        transformer = transformer_generator(labels, trans_type)
        labels_df[READS] = transform(labels, transformer)
    else:
        # perform the preprocessing on each sgRNA data individually
        for sg_rna in labels_df[SG_RNA].unique():
            sg_rna_df = labels_df[labels_df[SG_RNA] == sg_rna]
            sg_rna_labels = sg_rna_df[READS].values
            transformer = transformer_generator(sg_rna_labels, trans_type)
            labels_df.loc[labels_df[SG_RNA] == sg_rna, READS] = transform(sg_rna_labels, transformer)

    if trans_only_positive:
        dataset_df.loc[dataset_df[LABEL] == 1, READS] = labels_df[READS]
    else:
        dataset_df[READS] = labels_df[READS]

    return dataset_df


def k_fold_train(
        dataset_df, sg_rnas, model_task=Model_task.CLASSIFICATION_TASK,
        pretrained_models=None, data_type=Data_type.CHANGE_SEQ,
        predict_distance=False, k_fold_number=None, load_predefined_folds=False, sample_weight=True,
        model_type=Model_type.XGBOOST, model_parameters=None,
        fit_parameters=None, include_distance_feature=False,
        include_sequence_features=True, include_gmt_score=False,
        include_nuclea_seq_score=False, padding_type=Padding_type.NONE, aligned=True,
        trans_type=Data_trans_type.LOG1P, trans_all_fold=False,
        trans_only_positive=False, exclude_sg_rnas_without_positives=False, bulges=False,
        encoding_type=Encoding_type.ONE_HOT, flat_encoding=True, seq_len=None, gpu=True,
        val_size=None, skip_num_folds=0, save_model=False, path_prefix="",
        save_train_log=True, avoid_retrain=True, continue_train=False, testing=False):
    """
    Conducts k-fold cross-validation training.

    Args:
        dataset_df (pd.DataFrame): dataset for k-fold-cross-validation containing off-targets, sgRNAs, labels, etc.
        sg_rnas (list): List of sgRNA sequences.
        model_task (Model_task): Specifies whether the task is classification or regression.
        pretrained_models (list, optional): List of pre-trained models (one per fold).
                                            Used for transfer learning or fine-tuning.
        data_type (Data_type):  The dataset type.
        predict_distance (bool): If True, trains a model to predict edit distances.
        k_fold_number (int): Number of folds for cross-validation.
        load_predefined_folds (bool): If True, loads folds from a pre-defined file.
        sample_weight (bool): Whether to use sample weights during training.
        model_type (Model_type):  The type of model to train.
        model_parameters (dict, optional): Model-specific parameters.
        fit_parameters (dict, optional):  Parameters for the models "fit" method.
        include_distance_feature (bool): If True, includes the distance feature.
        include_sequence_features (bool): If True, includes sequence encoding features.
        include_gmt_score (bool): If True, includes the GMT score as a feature.
        include_nuclea_seq_score (bool): If True, includes the Nuclea-seq score as a feature.
        padding_type (Padding_type):  Type of padding for sequence alignment.
        aligned (bool): Indicates whether sequences are aligned.
        trans_type (Data_trans_type): Type of data transformation to apply.
        trans_all_fold (bool): If True, applies the transformation across all folds.
        trans_only_positive (bool): If True, applies the transformation only to positive examples.
        exclude_sg_rnas_without_positives (bool): If True, excludes sgRNAs without
                                                  positive examples from the training set.
        bulges (bool): Indicates whether bulges are considered.
        encoding_type (Encoding_type):  The type of encoding used.
        flat_encoding (bool): If True, uses flattened encoding.
        seq_len (int, optional): Sequences length.
        gpu (bool): If True, use GPU for training (if available).
        val_size (float, optional): Validation set size for neural networks.
        skip_num_folds (int): Number of initial folds to skip (useful for restarting interrupted training).
        save_model (bool):  If True, saves the trained model.
        path_prefix (str):  Prefix for model save path.
        save_train_log (bool): If True, saves training log (for neural networks).
        avoid_retrain (bool): If True, loads pre-trained model if available.
        continue_train (bool): If True, continues training on a pre-trained model.
        testing (bool): If True, limits training to a single sgRNA for testing purposes.

    Returns:
        list: A list of trained models (one for each fold).
    """

    # holds all the trained models
    models = []
    # in case we do not have k_fold, we train on all the dataset.
    sg_rna_folds_list = split_sg_rnas_into_folds(
        k_fold_number=k_fold_number, sg_rnas=sg_rnas, load_predefined_folds=load_predefined_folds, data_type=data_type)

    for i, sg_rna_fold in enumerate(sg_rna_folds_list[skip_num_folds:]):
        print("Train fold ", i + skip_num_folds)
        print("sgRNAs left for test for this fold:", sg_rna_fold)
        _, train_dataset_df = create_fold_sets(sg_rna_fold, sg_rnas, dataset_df,
                                               exclude_sg_rnas_without_positives)
        model = train(
            train_dataset_df=train_dataset_df, model_task=model_task,
            pretrained_model=None if pretrained_models is None else pretrained_models[i], data_type=data_type,
            predict_distance=predict_distance,
            sample_weight=sample_weight, model_type=model_type, model_parameters=model_parameters,
            fit_parameters=fit_parameters, include_distance_feature=include_distance_feature,
            include_sequence_features=include_sequence_features, include_gmt_score=include_gmt_score,
            include_nuclea_seq_score=include_nuclea_seq_score, padding_type=padding_type, aligned=aligned,
            trans_type=trans_type, trans_all_fold=trans_all_fold, trans_only_positive=trans_only_positive,
            bulges=bulges, encoding_type=encoding_type, flat_encoding=flat_encoding,
            seq_len=seq_len, gpu=gpu, val_size=val_size, save_model=save_model, path_prefix=path_prefix,
            save_train_log=save_train_log, avoid_retrain=avoid_retrain,
            k_fold_number=k_fold_number,
            fold_index=None if k_fold_number is None else i+skip_num_folds,
            continue_train=continue_train, testing=testing)

        models.append(model)

    return models


def train(
        train_dataset_df, model_task=Model_task.CLASSIFICATION_TASK,
        pretrained_model=None, data_type=Data_type.CHANGE_SEQ,
        predict_distance=False, sample_weight=True, model_type=Model_type.XGBOOST,
        model_parameters=None, fit_parameters=None,
        include_distance_feature=False, include_sequence_features=True,
        include_gmt_score=False, include_nuclea_seq_score=False,
        padding_type=Padding_type.NONE, aligned=True,
        trans_type=Data_trans_type.LOG1P, trans_all_fold=False,
        trans_only_positive=False, bulges=False, encoding_type=Encoding_type.ONE_HOT,
        flat_encoding=True, seq_len=None, gpu=True,
        val_size=None, save_model=False, path_prefix="", save_train_log=True,
        avoid_retrain=True, exclude_sg_rnas_without_positives=False,
        k_fold_number=None, fold_index=None, continue_train=False, testing=False):
    """
    Trains the prediction model.

    Args:
        train_dataset_df (pd.DataFrame): Training data containing off-targets, sgRNAs, labels, etc.
        model_task (Model_task): Specifies whether the task is classification or regression.
        pretrained_model (Model, optional):  Pre-trained model for transfer learning or fine-tuning.
        data_type (Data_type):  The dataset type.
        predict_distance (bool): If True, trains a model to predict edit distances.
        sample_weight (bool): Whether to use sample weights during training.
        model_type (Model_type):  The type of model to train.
        model_parameters (dict, optional): Model-specific parameters.
        fit_parameters (dict, optional):  Parameters for the models "fit" method.
        include_distance_feature (bool): If True, includes the distance feature.
        include_sequence_features (bool): If True, includes sequence encoding features.
        include_gmt_score (bool): If True, includes the GMT score as a feature.
        include_nuclea_seq_score (bool): If True, includes the Nuclea-seq score as a feature.
        padding_type (Padding_type):  Type of padding for sequence alignment.
        aligned (bool): Indicates whether sequences are aligned.
        trans_type (Data_trans_type): Type of data transformation to apply.
        trans_all_fold (bool): If True, applies the transformation across all folds.
        trans_only_positive (bool): If True, applies the transformation only to positive examples.
        bulges (bool): Indicates whether bulges are considered.
        encoding_type (Encoding_type):  The type of encoding used.
        flat_encoding (bool): If True, uses flattened encoding.
        seq_len (int, optional): Sequences length.
        gpu (bool): If True, use GPU for training (if available).
        val_size (float, optional): Validation set size for neural networks.
        save_model (bool):  If True, saves the trained model.
        path_prefix (str):  Prefix for model save path.
        save_train_log (bool): If True, saves training log (for neural networks).
        avoid_retrain (bool): If True, loads pre-trained model if available.
        exclude_sg_rnas_without_positives (bool): If True, excludes sgRNAs without
                                                  positive examples from the training set.
        k_fold_number (int or None): Number of folds for cross-validation.
        fold_index (int or None): Current fold index.
        continue_train (bool): If True, continues training on a pre-trained model.
        testing (bool): If True, limits training to a single sgRNA for testing purposes.

    Returns:
        Model: The trained model instance.
    """
    file_path_and_name = extract_model_path(
                model_task=model_task, data_type=data_type,
                include_distance_feature=include_distance_feature,
                include_sequence_features=include_sequence_features,
                include_gmt_score=include_gmt_score,
                include_nuclea_seq_score=include_nuclea_seq_score,
                trans_type=trans_type, trans_all_fold=trans_all_fold,
                trans_only_positive=trans_only_positive,
                exclude_sg_rnas_without_positives=exclude_sg_rnas_without_positives,
                path_prefix=path_prefix, model_type=model_type, encoding_type=encoding_type,
                bulges=bulges, sample_weight=sample_weight,
                k_fold_number=k_fold_number, fold_index=fold_index)

    if testing:
        test_sg_rna = train_dataset_df[SG_RNA].unique()[0]
        train_dataset_df = train_dataset_df[train_dataset_df[SG_RNA] == test_sg_rna]
        save_model, save_train_log = False, False

    if save_model or save_train_log:
        # TODO: think how you can automaticly save models with different folders
        # file_path_and_name = generate_next_folder_name(os.path.dirname(file_path_and_name)) + \
        #     os.path.basename(file_path_and_name)
        Path(file_path_and_name).parent.mkdir(parents=True, exist_ok=True)
        if fit_parameters is not None:
            with open(file_path_and_name + "_fit_parameters.json", "w") as f:
                json.dump(fit_parameters, f,
                          default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")
        if model_parameters is not None:
            with open(file_path_and_name + "_model_parameters.json", "w") as f:
                json.dump(model_parameters, f,
                          default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")

    # build features
    start = time.time()
    features_train = build_sequence_features(
        train_dataset_df,
        include_distance_feature=include_distance_feature,
        include_sequence_features=include_sequence_features,
        include_gmt_score=include_gmt_score, include_nuclea_seq_score=include_nuclea_seq_score,
        bulges=bulges, padding_type=padding_type, aligned=aligned,
        encoding_type=encoding_type, flat_encoding=flat_encoding, seq_len=seq_len)
    end = time.time()
    print("************** features build time:", end - start, "**************")

    # obtain regression labels
    if predict_distance:
        labels_train = train_dataset_df[DISTANCE].values
        # just in case we will use sample weight, then we will weight according the distance.
        class_train = labels_train.copy()
    else:
        # obtain classes
        class_train = train_dataset_df[LABEL].values
        if model_task == Model_task.REGRESSION_TASK:
            train_dataset_df = data_preprocessing(
                train_dataset_df, trans_type=trans_type, trans_all_fold=trans_all_fold,
                trans_only_positive=trans_only_positive)
            labels_train = train_dataset_df[READS].values
        else:
            labels_train = class_train

    indices = np.arange(len(labels_train))
    indices = shuffle(indices, random_state=SEED)
    class_train, labels_train = class_train[indices], labels_train[indices]
    if isinstance(features_train, list):
        features_train = [feature_item[indices] for feature_item in features_train]
        input_shape = [feature_item.shape[1:] for feature_item in features_train]
    else:
        features_train = features_train[indices]  # type: ignore
        input_shape = features_train.shape[1:]

    model, fit_kwargs = model_selection(
        model_task, model_type, model_parameters, gpu, None if continue_train else pretrained_model, val_size=val_size,
        input_shape=input_shape, file_path_and_name=file_path_and_name if save_train_log else None)
    if fit_parameters is not None:
        fit_kwargs.update(fit_parameters)
    start = time.time()
    # try to avoid retraining the model
    succeed_in_loading_model = False
    if avoid_retrain:
        try:
            model.load(file_path_and_name)
            succeed_in_loading_model = True
            print("model found, loading it instead of fitting it again. A new instace model will be saved")
        except Exception:
            print("model was not found, starting the fit")
    if not succeed_in_loading_model:
        if continue_train:
            if pretrained_model is None:
                raise ValueError("pretrained_model must be a Model when continue_train is True")
            model.model = pretrained_model
            if isinstance(model, NN_model):
                model.compile()
            else:
                raise ValueError("Cannot continue train for non NN-model")
        else:
            model.construct()
        model.fit(features_train, labels_train,
                  sample_weight=build_sampleweight(class_train) if sample_weight else None,
                  **fit_kwargs)
    end = time.time()
    print("************** training time:", end - start, "**************")

    if save_model:
        Path(file_path_and_name).parent.mkdir(parents=True, exist_ok=True)
        print(file_path_and_name)
        model.save_model_instance(file_path_and_name)

    # Force call to the garbage collector
    gc.collect()

    return model
