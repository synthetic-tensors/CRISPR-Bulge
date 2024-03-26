import gc

import pandas as pd

from OT_deep_score_src.dataset_utilities import load_dataset, load_order_sg_rnas, load_sg_rnas_list, \
    load_generated_dataset, split_to_train_and_test, load_dataset_from_file
from OT_deep_score_src.predict_utilities import k_fold_predict, load_and_predict, predict
from OT_deep_score_src.train_utilities import k_fold_train
from OT_deep_score_src.general_utilities import SG_RNA, Model_task, Data_type, Padding_type, Encoding_type

from OT_deep_score_src.models_inter import Model


class TrainModelSpec:
    """
    Defines the specifications for training a model.
    """

    def __init__(
            self, model_type, predict_distance, model_version, model_parameters,
            include_distance_feature, sample_weight, encoding_type, flat_encoding,
            read_threshold, aligned=True, tf_models_spec_list=None, fit_parameters=None,
            data_type=Data_type.CHANGE_SEQ, data_types_to_exclude=None, sg_rnas_to_exclude=None,
            continue_train=False, testing=False):
        """
        initilize TrainModelSpec instance

        Args:
        model_type (Model_type): The type of model architecture (e.g., C_1, C_2, C_3, XGBOOST, ...).
        predict_distance (bool): If True, trains a model to predict edit distances.
        model_version (str): Version identifier for the model.
        model_parameters (dict): Model-specific hyperparameters. Can be None.
        include_distance_feature (bool): If True, includes the distance feature as input.
        sample_weight (bool): If True, uses sample weights during training.
        encoding_type (Encoding_type): The encoding scheme for sequences.
        flat_encoding (bool): If True, uses a flattened encoding representation.
        read_threshold (int): Threshold for filtering off-target sites by read count.
        aligned (bool, optional): If True, sequences are aligned. Defaults to True.
        tf_models_spec_list (list[TrainModelSpec], optional): List of TrainModelSpec for transfer learning models.
        fit_parameters (dict, optional): Parameters for the model's fit method.
        data_type (Data_type, optional): The dataset type to train on. Defaults to Data_type.CHANGE_SEQ.
        data_types_to_exclude (list[Data_type], optional):  Dataset types to exclude sgRNAs from.
        sg_rnas_to_exclude (list[str], optional): Specific sgRNAs to exclude.
        continue_train (bool, optional): If True, continues training on a pre-existing model. Defaults to False.
        testing (bool, optional): If True, uses a small subset of data for testing purposes. Defaults to False.
        """
        self.model_type = model_type
        self.predict_distance = predict_distance
        self.model_version = model_version
        self.model_parameters = model_parameters.copy() if model_parameters is not None else model_parameters
        self.fit_parameters = fit_parameters.copy() if fit_parameters is not None else fit_parameters
        self.include_distance_feature = include_distance_feature
        self.sample_weight = sample_weight
        self.encoding_type = encoding_type
        self.flat_encoding = flat_encoding
        self.read_threshold = read_threshold
        self.aligned = aligned
        self.tf_models_spec_list = tf_models_spec_list
        self.data_type = data_type
        # sgRNAs to exclude
        self.sg_rnas_to_exclude = [] if data_types_to_exclude is None else [*set(
            [sg_rna for data_type_to_exclude in data_types_to_exclude for sg_rna in load_sg_rnas_list(
                data_type_to_exclude)])]
        self.sg_rnas_to_exclude = self.sg_rnas_to_exclude if sg_rnas_to_exclude is None else \
            [*set(self.sg_rnas_to_exclude + sg_rnas_to_exclude)]
        self.continue_train = continue_train
        # This is will take only small subset of the data for training,
        # used for testing the training or just loading the model
        self.testing = testing


def train_main(
        train_model_spec_list: list[TrainModelSpec],
        model_tasks=(Model_task.CLASSIFICATION_TASK, Model_task.REGRESSION_TASK),
        generated_off=False, padding_type=Padding_type.GAP, k_fold_number=None,
        load_predefined_folds=False, pretrained_models=None,
        include_gmt_score=False, include_nuclea_seq_score=False):
    """
    Manages the training process for multiple model specifications.

    Args:
        train_model_spec_list (list[TrainModelSpec]): A list of `TrainModelSpec` objects,
                                                      each defining a model configuration.
        model_tasks (tuple of Model_task, optional): Specifies model tasks
                                                     (classification, regression, or both). Defaults to both.
        generated_off (bool, optional): If True, trains on generated off-target sites. Defaults to False.
        padding_type (Padding_type, optional): Type of padding for sequence representation.
                                               Defaults to Padding_type.GAP.
        k_fold_number (int, optional): Number of folds for k-fold cross-validation.
                                       If None or 1, no cross-validation is performed.
        load_predefined_folds (bool, optional): If True, loads predefined sgRNAs for folds of the cross-validation.
                                                Defaults to False.
        pretrained_models (list, optional): List of pre-trained models to use for transfer learning. Defaults to None.
        include_gmt_score (bool, optional): If True, includes the GMT score as an input feature. Defaults to False.
        include_nuclea_seq_score (bool, optional): If True, includes the NucleaSeq score as an input feature.
                                                   Defaults to False.
    """
    for train_model_spec in train_model_spec_list:
        # Force call to the garbage collector
        gc.collect()

        aligned_str = "aligned" if train_model_spec.aligned else "non_aligned"
        if generated_off and train_model_spec.aligned:
            raise ValueError("generated off-targets are not aligned")
        # load sgRNAs and data
        # TODO: consider saving the datasets instead of loading them again and again
        sg_rnas = load_order_sg_rnas(train_model_spec.data_type)
        if generated_off:
            generated_dataset_df = load_generated_dataset(sg_rnas)
            if k_fold_number is None:
                generated_dataset_df, _ = split_to_train_and_test(generated_dataset_df, sg_rnas)
        else:
            generated_dataset_df = None
        regular_dataset_df = load_dataset(train_model_spec.data_type, sg_rnas,
                                          read_threshold=train_model_spec.read_threshold,
                                          sg_rnas_to_exclude=train_model_spec.sg_rnas_to_exclude)
        if k_fold_number is None:
            regular_dataset_df, _ = split_to_train_and_test(regular_dataset_df, sg_rnas)

        for model_task in model_tasks:
            print("********************************************************************")
            print(train_model_spec.model_type)
            print(model_task)
            print("********************************************************************")
            pred_type_path = "distance_models" if train_model_spec.predict_distance else "cleavage_models"
            if generated_off and train_model_spec.predict_distance:
                train_dataset_df = generated_dataset_df
            else:
                train_dataset_df = regular_dataset_df

            # if model is exists in files, then the model will be loaded
            models = k_fold_train(
                train_dataset_df,
                sg_rnas,
                model_task=model_task,
                pretrained_models=pretrained_models,
                data_type=train_model_spec.data_type,
                predict_distance=train_model_spec.predict_distance,
                k_fold_number=k_fold_number,
                load_predefined_folds=load_predefined_folds,
                sample_weight=train_model_spec.sample_weight,
                model_type=train_model_spec.model_type,
                model_parameters=train_model_spec.model_parameters,
                fit_parameters=train_model_spec.fit_parameters,
                include_distance_feature=train_model_spec.include_distance_feature,
                include_sequence_features=True,
                include_gmt_score=include_gmt_score,
                include_nuclea_seq_score=include_nuclea_seq_score,
                padding_type=padding_type,
                aligned=train_model_spec.aligned,
                bulges=True,
                encoding_type=train_model_spec.encoding_type,
                flat_encoding=train_model_spec.flat_encoding,
                gpu=True,
                val_size=0.1,
                save_model=True,
                path_prefix="{}/{}/{}/{}/".format(
                    train_model_spec.model_version,
                    "read_ts_{}".format(train_model_spec.read_threshold),
                    pred_type_path, aligned_str),
                continue_train=train_model_spec.continue_train,
                testing=train_model_spec.testing)

            if train_model_spec.tf_models_spec_list is not None:
                train_main(
                    train_model_spec.tf_models_spec_list,
                    model_tasks=(model_task,),
                    generated_off=generated_off, padding_type=padding_type,
                    k_fold_number=k_fold_number,
                    load_predefined_folds=load_predefined_folds,
                    pretrained_models=models, include_gmt_score=include_gmt_score,
                    include_nuclea_seq_score=include_nuclea_seq_score)


def predict_main(
        train_model_spec_list: list[TrainModelSpec], read_threshold_test,
        model_tasks=(Model_task.REGRESSION_TASK, Model_task.CLASSIFICATION_TASK),
        prediction_table_path_prefix="", generated_off=False, padding_type=Padding_type.GAP,
        test_data_type=Data_type.CHANGE_SEQ, k_fold_number=None,
        load_predefined_folds=False, include_gmt_score=False, include_nuclea_seq_score=False,
        save_predictions=True):
    """
    Generates predictions using trained models and optionally saves results.

    Args:
        train_model_spec_list (list[TrainModelSpec]): A list of `TrainModelSpec` objects for trained models.
        read_threshold_test (int): Read count threshold for filtering off-target sites in the test dataset.
        model_tasks (tuple of Model_task, optional): Specifies tasks (classification, regression, or both).
                                                     Defaults to both.
        prediction_table_path_prefix (str, optional): Prefix for the path to save the predictions table. Defaults to "".
        generated_off (bool, optional): If True, uses generated off-targets for prediction. Defaults to False.
                                        Currently not doing much.
        padding_type (Padding_type, optional): Padding type for sequence representation. Defaults to Padding_type.GAP.
        test_data_type (Data_type, optional): Type of the test dataset. Defaults to Data_type.CHANGE_SEQ.
        k_fold_number (int, optional): Number of folds for k-fold cross-validation. If None or 1, no cross-validation.
        load_predefined_folds (bool, optional): If True, loads predefined sgRNAs for folds of the cross-validation.
                                                Defaults to False.
        include_gmt_score (bool, optional): If True, includes the GMT score as a feature. Defaults to False.
        include_nuclea_seq_score (bool, optional): If True, includes the NucleaSeq score as a feature.
                                                   Defaults to False.
        save_predictions (bool):  If True, saves the prediction results to a CSV file. Defaults to True.

    Returns:
        pd.DataFrame: The DataFrame containing test data and predictions.
    """
    test_sg_rnas = None
    if test_data_type in (
            Data_type.CHANGE_SEQ, Data_type.GUIDE_SEQ, Data_type.FULL_GUIDE_SEQ, Data_type.NEW_GUIDE_SEQ):
        # load sgRNAs
        test_sg_rnas = load_order_sg_rnas(test_data_type)
        test_dataset_df = load_dataset(test_data_type, test_sg_rnas, read_threshold=read_threshold_test)
        if k_fold_number is None:
            _, test_dataset_df = split_to_train_and_test(test_dataset_df, test_sg_rnas)
    elif test_data_type in (
            Data_type.TSAI_2015_GUIDE_SEQ, Data_type.CHEN_2017_GUIDE_SEQ,
            Data_type.LISTGARTEN_2018_GUIDE_SEQ, Data_type.REFINED_TURE_OT):
        test_dataset_df = load_dataset_from_file(test_data_type)
        test_sg_rnas = list(test_dataset_df[SG_RNA].unique())
    else:
        raise ValueError("test_data_type is not one of the supported data types")

    for train_model_spec in train_model_spec_list:
        pred_type_path = "distance_models" if train_model_spec.predict_distance else "cleavage_models"
        if generated_off:
            pred_type_path = "generated_off_targets/" + pred_type_path
        print("**********************************")
        print(train_model_spec.model_type)
        print("**********************************")
        for model_task in model_tasks:
            if isinstance(k_fold_number, int):
                test_dataset_df = k_fold_predict(
                    test_dataset_df=test_dataset_df, test_data_type=test_data_type, test_sg_rnas=test_sg_rnas,
                    model_task=model_task, train_model_spec=train_model_spec,
                    padding_type=padding_type, pred_type_path=pred_type_path, k_fold_number=k_fold_number,
                    load_predefined_folds=load_predefined_folds,
                    include_gmt_score=include_gmt_score, include_nuclea_seq_score=include_nuclea_seq_score)
            else:
                test_dataset_df = load_and_predict(
                    test_dataset_df, model_task, train_model_spec, padding_type,
                    pred_type_path, k_fold_number=k_fold_number, fold_index=None,
                    include_gmt_score=include_gmt_score, include_nuclea_seq_score=include_nuclea_seq_score)
    if save_predictions:
        test_dataset_df.to_csv(prediction_table_path_prefix + ".csv", index=False)

    return test_dataset_df


def ensemble_predict(ensemble_components_file_path_and_name_list, dataset_df):
    """
    Generates predictions using an ensemble of pre-trained models and saves the results.

    Args:
        ensemble_components_file_path_and_name_list (list): A list of file paths and names
                                                            of the pre-trained ensemble component models.
        dataset_df (pd.DataFrame or str):  Either a DataFrame containing the test data or
                                               a file path to a CSV file containing the test data.

    Returns:
        pd.DataFrame: The DataFrame containing test data with ensemble predictions.
    """
    if isinstance(dataset_df, str):
        dataset_df = pd.read_csv(dataset_df)

    num_of_esemble_components = len(ensemble_components_file_path_and_name_list)
    for i, file_path_and_name in enumerate(ensemble_components_file_path_and_name_list):
        model = Model.load_model_instance(file_path_and_name)

        y_prediction = predict(
            test_dataset_df=dataset_df, model=model,
            include_distance_feature=False, include_sequence_features=True,
            include_gmt_score=False, include_nuclea_seq_score=False,
            padding_type=Padding_type.GAP, aligned=True,
            bulges=True, encoding_type=Encoding_type.ONE_HOT,
            flat_encoding=False  # Flat encoding for the NN models is False
            )

        if num_of_esemble_components > 1:
            dataset_df["pred_ensemble_component_{}".format(i)] = y_prediction
        else:
            dataset_df["pred"] = y_prediction

    if num_of_esemble_components > 1:
        dataset_df["pred_averege_ensemble"] = dataset_df[
            ["pred_ensemble_component_{}".format(i) for i in range(num_of_esemble_components)]].mean(axis=1)

    # save the predictions
    dataset_df.to_csv("predictions.csv", index=False)

    return dataset_df
