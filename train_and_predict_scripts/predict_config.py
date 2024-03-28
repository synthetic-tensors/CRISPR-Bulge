from train_and_predict_scripts.utilities import TrainModelSpec, predict_main
import random

from OT_deep_score_src.general_utilities import SEED, Model_type, Data_type, Encoding_type

random.seed(SEED)


def predict_settings(version, setting_number):
    """
    Determines appropriate prediction settings based on the provided version and setting number.

    Args:
        version (str): Version identifier for the model.
        setting_number (int): Number indicating the specific prediction scenario to configure.

    Returns:
        tuple: A tuple containing:
            * settings (dict): A dictionary containing prediction parameters.
            * version (str): The potentially updated model version string.
    """
    settings = {}
    if version.startswith("4_revision"):
        settings["model_parameters"] = {"batch_size": 512}
    elif version.startswith("5_revision"):
        settings["model_parameters"] = {"batch_size": 512, "learning_rate": 0.0005}
    else:
        raise ValueError("Not supporting this model version")

    if setting_number == 1:
        # for the th vs no-th ans sw vs no-sw
        settings["read_threshold_test"] = 100
        settings["read_threshold_train_options"] = (0, 100)
        settings["sample_weight_options"] = (False, True)
        settings["train_data_type"] = Data_type.CHANGE_SEQ
        settings["test_data_type"] = Data_type.CHANGE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = ""
        settings["data_types_to_exclude"] = None
        settings["k_fold_number"] = 10
        settings["encoding_type"] = Encoding_type.ONE_HOT
    elif setting_number == 2:
        # for prediction on guide in cross validation
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.CHANGE_SEQ
        settings["test_data_type"] = Data_type.GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_train_on_{}".format(Data_type.CHANGE_SEQ)
        settings["data_types_to_exclude"] = None
        settings["k_fold_number"] = 10
        settings["encoding_type"] = Encoding_type.ONE_HOT
    elif setting_number == 3:
        # for prediction on guide in cross validation
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.GUIDE_SEQ
        settings["test_data_type"] = Data_type.GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = ""
        settings["data_types_to_exclude"] = None
        settings["k_fold_number"] = 10
        settings["encoding_type"] = Encoding_type.ONE_HOT
    elif setting_number == 4:
        # for prediction on guide in cross validation
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.GUIDE_SEQ
        settings["test_data_type"] = Data_type.GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = ""
        settings["data_types_to_exclude"] = None
        settings["k_fold_number"] = 10
        settings["encoding_type"] = Encoding_type.ONE_HOT

        version = "{}_continue_from_change_seq".format(version)
    elif setting_number == 5:
        # for prediction on the new 20 GUIDE-seq data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.FULL_GUIDE_SEQ
        settings["test_data_type"] = Data_type.NEW_GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = ""
        settings["data_types_to_exclude"] = [Data_type.NEW_GUIDE_SEQ]
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT

        version = "{}_exclude_{}".format(version, settings["data_types_to_exclude"][0])
        version = "{}_continue_from_change_seq".format(version)
    elif setting_number == 6:
        # for prediction on the new 20 GUIDE-seq data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.FULL_GUIDE_SEQ
        settings["test_data_type"] = Data_type.NEW_GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = ""
        settings["data_types_to_exclude"] = [Data_type.NEW_GUIDE_SEQ]
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT

        version = "{}_exclude_{}".format(version, settings["data_types_to_exclude"][0])
    elif setting_number == 7:
        # for prediction on the new 20 GUIDE-seq data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.CHANGE_SEQ
        settings["test_data_type"] = Data_type.NEW_GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_train_on_{}".format(Data_type.CHANGE_SEQ)
        settings["data_types_to_exclude"] = [Data_type.NEW_GUIDE_SEQ]
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT

        version = "{}_exclude_{}".format(version, settings["data_types_to_exclude"][0])
    elif setting_number == 8:
        # for prediction RHAMP-seq data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.FULL_GUIDE_SEQ
        settings["test_data_type"] = Data_type.REFINED_TURE_OT
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_exclude_{}".format(Data_type.RHAMP_SEQ)
        settings["data_types_to_exclude"] = [Data_type.RHAMP_SEQ]
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT

        version = "{}_exclude_{}".format(version, settings["data_types_to_exclude"][0])
        version = "{}_continue_from_change_seq".format(version)
    elif setting_number == 9:
        # for prediction RHAMP-seq data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.CHANGE_SEQ
        settings["test_data_type"] = Data_type.REFINED_TURE_OT
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_train_on_{}_exclude_{}".format(Data_type.CHANGE_SEQ, Data_type.RHAMP_SEQ)
        settings["data_types_to_exclude"] = [Data_type.RHAMP_SEQ]
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT

        version = "{}_exclude_{}".format(version, settings["data_types_to_exclude"][0])
    elif setting_number == 10:
        # for prediction CHEN_2017_GUIDE_SEQ data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.FULL_GUIDE_SEQ
        settings["test_data_type"] = Data_type.CHEN_2017_GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_exclude_{}".format(Data_type.RHAMP_SEQ)
        settings["data_types_to_exclude"] = [Data_type.RHAMP_SEQ]
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT

        version = "{}_exclude_{}".format(version, settings["data_types_to_exclude"][0])
        version = "{}_continue_from_change_seq".format(version)
    elif setting_number == 11:
        # for prediction CHEN_2017_GUIDE_SEQ data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.CHANGE_SEQ
        settings["test_data_type"] = Data_type.CHEN_2017_GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_train_on_{}_exclude_{}".format(Data_type.CHANGE_SEQ, Data_type.RHAMP_SEQ)
        settings["data_types_to_exclude"] = [Data_type.RHAMP_SEQ]
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT

        version = "{}_exclude_{}".format(version, settings["data_types_to_exclude"][0])
    elif setting_number == 12:
        # for prediction LISTGARTEN_2018_GUIDE_SEQ data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.FULL_GUIDE_SEQ
        settings["test_data_type"] = Data_type.LISTGARTEN_2018_GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_exclude_{}".format(Data_type.RHAMP_SEQ)
        settings["data_types_to_exclude"] = [Data_type.RHAMP_SEQ]
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT

        version = "{}_exclude_{}".format(version, settings["data_types_to_exclude"][0])
        version = "{}_continue_from_change_seq".format(version)
    elif setting_number == 13:
        # for prediction LISTGARTEN_2018_GUIDE_SEQ data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.CHANGE_SEQ
        settings["test_data_type"] = Data_type.LISTGARTEN_2018_GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_train_on_{}_exclude_{}".format(Data_type.CHANGE_SEQ, Data_type.RHAMP_SEQ)
        settings["data_types_to_exclude"] = [Data_type.RHAMP_SEQ]
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT

        version = "{}_exclude_{}".format(version, settings["data_types_to_exclude"][0])
    elif setting_number == 14:
        # for prediction TSAI_2015_GUIDE_SEQ data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.FULL_GUIDE_SEQ
        settings["test_data_type"] = Data_type.TSAI_2015_GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_exclude_{}".format(Data_type.RHAMP_SEQ)
        settings["data_types_to_exclude"] = [Data_type.RHAMP_SEQ]
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT

        version = "{}_exclude_{}".format(version, settings["data_types_to_exclude"][0])
        version = "{}_continue_from_change_seq".format(version)
    elif setting_number == 15:
        # for prediction TSAI_2015_GUIDE_SEQ data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.CHANGE_SEQ
        settings["test_data_type"] = Data_type.TSAI_2015_GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_train_on_{}_exclude_{}".format(Data_type.CHANGE_SEQ, Data_type.RHAMP_SEQ)
        settings["data_types_to_exclude"] = [Data_type.RHAMP_SEQ]
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT

        version = "{}_exclude_{}".format(version, settings["data_types_to_exclude"][0])
    elif setting_number == 16:
        # for the th vs no-th ans sw vs no-sw with CRISPR-Net Encoding
        settings["read_threshold_test"] = 100
        settings["read_threshold_train_options"] = (0, 100)
        settings["sample_weight_options"] = (False, True)
        settings["train_data_type"] = Data_type.CHANGE_SEQ
        settings["test_data_type"] = Data_type.CHANGE_SEQ
        settings["output_file_prefix"] = "predictions_{}_".format(Encoding_type.CRISPR_NET)
        settings["output_file_suffix"] = ""
        settings["data_types_to_exclude"] = None
        settings["k_fold_number"] = 10
        settings["encoding_type"] = Encoding_type.CRISPR_NET
    elif setting_number == 17:
        # for prediction RHAMP-seq data with CRISPR-Net training data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.CRISPR_NET
        settings["test_data_type"] = Data_type.REFINED_TURE_OT
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_trained_on_{}".format(Data_type.CRISPR_NET)
        settings["data_types_to_exclude"] = None
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT
    elif setting_number == 18:
        # for prediction CHEN_2017_GUIDE_SEQ data with CRISPR-Net training data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.CRISPR_NET
        settings["test_data_type"] = Data_type.CHEN_2017_GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_trained_on_{}".format(Data_type.CRISPR_NET)
        settings["data_types_to_exclude"] = None
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT
    elif setting_number == 19:
        # for prediction LISTGARTEN_2018_GUIDE_SEQ data with CRISPR-Net training data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.CRISPR_NET
        settings["test_data_type"] = Data_type.LISTGARTEN_2018_GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_trained_on_{}".format(Data_type.CRISPR_NET)
        settings["data_types_to_exclude"] = None
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT
    elif setting_number == 20:
        # for prediction TSAI_2015_GUIDE_SEQ data with CRISPR-Net training data
        settings["read_threshold_test"] = 0
        settings["read_threshold_train_options"] = (0,)
        settings["sample_weight_options"] = (False,)
        settings["train_data_type"] = Data_type.CRISPR_NET
        settings["test_data_type"] = Data_type.TSAI_2015_GUIDE_SEQ
        settings["output_file_prefix"] = "predictions_"
        settings["output_file_suffix"] = "_trained_on_{}".format(Data_type.CRISPR_NET)
        settings["data_types_to_exclude"] = None
        settings["k_fold_number"] = 1
        settings["encoding_type"] = Encoding_type.ONE_HOT
    else:
        raise ValueError("Not supporting this prediction settings")
    return settings, version


def create_model_spec_list(
        version, settings, include_distance_feature, encoding_type,
        aligned, setting_number, model_types):
    """
    Generates a list of TrainModelSpec objects based on prediction settings.

    Args:
        version (str): Model version identifier.
        settings (dict): Dictionary containing prediction parameters.
        include_distance_feature (bool): Whether to include the edit distance feature in model input.
        encoding_type (Encoding_type): Type of encoding to use for sequences.
        aligned (bool): Indicates whether the sequences are aligned.
        setting_number (int): Number specifying the prediction scenario.
        model_types (tuple of Model_type or None): If specified, predict with model types.
                                                   Otherwise, predicts with all supported types.


    Returns:
        list: A list of TrainModelSpec objects, each representing a model configuration.
    """
    if model_types is None:
        model_types = [Model_type.C_1, Model_type.C_2, Model_type.C_3]
    train_model_spec_list = []
    for read_threshold_train in settings["read_threshold_train_options"]:
        for sample_weight in settings["sample_weight_options"]:
            train_model_spec_list_i = [
                TrainModelSpec(
                    model_type=model_type, predict_distance=False, model_version=version,
                    model_parameters=settings["model_parameters"],
                    include_distance_feature=include_distance_feature, sample_weight=sample_weight,
                    encoding_type=encoding_type, flat_encoding=False,
                    read_threshold=read_threshold_train, aligned=aligned,
                    data_type=settings["train_data_type"],
                    data_types_to_exclude=settings["data_types_to_exclude"]) for model_type in model_types]
            if setting_number == 1 and Model_type.XGBOOST in model_types:
                train_model_spec_list.append(
                    TrainModelSpec(
                        model_type=Model_type.XGBOOST, predict_distance=False, model_version="4_revision",
                        model_parameters=None, include_distance_feature=include_distance_feature,
                        sample_weight=sample_weight, encoding_type=encoding_type,
                        flat_encoding=True, read_threshold=read_threshold_train,
                        aligned=aligned, data_type=settings["train_data_type"],
                        data_types_to_exclude=settings["data_types_to_exclude"])
                )
            elif setting_number == 16 and Model_type.XGBOOST in model_types:
                train_model_spec_list.append(
                    TrainModelSpec(
                        model_type=Model_type.XGBOOST, predict_distance=False, model_version=version,
                        model_parameters=None, include_distance_feature=include_distance_feature,
                        sample_weight=sample_weight, encoding_type=encoding_type,
                        flat_encoding=True, read_threshold=read_threshold_train,
                        aligned=aligned, data_type=settings["train_data_type"],
                        data_types_to_exclude=settings["data_types_to_exclude"])
                )
            train_model_spec_list.extend(train_model_spec_list_i)

    return train_model_spec_list


def main(version, setting_number, model_types=None):
    """
    Coordinates the prediction workflow.

    1. Determines prediction settings.
    2. Creates a list of model specifications.
    3. Executes the prediction process using "predict_main".

    Args:
        version (str): Model version identifier.
        setting_number (int): Number specifying the prediction scenario.
        model_types (tuple of Model_type or None): If specified, predict with model types.
                                                   Otherwise, predicts with all supported types.

    Returns:
        pd.DataFrame: The DataFrame containing the test data and predictions.
    """
    settings, version = predict_settings(version, setting_number)
    print("predictions settings:", version, settings, sep="\n")

    aligned = True
    include_distance_feature = False
    encoding_type = settings["encoding_type"]

    train_model_spec_list = create_model_spec_list(
        version, settings, include_distance_feature, encoding_type, aligned,
        setting_number, model_types)

    test_dataset_df = predict_main(
        train_model_spec_list,
        prediction_table_path_prefix="{}{}_cleavage_v{}{}{}".format(
            settings["output_file_prefix"], settings["test_data_type"],
            version, "_folds_{}".format(settings["k_fold_number"]), settings["output_file_suffix"]),
        k_fold_number=settings["k_fold_number"], load_predefined_folds=True,
        read_threshold_test=settings["read_threshold_test"], test_data_type=settings["test_data_type"])

    return test_dataset_df


if __name__ == "__main__":
    main(version="5_revision", setting_number=1)
