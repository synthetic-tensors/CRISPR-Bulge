import gc
import concurrent.futures
import random
import pandas as pd
import numpy as np
from OT_deep_score_src.models_utilities import load_order_sg_rnas, create_fold_sets, load_sg_rnas_list
from OT_deep_score_src.train_utilities import load_model_from_pkl, predict, k_fold_train
from OT_deep_score_src.general_utilities import DATASETS_PATH, SEED, SG_RNA, SG_RNA_SEQ, OFF_TARGET, READS, LABEL, \
    Model_task, Data_type, Padding_type, DEFAULT_READ_THRESHOLD


random.seed(SEED)


class TrainModelSpec:
    def __init__(
            self, model_type, predict_distance, model_version, model_parameters,
            include_distance_feature, sample_weight, fixed_size_encoding, flat_encoding,
            tf_models_spec_list=None, fit_parameters=None,
            data_type=Data_type.CHANGE_SEQ, data_types_to_exclude=None, sg_rnas_to_exclude=None,
            continue_train=False, testing=False):
        self.model_type = model_type
        self.predict_distance = predict_distance
        self.model_version = model_version
        self.model_parameters = model_parameters.copy() if model_parameters is not None else model_parameters
        self.fit_parameters = fit_parameters.copy() if fit_parameters is not None else fit_parameters
        self.include_distance_feature = include_distance_feature
        self.sample_weight = sample_weight
        self.fixed_size_encoding = fixed_size_encoding
        self.flat_encoding = flat_encoding
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
        aligned, train_model_spec_list: list[TrainModelSpec],
        model_tasks=(Model_task.CLASSIFICATION_TASK, Model_task.REGRESSION_TASK),
        generated_off=False, padding_type=Padding_type.GAP,
        read_threshold=DEFAULT_READ_THRESHOLD, k_fold_numder=None, pretrained_models=None,
        include_gmt_score=False, include_nuclea_seq_score=False):
    aligned_str = "aligned" if aligned else "non_aligned"

    if generated_off and aligned:
        raise ValueError("generated off-targets are not aligned")
    for train_model_spec in train_model_spec_list:
        # Force call to the garbage collector
        gc.collect()

        # load sgRNAs and data
        # TODO: consider saving the datasets instead of loading them again and again
        sg_rnas = load_order_sg_rnas(train_model_spec.data_type)
        if generated_off:
            generated_dataset_df = load_generated_dataset(sg_rnas)
            if k_fold_numder is None:
                generated_dataset_df, _ = split_to_train_and_test(generated_dataset_df, sg_rnas)
        else:
            generated_dataset_df = None
        regular_dataset_df = load_dataset(train_model_spec.data_type, sg_rnas, read_threshold=read_threshold)
        if k_fold_numder is None:
            regular_dataset_df, _ = split_to_train_and_test(regular_dataset_df, sg_rnas)

        for model_task in model_tasks:
            print("********************************************************************")
            print("********************************************************************")
            print(train_model_spec.model_type)
            print(model_task)
            print("********************************************************************")
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
                k_fold_number=k_fold_numder,
                sample_weight=train_model_spec.sample_weight,
                model_type=train_model_spec.model_type,
                model_parameters=train_model_spec.model_parameters,
                fit_parameters=train_model_spec.fit_parameters,
                include_distance_feature=train_model_spec.include_distance_feature,
                include_sequence_features=True,
                include_gmt_score=include_gmt_score,
                include_nuclea_seq_score=include_nuclea_seq_score,
                padding_type=padding_type,
                aligned=aligned,
                bulges=True,
                fixed_size_encoding=train_model_spec.fixed_size_encoding,
                flat_encoding=train_model_spec.flat_encoding,
                gpu=True,
                val_size=0.1,
                save_model=True,
                path_prefix="{}/{}/{}/{}/".format("read_ts_{}".format(
                    read_threshold), pred_type_path, aligned_str, train_model_spec.model_version),
                continue_train=train_model_spec.continue_train,
                testing=train_model_spec.testing)

            if train_model_spec.tf_models_spec_list is not None:
                train_main(
                    aligned, train_model_spec.tf_models_spec_list,
                    model_tasks=(model_task,),
                    generated_off=generated_off, padding_type=padding_type,
                    read_threshold=read_threshold, k_fold_numder=k_fold_numder, pretrained_models=models,
                    include_gmt_score=include_gmt_score, include_nuclea_seq_score=include_nuclea_seq_score)


def k_fold_predict(
        dataset_df, sg_rnas, model_task, train_model_spec,
        aligned, aligned_str, padding_type, pred_type_path, read_threshold_train, k_fold_number,
        include_gmt_score, include_nuclea_seq_score):
    if sg_rnas is None:
        raise ValueError("K-fold cannot accept sgRNAs list that is None.")
    if k_fold_number is None:
        raise ValueError("K-fold test cannot accept k_fold_number is None or only 1.")
    # in case we don't have k_fold, we take the entire dataset.
    sg_rna_folds_list = np.array_split(sg_rnas, k_fold_number)

    test_fold_datasets = []
    for fold_index, sg_rna_fold in enumerate(sg_rna_folds_list):
        print("test fold ", fold_index)
        test_fold_dataset_df, _ = create_fold_sets(
            sg_rna_fold, sg_rnas, dataset_df, exclude_sg_rnas_without_positives=False)

        test_fold_dataset_df = load_and_predict(
            test_fold_dataset_df, model_task, train_model_spec, aligned, aligned_str, padding_type, pred_type_path,
            read_threshold_train, k_fold_number, fold_index, include_gmt_score, include_nuclea_seq_score)
        test_fold_dataset_df["fold"] = fold_index
        test_fold_datasets.append(test_fold_dataset_df)

    return pd.concat(test_fold_datasets, ignore_index=True)


def load_and_predict(
        test_dataset_df, model_task, train_model_spec,
        aligned, aligned_str, padding_type, pred_type_path,
        read_threshold_train, k_fold_number, fold_index,
        include_gmt_score, include_nuclea_seq_score):
    model = load_model_from_pkl(
        model_task=model_task, data_type=train_model_spec.data_type,
        include_distance_feature=train_model_spec.include_distance_feature, include_sequence_features=True,
        include_gmt_score=include_gmt_score, include_nuclea_seq_score=include_nuclea_seq_score,
        sample_weight=train_model_spec.sample_weight, model_type=train_model_spec.model_type, bulges=True,
        fixed_size_encoding=train_model_spec.fixed_size_encoding,
        path_prefix="{}/{}/{}/{}/".format(
            "read_ts_{}".format(read_threshold_train), pred_type_path, aligned_str, train_model_spec.model_version),
        k_fold_number=k_fold_number, fold_index=fold_index)

    y_prediction = predict(
        test_dataset_df=test_dataset_df, model=model,
        include_distance_feature=train_model_spec.include_distance_feature, include_sequence_features=True,
        include_gmt_score=include_gmt_score, include_nuclea_seq_score=include_nuclea_seq_score,
        padding_type=padding_type, aligned=aligned,
        bulges=True, fixed_size_encoding=train_model_spec.fixed_size_encoding,
        flat_encoding=train_model_spec.flat_encoding
        )
    if y_prediction.ndim == 2:
        y_prediction = y_prediction[:, 1]
    test_dataset_df[
        "{}-{}-{}{}".format(
            "reg" if model_task == Model_task.REGRESSION_TASK else "class",
            "distance" if train_model_spec.predict_distance else "cleavage",
            train_model_spec.model_type,
            "-dist" if train_model_spec.include_distance_feature else "")] = y_prediction

    return test_dataset_df


def predict_main(
        aligned, train_model_spec_list: list[TrainModelSpec],
        model_tasks=(Model_task.REGRESSION_TASK, Model_task.CLASSIFICATION_TASK),
        prediction_table_path_prefix="", generated_off=False, padding_type=Padding_type.GAP,
        test_data_type=Data_type.CHANGE_SEQ, read_threshold_train=DEFAULT_READ_THRESHOLD,
        read_threshold_test=DEFAULT_READ_THRESHOLD, k_fold_number=None,
        include_gmt_score=False, include_nuclea_seq_score=False):
    aligned_str = "aligned" if aligned else "non_aligned"
    test_sg_rnas = None
    if test_data_type in (
            Data_type.CHANGE_SEQ, Data_type.GUIDE_SEQ, Data_type.FULL_GUIDE_SEQ, Data_type.NEW_GUIDE_SEQ):
        # load sgRNAs
        test_sg_rnas = load_order_sg_rnas(test_data_type)
        test_dataset_df = load_dataset(test_data_type, test_sg_rnas, read_threshold=read_threshold_test)
        if k_fold_number is None:
            _, test_dataset_df = split_to_train_and_test(test_dataset_df, test_sg_rnas)
    elif test_data_type == Data_type.TRUE_OT:
        test_dataset_df = load_true_ot_dataset()
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
                    test_dataset_df, test_sg_rnas, model_task, train_model_spec,
                    aligned, aligned_str, padding_type, pred_type_path, read_threshold_train, k_fold_number,
                    include_gmt_score, include_nuclea_seq_score)
            else:
                test_dataset_df = load_and_predict(
                    test_dataset_df, model_task, train_model_spec, aligned, aligned_str, padding_type,
                    pred_type_path, read_threshold_train, k_fold_number=k_fold_number, fold_index=None,
                    include_gmt_score=include_gmt_score, include_nuclea_seq_score=include_nuclea_seq_score)

    test_dataset_df.to_csv(prediction_table_path_prefix + "predictions_{}.csv".format(aligned_str), index=False)


def split_to_train_and_test(dataset_df, sg_rnas, sg_rnas_in_test=15):
    train_sg_rnas = sg_rnas[:-sg_rnas_in_test]
    train_dataset_df = dataset_df[dataset_df[SG_RNA].isin(train_sg_rnas)]
    test_sg_rnas = sg_rnas[-sg_rnas_in_test:]
    test_dataset_df = dataset_df[dataset_df[SG_RNA].isin(test_sg_rnas)]

    return train_dataset_df, test_dataset_df


def load_dataset(
        data_type, sg_rnas, read_threshold=100, exclude_sg_rnas_without_positives=False,
        sg_rnas_to_exclude=None):
    dataset_df = pd.read_csv(
        DATASETS_PATH + "{}/include_on_targets/{}_CR_Lazzarotto_2020_dataset.csv".format(data_type, data_type))
    dataset_df = dataset_df[dataset_df[OFF_TARGET].str.find("N") == -1]
    # drop positives sites with less then read_threshold and set the labels
    dataset_df = dataset_df[(dataset_df[READS] >= read_threshold) | (dataset_df[READS] == 0)]
    dataset_df[LABEL] = 0
    dataset_df.loc[dataset_df[READS] > 0, LABEL] = 1
    # exclude sgRNAs without positives if needed
    if exclude_sg_rnas_without_positives:
        sg_rnas_with_positives = [target for target in sg_rnas if
                                  len(dataset_df[(dataset_df[SG_RNA] == target) & (dataset_df[LABEL] == 1)]) != 0]
        for sg_rna in sg_rnas:
            if sg_rna not in sg_rnas_with_positives:
                print("removed target:", sg_rna, "from dataset, since it has no positives")
        sg_rnas = sg_rnas_with_positives

    # drop sgRNAs not in the list provided
    dataset_df = dataset_df[dataset_df[SG_RNA].isin(sg_rnas)]

    # if provided drop sgRNAs that user provide
    if sg_rnas_to_exclude is not None:
        dataset_df = dataset_df[~dataset_df[SG_RNA].isin(sg_rnas_to_exclude)]

    return dataset_df


def load_generated_dataset_load_fun(sg_rna):
    sg_rna_dataset_df = pd.read_csv(DATASETS_PATH + "/generated_off_targets/{}.csv".format(sg_rna))
    sg_rna_dataset_df.rename({"off-target": OFF_TARGET}, axis=1, inplace=True)
    sg_rna_dataset_df[SG_RNA_SEQ] = sg_rna_dataset_df[SG_RNA]
    # Note - remove_alignment is False due to the fact that the generated dataset is non-aligned

    return sg_rna_dataset_df


def load_generated_dataset(sg_rnas):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_dfs = [executor.submit(
            load_generated_dataset_load_fun, sg_rna) for sg_rna in sg_rnas]
        dataset_df = pd.concat([f.result() for f in concurrent.futures.as_completed(future_dfs)])

    return dataset_df


def load_true_ot_dataset():
    dataset_df = pd.read_csv(
        DATASETS_PATH + "{}/true_ot_PK_Kota_2021_new_alignment.csv".format(Data_type.TRUE_OT))

    return dataset_df
