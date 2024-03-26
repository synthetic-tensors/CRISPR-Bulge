"""
This module contains the utilizes functions for loading data and etc.
"""
import random
import pandas as pd
import numpy as np
import concurrent.futures

from OT_deep_score_src import general_utilities
from OT_deep_score_src.general_utilities import DATASETS_PATH, Data_type, SEED, SG_RNA, SG_RNA_SEQ, \
    OFF_TARGET, READS, LABEL


random.seed(SEED)


def load_sg_rnas_list(data_type=Data_type.CHANGE_SEQ):
    """
    Loads a list of sgRNAs from a pre-defined file for the specified data type.

    Args:
        data_type (Data_type, optional): The type of dataset to load. Defaults to Data_type.CHANGE_SEQ.

    Returns:
        list: A list of sgRNA sequences.
    """
    sg_rnas_s = pd.read_csv(
        general_utilities.DATASETS_PATH + "{}_sgRNAs_list.csv".format(data_type),
        header=None).squeeze("columns")
    return list(sg_rnas_s)


def load_order_sg_rnas(data_type=Data_type.CHANGE_SEQ):
    """
    Loads a list of sgRNAs in a specific order for k-fold cross-validation.

    Args:
        data_type (Data_type, optional): The type of dataset to load. Defaults to Data_type.CHANGE_SEQ.

    Returns:
        list: A list of sgRNA sequences in a predefined order.
    """
    sg_rnas_s = pd.read_csv(
        general_utilities.DATASETS_PATH + "{}_sgRNAs_ordering.csv".format(data_type),
        header=None).squeeze("columns")
    return list(sg_rnas_s)


def order_sg_rnas(data_type=Data_type.CHANGE_SEQ):
    """
    Creates and saves an ordered list of sgRNAs for k-fold cross-validation.

    Args:
        data_type (Data_type, optional): The type of dataset. Defaults to Data_type.CHANGE_SEQ.

    Returns:
        list: A list of sgRNA sequences in a randomized order.
    """
    if data_type in [Data_type.CHANGE_SEQ,  Data_type.FULL_GUIDE_SEQ, Data_type.GUIDE_SEQ, Data_type.NEW_GUIDE_SEQ]:
        dataset_df = pd.read_csv(
            general_utilities.DATASETS_PATH + "{}/include_on_targets/{}_CR_Lazzarotto_2020_dataset.csv".format(
                data_type, data_type))
    else:
        dataset_df = pd.read_csv(general_utilities.DATASETS_PATH + "{}.csv".format(data_type))

    sg_rnas = list(dataset_df[SG_RNA].unique())
    print("There are", len(sg_rnas), "unique sgRNAs in the", data_type, "dataset")

    # sort the sgRNAs and shuffle them
    sg_rnas.sort()
    random.seed(SEED)
    random.shuffle(sg_rnas)

    # save the sgRNAs order into csv file
    sg_rnas_s = pd.Series(sg_rnas)
    # to csv - you can read this to Series using -
    # pd.read_csv("file_name.csv", header=None, squeeze=True)
    sg_rnas_s.to_csv(general_utilities.DATASETS_PATH + "{}_sgRNAs_ordering.csv".format(data_type),
                     header=False, index=False)

    return sg_rnas


def split_sg_rnas_into_folds(k_fold_number, sg_rnas,
                             load_predefined_folds=False, data_type=None):
    """
    Splits sgRNAs into k-folds for cross-validation.

    Args:
        k_fold_number (int): The number of folds for cross-validation.
        sg_rnas (list): A list of sgRNA sequences.
        load_predefined_folds (bool, optional): If True, loads folds from a file. Defaults to False.
        data_type (Data_type, optional):  The dataset type (if loading predefined folds).

    Returns:
        list: A list of lists, where each inner list contains sgRNAs for a specific fold.
    """
    if load_predefined_folds:
        print("Loading folds sgRNAs from predefinded file.")
        try:
            # each row is fold
            folds_df = pd.read_csv(
                general_utilities.DATASETS_PATH + "{}_sgRNAs_folds_split.csv".format(data_type))
            sg_rna_folds_list = folds_df.values
            sg_rna_folds_list = [fold[~pd.isnull(fold)] for fold in list(sg_rna_folds_list)]
        except FileNotFoundError:
            raise ValueError("predefined folds file is not exists for this dataset")

        # confirm that the sgRNAs in the folds are the right ones
        sg_rnas_in_folds = set([sg_rna for fold in sg_rna_folds_list for sg_rna in fold])
        if len(sg_rnas_in_folds) != len(sg_rnas) or len(sg_rnas_in_folds.difference(sg_rnas)) != 0:
            raise ValueError("sgRNAs in folds do not match the one provided")
    else:
        sg_rna_folds_list = np.array_split(
            sg_rnas, k_fold_number) if (k_fold_number is not None and k_fold_number > 1) else [[]]

    return sg_rna_folds_list


def create_fold_sets(target_fold, targets, dataset_df,
                     exclude_sg_rnas_without_positives):
    """
    Creates training and testing datasets based on a specified target fold.

    Args:
        target_fold (list): A list of sgRNAs representing the test fold.
        targets (list): A list of all available sgRNAs.
        dataset_df (pd.DataFrame): dataset DataFrame containing sgRNAs, off-targets, labels, etc.
        exclude_sg_rnas_without_positives (bool): If True, excludes sgRNAs without positive
            examples from the training set. It does not matter in the test set, as we cannot evaluate the
            performance when evaluating per sgRNA. Moreover, we can always remove them in the evaluation stage.

    Returns:
        tuple:
            * dataset_df_test (pd.DataFrame): DataFrame for the test set.
            * dataset_df_train (pd.DataFrame): DataFrame for the training set.
    """
    test_targets = target_fold
    train_targets = [target for target in targets if target not in target_fold]
    if exclude_sg_rnas_without_positives:
        for target in train_targets.copy():
            if len(dataset_df[(dataset_df[SG_RNA] == target) & (dataset_df[LABEL] == 1)]) == 0:
                print("removing target:", target, "from training set, since it has no positives")
                train_targets.remove(target)
    dataset_df_test = dataset_df[dataset_df[SG_RNA].isin(test_targets)]
    dataset_df_train = dataset_df[dataset_df[SG_RNA].isin(train_targets)]

    return dataset_df_test, dataset_df_train


def split_to_train_and_test(dataset_df, sg_rnas, sg_rnas_in_test=15):
    """
    Splits the dataset into train and test sets based on a specified number of sgRNAs in the test set.

    Args:
        dataset_df (pd.DataFrame): The dataset to split.
        sg_rnas (list): A list of sgRNA sequences.
        sg_rnas_in_test (int, optional): Number of sgRNAs to include in the test set. Defaults to 15.

    Returns:
        tuple:
            * train_dataset_df (pd.DataFrame): The training dataset.
            * test_dataset_df (pd.DataFrame): The testing dataset.

    """
    train_sg_rnas = sg_rnas[:-sg_rnas_in_test]
    train_dataset_df = dataset_df[dataset_df[SG_RNA].isin(train_sg_rnas)]
    test_sg_rnas = sg_rnas[-sg_rnas_in_test:]
    test_dataset_df = dataset_df[dataset_df[SG_RNA].isin(test_sg_rnas)]

    return train_dataset_df, test_dataset_df


def load_dataset(
        data_type, sg_rnas, read_threshold=100, exclude_sg_rnas_without_positives=False,
        sg_rnas_to_exclude=None):
    """
    Loads and preprocesses a off-target dataset.

    Args:
        data_type (Data_type): The type of dataset to load.
        sg_rnas (list): A list of sgRNA sequences to include in the dataset.
        read_threshold (int, optional): Minimum read count threshold for positive examples. Defaults to 100.
        exclude_sg_rnas_without_positives (bool, optional): If True, excludes sgRNAs without positive
                                                            examples. Defaults to False.
        sg_rnas_to_exclude (list, optional): A list of sgRNAs to exclude from the dataset.

    Returns:
        pd.DataFrame: The processed dataset.
    """
    if data_type in [Data_type.CHANGE_SEQ,  Data_type.FULL_GUIDE_SEQ, Data_type.GUIDE_SEQ, Data_type.NEW_GUIDE_SEQ]:
        dataset_df = pd.read_csv(
            DATASETS_PATH + "{}/include_on_targets/{}_CR_Lazzarotto_2020_dataset.csv".format(data_type, data_type))
    else:
        dataset_df = pd.read_csv(
            DATASETS_PATH + "{}.csv".format(data_type))
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
        print("Excluding the following sgRNAs from training:", sg_rnas_to_exclude)
        dataset_df = dataset_df[~dataset_df[SG_RNA].isin(sg_rnas_to_exclude)]

    return dataset_df


def load_generated_dataset_load_fun(sg_rna):
    """
    Loads a generated off-target dataset for a single sgRNA (helper function).

    Args:
        sg_rna (str): The sgRNA sequence.

    Returns:
        pd.DataFrame: The dataset for the specified sgRNA.
    """
    sg_rna_dataset_df = pd.read_csv(DATASETS_PATH + "/generated_off_targets/{}.csv".format(sg_rna))
    sg_rna_dataset_df.rename({"off-target": OFF_TARGET}, axis=1, inplace=True)
    sg_rna_dataset_df[SG_RNA_SEQ] = sg_rna_dataset_df[SG_RNA]
    # Note - remove_alignment is False due to the fact that the generated dataset is non-aligned

    return sg_rna_dataset_df


def load_generated_dataset(sg_rnas):
    """
    Loads a generated off-target dataset for multiple sgRNAs using parallel processing.

    Args:
        sg_rnas (list): A list of sgRNA sequences.

    Returns:
        pd.DataFrame: The combined dataset for all sgRNAs.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_dfs = [executor.submit(
            load_generated_dataset_load_fun, sg_rna) for sg_rna in sg_rnas]
        dataset_df = pd.concat([f.result() for f in concurrent.futures.as_completed(future_dfs)])

    return dataset_df


def load_dataset_from_file(file_name):
    """
    Loads a dataset from a specified CSV file.

    Args:
        file_name (str): The name of the CSV file (without the extension).

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    dataset_df = pd.read_csv(
        DATASETS_PATH + "{}.csv".format(file_name))

    return dataset_df
