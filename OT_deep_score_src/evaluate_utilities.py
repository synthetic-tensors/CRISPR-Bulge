"""
This module contains the functions for evaluation of the models
"""
import pandas as pd
import numpy as np
from OT_deep_score_src.general_utilities import Model_task, SG_RNA, LABEL, READS, BULGES
from scipy.stats.stats import pearsonr, spearmanr

from sklearn.metrics import average_precision_score


def measure_clevage_acc(prediction_file_name, models=None, only_bulges=False, only_mismatches=False,
                        model_task=Model_task.REGRESSION_TASK):
    """
    Calculates prediction accuracy metrics (AUPR, Pearson, Spearman) for a set of models.

    Args:
        prediction_file_name (str or pd.DataFrame): Path to a CSV file containing predictions or a
                                                    DataFrame with the predictions.
        models (list): List of model names (columns in the predictions data) to evaluate.
        only_bulges (bool, optional): If True, evaluates only on off-targets with bulges. Defaults to False.
        only_mismatches (bool, optional): If True, evaluates only on off-targets with mismatches (no bulges).
                                          Defaults to False.
        model_task (Model_task, optional): Specifies whether the models are regression or classification based.
                                           Defaults to Model_task.REGRESSION_TASK.

    Returns:
        pd.DataFrame: DataFrame containing AUPR, Pearson, and Spearman correlations for each model.
    """
    if models is None:
        raise ValueError("models parameter should not be None")

    if isinstance(prediction_file_name, str):
        predictions = pd.read_csv(prediction_file_name)
    else:
        predictions = prediction_file_name
    if only_bulges:
        predictions = predictions[predictions[BULGES] != 0]
    elif only_mismatches:
        predictions = predictions[predictions[BULGES] == 0]

    predictions_pos = predictions[(~predictions[READS].isna()) & (predictions[READS] > 0)]
    auprs = [average_precision_score(predictions[LABEL], predictions[model]) for
             model in models]
    pearsons = [pearsonr(np.log(predictions_pos[READS]+1) if model_task == Model_task.REGRESSION_TASK else
                         predictions_pos[READS], predictions_pos[model])[0] for model in models]
    spearmans = [spearmanr(predictions_pos[READS], predictions_pos[model])[0] for
                 model in models]

    results_df = pd.DataFrame(data={"model": models, "AUPR": auprs, "Pearson": pearsons, "Spearman": spearmans})
    pd.set_option("display.precision", 3)

    return results_df


def measure_clevage_acc_per_fold_scores(prediction_file_name, models=None, only_bulges=False, only_mismatches=False,
                                        model_task=Model_task.REGRESSION_TASK):
    """
    Calculates prediction accuracy metrics per fold.

    Args:
        prediction_file_name (str or pd.DataFrame): Path to a CSV file containing predictions or a
                                                    DataFrame with the predictions.
        models (list): List of model names (columns in the predictions data) to evaluate.
        only_bulges (bool, optional): If True, evaluates only on off-targets with bulges. Defaults to False.
        only_mismatches (bool, optional): If True, evaluates only on off-targets with mismatches (no bulges).
                                          Defaults to False.
        model_task (Model_task, optional): Specifies whether the models are regression or classification based.
                                           Defaults to Model_task.REGRESSION_TASK.

    Returns:
        tuple:
            * folds_auprs (np.array): AUPR scores per fold for each model.
            * folds_pearsons (np.array): Pearson correlations per fold for each model.
            * folds_spearmans (np.array): Spearman correlations per fold for each model.
    """
    if models is None:
        raise ValueError("models parameter should not be None")

    folds_auprs, folds_pearsons, folds_spearmans = [], [], []
    if isinstance(prediction_file_name, str):
        df = pd.read_csv(prediction_file_name)
    else:
        df = prediction_file_name
    if only_bulges:
        df = df[df[BULGES] != 0]
    elif only_mismatches:
        df = df[df[BULGES] == 0]

    folds = df["fold"].unique()
    for fold_index in folds:
        predictions_fold = df[df["fold"] == fold_index]
        predictions_fold_pos = predictions_fold[
            (~predictions_fold[READS].isna()) & (predictions_fold[READS] > 0)]
        if len(predictions_fold_pos) < 2:
            print("skiping fold {} as it has less then 2 positives sites".format(fold_index))
            continue
        auprs = [average_precision_score(predictions_fold[LABEL], predictions_fold[model]) for
                 model in models]
        pearsons = [pearsonr(np.log(predictions_fold_pos[READS]+1) if model_task == Model_task.REGRESSION_TASK else
                             predictions_fold_pos[READS], predictions_fold_pos[model])[0] for model in models]
        spearmans = [spearmanr(predictions_fold_pos[READS], predictions_fold_pos[model])[0] for
                     model in models]
        folds_auprs.append(auprs)
        folds_pearsons.append(pearsons)
        folds_spearmans.append(spearmans)

    folds_auprs, folds_pearsons, folds_spearmans = \
        np.array(folds_auprs), np.array(folds_pearsons), np.array(folds_spearmans)

    return folds_auprs, folds_pearsons, folds_spearmans


def measure_clevage_acc_per_fold(prediction_file_name, models=None, only_bulges=False, only_mismatches=False,
                                 model_task=Model_task.REGRESSION_TASK):
    """
    Calculates average and standard deviation of cleavage prediction accuracy metrics across folds.

    Args:
        prediction_file_name (str or pd.DataFrame): Path to a CSV file containing predictions or
                                                    a DataFrame with the predictions.
        models (list): List of model names (columns in the predictions data) to evaluate.
        only_bulges (bool, optional): If True, evaluates only on off-targets with bulges. Defaults to False.
        only_mismatches (bool, optional): If True, evaluates only on off-targets with mismatches (no bulges).
                                          Defaults to False.
        model_task (Model_task, optional): Specifies whether the models are regression or classification based.
                                           Defaults to Model_task.REGRESSION_TASK.

    Returns:
        pd.DataFrame: DataFrame containing mean and standard deviation of AUPR, Pearson, and Spearman correlations
        for each model.
    """
    folds_auprs, folds_pearsons, folds_spearmans = \
        measure_clevage_acc_per_fold_scores(prediction_file_name, models, only_bulges, only_mismatches, model_task)

    auprs_means, pearsons_means, spearmans_means = \
        folds_auprs.mean(axis=0), folds_pearsons.mean(axis=0), folds_spearmans.mean(axis=0)
    auprs_stds, pearsons_stds, spearmans_stds = \
        folds_auprs.std(axis=0), folds_pearsons.std(axis=0), folds_spearmans.std(axis=0)
    results_df = pd.DataFrame(data={"model": models, "AUPR": auprs_means,
                                    "AUPR std": auprs_stds, "Pearson": pearsons_means,
                                    "Pearson std": pearsons_stds, "Spearman": spearmans_means,
                                    "Spearman std": spearmans_stds})
    pd.set_option("display.precision", 3)

    return results_df


def measure_clevage_acc_per_guide(prediction_file_name, models=None, only_bulges=False, only_mismatches=False,
                                  model_task=Model_task.REGRESSION_TASK):
    """
    Calculates prediction accuracy metrics per sgRNA.

    Args:
        prediction_file_name (str or pd.DataFrame): Path to a CSV file containing predictions or a
                                                    DataFrame with the predictions.
        models (list): List of model names (columns in the predictions data) to evaluate.
        only_bulges (bool, optional): If True, evaluates only on off-targets with bulges. Defaults to False.
        only_mismatches (bool, optional): If True, evaluates only on off-targets with mismatches (no bulges).
                                          Defaults to False.
        model_task (Model_task, optional): Specifies whether the models are regression or classification based.
                                           Defaults to Model_task.REGRESSION_TASK.

    Returns:
        pd.DataFrame: DataFrame containing AUPR, Pearson, and Spearman correlations for each model.
    """
    if models is None:
        raise ValueError("models parameter should be None")

    targets_auprs, targets_pearsons, targets_spearmans = [], [], []
    if isinstance(prediction_file_name, str):
        df = pd.read_csv(prediction_file_name)
    else:
        df = prediction_file_name
    if only_bulges:
        df = df[df[BULGES] != 0]
    elif only_mismatches:
        df = df[df[BULGES] == 0]

    targets = df[SG_RNA].unique()
    for target in targets:
        predictions_target = df[df[SG_RNA] == target]
        predictions_target_pos = predictions_target[
            (~predictions_target[READS].isna()) & (predictions_target[READS] > 0)]
        if len(predictions_target_pos) < 2:
            print("skiping sgRNA {} as it has less then 2 positives sites".format(target))
            continue
        auprs = [average_precision_score(predictions_target[LABEL], predictions_target[model]) for
                 model in models]
        pearsons = [pearsonr(np.log(predictions_target_pos[READS]+1) if model_task == Model_task.REGRESSION_TASK else
                             predictions_target_pos[READS], predictions_target_pos[model])[0] for model in models]
        spearmans = [spearmanr(predictions_target_pos[READS], predictions_target_pos[model])[0] for
                     model in models]
        targets_auprs.append(auprs)
        targets_pearsons.append(pearsons)
        targets_spearmans.append(spearmans)

    targets_auprs, targets_pearsons, targets_spearmans = \
        np.array(targets_auprs), np.array(targets_pearsons), np.array(targets_spearmans)
    auprs_means, pearsons_means, spearmans_means = \
        targets_auprs.mean(axis=0), targets_pearsons.mean(axis=0), targets_spearmans.mean(axis=0)
    auprs_stds, pearsons_stds, spearmans_stds = \
        targets_auprs.std(axis=0), targets_pearsons.std(axis=0), targets_spearmans.std(axis=0)
    results_df = pd.DataFrame(data={"model": models, "AUPR": auprs_means,
                                    "AUPR std": auprs_stds, "Pearson": pearsons_means,
                                    "Pearson std": pearsons_stds, "Spearman": spearmans_means,
                                    "Spearman std": spearmans_stds})
    pd.set_option("display.precision", 3)

    return results_df
