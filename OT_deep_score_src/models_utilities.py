"""
This module contains functions realated to the models (loading models and etc)
"""
import numpy as np

from OT_deep_score_src.general_utilities import Data_type, Data_trans_type, SG_RNA, OFF_TARGET, Encoding_type, \
    Model_task, Model_type
from OT_deep_score_src.naming_utilities import extract_model_path
from OT_deep_score_src.models_inter import Model
from OT_deep_score_src.base_models import XGboost_model, CatBoost_model, SVM_model, SVM_linear_model, \
    MLP_model, RandomForest_model, AdaBoost_model, SGD_model, Lasso_model
from OT_deep_score_src.nn_models import C_1, C_2, C_3, D_1, D_2, D_3, D2C_2, D2C_2_1, D2C_3

from models.nuclea_seq_modeling.modeling import log10_crispr_specificity
from models.moff_modeling.modeling import GMT_score


def nuclea_seq_score_prediction(sg_rna, off_target):
    """
    Calculates the predicted cleavage specificity score using the Nuclea-seq model (WT variant).

    Args:
        sg_rna (str): The sgRNA sequence.
        off_target (str): The off-target sequence.

    Returns:
        float: The predicted Nuclea-seq score (log10).
    """
    # TODO: this is only a plaster, I am not sure how the handle the gap in the sgRNA.
    # sg_rna_without_gaps = sg_rna.replace("-", "")
    # assume N in the PAM is the corresponding nucleotide in the off-target sequence
    # pam_seq = off_target[-3] + sg_rna[-2:] if sg_rna[-3] == "N" else sg_rna_without_gaps[-3:] - wrong
    pam_seq = "T" + off_target[-2:]  # Nuclea-seq was trained only with TGG
    return log10_crispr_specificity("WT", pam_seq, sg_rna[:-3], off_target[:-3])


def gmt_score_prediction(dataset_df):
    """
    Calculates predicted GMT scores using the MOFF model and assigns them to the DataFrame.

    Args:
        dataset_df (pd.DataFrame): DataFrame containing at least "sgRNA" and "off-target" columns.

    Returns:
        np.ndarray: An array of predicted GMT scores corresponding to the input data.
    """
    dataset_df = dataset_df.rename({SG_RNA: "sgRNA", OFF_TARGET: "off-target"}, axis=1)
    dataset_moff_df = dataset_df[["sgRNA", "off-target"]].drop_duplicates(subset=["sgRNA"])
    dataset_moff_df["off-target"] = dataset_moff_df["sgRNA"].values
    dataset_moff_df = GMT_score(dataset_moff_df)

    dataset_df["GMT"] = np.nan
    for sg_rna in dataset_df["sgRNA"].unique():
        dataset_df.loc[dataset_df["sgRNA"] == sg_rna, ["GMT"]] = \
            dataset_moff_df.loc[dataset_moff_df["sgRNA"] == sg_rna, "GOP"].values[0]

    return dataset_df["GMT"].values


def model_selection(model_task, model_type, model_parameters, gpu,
                    pretrained_model, val_size,
                    input_shape, file_path_and_name):
    """
    Selects and instantiates an appropriate model based on provided criteria.

    Args:
        model_task (Model_task): Specifies whether the task is classification or regression.
        model_type (Model_type): The desired model type (e.g., XGBOOST, SVM, MLP).
        model_parameters (dict, optional): Model-specific parameters.
        gpu (bool): Indicates whether to use a GPU for training.
        pretrained_model (str or None): Path to a pre-trained model for transfer learning.
        val_size (float): Validation set size for neural network models.
        input_shape (tuple): Input shape for neural network models.
        file_path_and_name (str):  Path for storing the model fit log (neural networks).

    Returns:
        tuple:
            * model (Model): The instantiated model object.
            * fit_kwargs (dict): Keyword arguments for the model "fit" method.
    """
    if model_parameters is None:
        model_parameters = {}
    fit_kwargs = {}
    if model_type == Model_type.XGBOOST:
        model = XGboost_model(
            model_task=model_task, gpu=gpu, **model_parameters)
        fit_kwargs = {"pretrained_model": pretrained_model}
    elif model_type == Model_type.SVM:
        model = SVM_model(model_task=model_task, **model_parameters)
    elif model_type == Model_type.SVM_LINEAR:
        model = SVM_linear_model(model_task=model_task, **model_parameters)
    elif model_type == Model_type.SGD:
        model = SGD_model(model_task=model_task, **model_parameters)
    elif model_type == Model_type.MLP:
        model = MLP_model(model_task=model_task, input_shape=input_shape, **model_parameters)
        fit_kwargs = {"val_size": val_size, "train_fit_log_path": file_path_and_name}
    elif model_type == Model_type.ADABOOST:
        model = AdaBoost_model(model_task=model_task, **model_parameters)
    elif model_type == Model_type.RF:
        model = RandomForest_model(model_task=model_task, **model_parameters)
    elif model_type == Model_type.LASSO:
        model = Lasso_model(model_task=model_task, **model_parameters)
    elif model_type == Model_type.CATBOOST:
        model = CatBoost_model(model_task=model_task, gpu=gpu, **model_parameters)
    elif model_type == Model_type.C_1:
        model = C_1(model_task=model_task, input_shape=input_shape, **model_parameters)
        fit_kwargs = {"val_size": val_size, "train_fit_log_path": file_path_and_name}
    elif model_type == Model_type.C_2:
        model = C_2(model_task=model_task, input_shape=input_shape, **model_parameters)
        fit_kwargs = {"val_size": val_size, "train_fit_log_path": file_path_and_name}
    elif model_type == Model_type.C_3:
        model = C_3(model_task=model_task, input_shape=input_shape, **model_parameters)
        fit_kwargs = {"val_size": val_size, "train_fit_log_path": file_path_and_name}
    elif model_type == Model_type.D_1:
        model = D_1(model_task=model_task, input_shape=input_shape, **model_parameters)
        fit_kwargs = {"val_size": val_size, "train_fit_log_path": file_path_and_name}
    elif model_type == Model_type.D_2:
        model = D_2(model_task=model_task, input_shape=input_shape, **model_parameters)
        fit_kwargs = {"val_size": val_size, "train_fit_log_path": file_path_and_name}
    elif model_type == Model_type.D_3:
        model = D_3(model_task=model_task, input_shape=input_shape, **model_parameters)
        fit_kwargs = {"val_size": val_size, "train_fit_log_path": file_path_and_name}
    elif model_type == Model_type.D2C_2:
        model = D2C_2(model_task=model_task, pretrained_model=pretrained_model,
                      input_shape=input_shape, **model_parameters)
        fit_kwargs = {"val_size": val_size, "train_fit_log_path": file_path_and_name}
    elif model_type == Model_type.D2C_2_1:
        model = D2C_2_1(model_task=model_task, pretrained_model=pretrained_model,
                        input_shape=input_shape, **model_parameters)
        fit_kwargs = {"val_size": val_size, "train_fit_log_path": file_path_and_name}
    elif model_type == Model_type.D2C_3:
        model = D2C_3(model_task=model_task, pretrained_model=pretrained_model,
                      input_shape=input_shape, **model_parameters)
        fit_kwargs = {"val_size": val_size, "train_fit_log_path": file_path_and_name}
    else:
        raise ValueError("model_type is not one of the options")

    return model, fit_kwargs


def load_model_from_pkl(
        model_task=Model_task.CLASSIFICATION_TASK, data_type=Data_type.CHANGE_SEQ, sample_weight=True,
        model_type=Model_type.XGBOOST, include_distance_feature=False, include_sequence_features=True,
        include_gmt_score=False, include_nuclea_seq_score=False,
        trans_type=Data_trans_type.LOG1P, trans_all_fold=False, trans_only_positive=False,
        bulges=False, encoding_type=Encoding_type.ONE_HOT, path_prefix="", k_fold_number=None, fold_index=None,
        exclude_sg_rnas_without_positives=False):
    """
    Loads a pre-trained off-target prediction model from a pickle file.

    Args:
        model_task (Model_task, optional): The task type (classification or regression).
                                           Defaults to Model_task.CLASSIFICATION_TASK.
        data_type (Data_type, optional): The type of dataset for which the model was trained.
                                         Defaults to Data_type.CHANGE_SEQ.
        sample_weight (bool, optional): Whether the model was trained using sample weights.
                                        Defaults to True.
        model_type (Model_type, optional): The type of model (e.g., XGboost, SVM).
                                           Defaults to Model_type.XGBOOST.
        include_distance_feature (bool, optional): Whether the model includes the distance feature.
                                                   Defaults to False.
        include_sequence_features (bool, optional): Whether the model includes sequence features.
                                                    Defaults to True.
        include_gmt_score (bool, optional): Whether the model includes the GMT score.
                                            Defaults to False.
        include_nuclea_seq_score (bool, optional): Whether the model includes the Nuclea-seq score.
                                                   Defaults to False.
        trans_type (Data_trans_type, optional): The type of data transformation used during training.
                                                Defaults to Data_trans_type.LOG1P.
        trans_all_fold (bool, optional): Whether the transformation was applied across all folds.
                                         Defaults to False.
        trans_only_positive (bool, optional): Whether the transformation was applied only to positive examples.
                                              Defaults to False.
        bulges (bool, optional): Whether the model was trained to handle bulges.
                                 Defaults to False.
        encoding_type (Encoding_type, optional): The type of encoding used.
                                                 Defaults to Encoding_type.ONE_HOT.
        path_prefix (str, optional): An optional prefix for the model file path. Defaults to "".
        k_fold_number (int, optional): The k-fold number for cross-validation models. Defaults to None.
        fold_index (int, optional): The fold index for cross-validation models, to load spesific fold train.
                                    Defaults to None.
        exclude_sg_rnas_without_positives (bool, optional): Whether the model was trained excluding
                                                            sgRNAs without any positive examples.
                                                            Defaults to False.

    Returns:
        Model: The loaded model object.
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
        bulges=bulges, sample_weight=sample_weight, k_fold_number=k_fold_number, fold_index=fold_index)
    print(file_path_and_name)
    return Model.load_model_instance(file_path_and_name)
