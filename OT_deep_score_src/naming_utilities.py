"""
This module contains functions to handle models naming
"""
from OT_deep_score_src.general_utilities import FILES_DIR, Model_task, Data_trans_type, Encoding_type


##########################################################################
def extract_model_name(model_type, model_task, include_distance_feature, include_sequence_features,
                       include_gmt_score, include_nuclea_seq_score, sample_weight, read_threshold,
                       trans_type=Data_trans_type.LOG1P, trans_all_fold=False, trans_only_positive=False,
                       exclude_sg_rnas_without_positives=False, encoding_type=False,
                       aligned=True):
    """
    extract model name
    """
    model_name = "{}-".format(model_type)
    model_name += "SW-" if sample_weight else ""
    model_name += "" if aligned else "NoAlign-"
    model_name += "thReads-{}-".format(read_threshold)
    model_name += "Classification" if model_task == Model_task.CLASSIFICATION_TASK else "Regression"
    model_name += "-seq" if include_sequence_features else ""
    model_name += "-dist" if include_distance_feature else ""
    model_name += "-GMT" if include_gmt_score else ""
    model_name += "-Nuclea" if include_nuclea_seq_score else ""
    model_name += "-positiveSgRNAs" if exclude_sg_rnas_without_positives else ""
    if model_task != Model_task.CLASSIFICATION_TASK:
        model_name += "-noTrans" if trans_type == Data_trans_type.NONE else ""
        model_name += "-log1pMaxTrans" if trans_type == Data_trans_type.LOG1P_MAX else ""
        model_name += "-maxTrans" if trans_type == Data_trans_type.MAX else ""
        model_name += "-standardTrans" if trans_type == Data_trans_type.STANDARD else ""
        model_name += "-boxTrans" if trans_type == Data_trans_type.BOX_COX else ""
        model_name += "-yeoTrans" if trans_type == Data_trans_type.YEO_JOHNSON else ""
        model_name += "-foldTrans" if trans_all_fold else ""
        model_name += "-positiveTrans" if trans_only_positive else ""
    model_name += "-{}".format(encoding_type) if encoding_type != Encoding_type.ONE_HOT else ""
    return model_name


##########################################################################
def prefix_and_suffix_path(model_task, model_type, data_type, include_distance_feature, include_sequence_features,
                           include_gmt_score, include_nuclea_seq_score, trans_type, trans_all_fold,
                           trans_only_positive, exclude_sg_rnas_without_positives, path_prefix):
    suffix = "_with_dist" if include_distance_feature else ""
    suffix += "" if include_sequence_features else "_without_seq"
    suffix += "_GMT" if include_gmt_score else ""
    suffix += "_Nuclea" if include_nuclea_seq_score else ""

    path_prefix += "{}/{}/{}/{}/".format(data_type, model_task, model_type, trans_type)
    path_prefix = path_prefix + "trans_only_positive/" if trans_only_positive else path_prefix
    path_prefix = path_prefix + "trans_on_entire_fold/" if trans_all_fold else path_prefix
    path_prefix = path_prefix + "drop_sg_rna_with_non_positives/" if exclude_sg_rnas_without_positives else path_prefix

    return path_prefix, suffix


def extract_model_path(model_task, data_type, include_distance_feature, include_sequence_features,
                       include_gmt_score, include_nuclea_seq_score, trans_type, trans_all_fold,
                       trans_only_positive, exclude_sg_rnas_without_positives, path_prefix,
                       model_type, encoding_type, bulges, sample_weight,
                       k_fold_number=None, fold_index=None):
    """
    extract model path
    """
    sw_txt = "_sw" if sample_weight else ""
    path_prefix, suffix = prefix_and_suffix_path(
        model_task, model_type, data_type, include_distance_feature,
        include_sequence_features, include_gmt_score, include_nuclea_seq_score,
        trans_type, trans_all_fold, trans_only_positive,
        exclude_sg_rnas_without_positives, path_prefix)
    model_path = FILES_DIR + ("bulges/" if bulges else "") + \
        ("{}_folds/".format(k_fold_number) if k_fold_number is not None else "no_folds/") + \
        path_prefix + ("model_fold_{}{}".format(fold_index, sw_txt) if k_fold_number is not None else
                       "model{}".format(sw_txt)) + suffix + ("_{}".format(encoding_type) if
                                                             encoding_type != Encoding_type.ONE_HOT else "")

    return model_path


# TODO: delete if not in usage anymore.
def extract_model_results_path(model_task, data_type, include_distance_feature,
                               include_sequence_features, include_gmt_score, include_nuclea_seq_score,
                               trans_type, trans_all_fold, trans_only_positive,
                               exclude_sg_rnas_without_positives, evaluate_only_distance, suffix_add, path_prefix,
                               model_type, encoding_type, bulges, k_fold_number=None):
    """
    extract model results path
    """
    path_prefix, suffix = prefix_and_suffix_path(
        model_task, model_type, data_type, include_distance_feature,
        include_sequence_features, include_gmt_score, include_nuclea_seq_score,
        trans_type, trans_all_fold, trans_only_positive,
        exclude_sg_rnas_without_positives, path_prefix)
    suffix = suffix + ("" if evaluate_only_distance is None else "_distance_" + str(evaluate_only_distance))
    suffix = suffix + suffix_add
    model_resutls_path = FILES_DIR + "bulges/" if bulges else "" + \
        "{}_folds/".format(k_fold_number) if k_fold_number is not None else "no_folds/" + \
        + path_prefix + "results/" + suffix + (
            "_{}".format(encoding_type) if encoding_type != Encoding_type.ONE_HOT else "") + ".csv"

    return model_resutls_path
