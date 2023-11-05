"""
This script uses the trained models to predict on the the TrueOT models.
"""


import random

from train_and_predict_scripts.utilities import TrainModelSpec, predict_main
from OT_deep_score_src.general_utilities import SEED, Model_type, Data_type

random.seed(SEED)


def main():
    test_data_type = Data_type.TRUE_OT
    sample_weight = False
    k_fold_number = 1
    read_threshold_train = 0
    aligned = True

    version = 1
    version = "{}_exclude_rhampseq_sg_rnas".format(version)
    version = "{}_continue_from_change_seq".format(version)
    model_parameters = {"batch_size": 512}
    include_distance_feature = False
    fixed_size_encoding = False

    train_model_spec_list = [
        TrainModelSpec(
            model_type=Model_type.C_1, predict_distance=False, model_version=version,
            model_parameters=model_parameters,
            include_distance_feature=include_distance_feature, sample_weight=sample_weight,
            fixed_size_encoding=fixed_size_encoding, flat_encoding=False,
            data_type=Data_type.FULL_GUIDE_SEQ,
            data_types_to_exclude=[Data_type.RHAMP_SEQ]),
        TrainModelSpec(
            model_type=Model_type.C_2, predict_distance=False, model_version=version,
            model_parameters=model_parameters,
            include_distance_feature=include_distance_feature, sample_weight=sample_weight,
            fixed_size_encoding=fixed_size_encoding, flat_encoding=False,
            data_type=Data_type.FULL_GUIDE_SEQ,
            data_types_to_exclude=[Data_type.RHAMP_SEQ]),
        TrainModelSpec(
            model_type=Model_type.C_3, predict_distance=False, model_version=version,
            model_parameters=model_parameters,
            include_distance_feature=include_distance_feature, sample_weight=sample_weight,
            fixed_size_encoding=fixed_size_encoding, flat_encoding=False,
            data_type=Data_type.FULL_GUIDE_SEQ,
            data_types_to_exclude=[Data_type.RHAMP_SEQ])
        ]

    predict_main(
        aligned, train_model_spec_list,
        prediction_table_path_prefix="{}_cleavage{}_v_{}_read_ts_{}{}".format(
            test_data_type, "_sw" if sample_weight else "", version,
            read_threshold_train, "_folds_{}".format(k_fold_number) if k_fold_number is not None else ""),
        test_data_type=test_data_type,
        read_threshold_train=read_threshold_train, k_fold_number=k_fold_number)


if __name__ == "__main__":
    main()
