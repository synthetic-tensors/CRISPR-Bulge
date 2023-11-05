"""
This script show an example of how to predic with the models only on the GUIDE-seq dataset in 10-fold manner.
"""


import random

from train_and_predict_scripts.utilities import TrainModelSpec, predict_main
from OT_deep_score_src.general_utilities import SEED, Model_type, Data_type

random.seed(SEED)


def main():
    data_type = Data_type.GUIDE_SEQ
    k_fold_number = 10
    read_threshold_train = 0
    aligned = True

    version = 1
    model_parameters = {"batch_size": 512}
    include_distance_feature = False
    fixed_size_encoding = False

    for sample_weight in (False, True):
        train_model_spec_list = [
            TrainModelSpec(
                model_type=Model_type.XGBOOST, predict_distance=False, model_version=1, model_parameters=None,
                include_distance_feature=include_distance_feature, sample_weight=sample_weight,
                fixed_size_encoding=fixed_size_encoding, flat_encoding=True,
                data_type=data_type),
            TrainModelSpec(
                model_type=Model_type.C_1, predict_distance=False, model_version=version,
                model_parameters=model_parameters,
                include_distance_feature=include_distance_feature, sample_weight=sample_weight,
                fixed_size_encoding=fixed_size_encoding, flat_encoding=False,
                data_type=data_type),
            TrainModelSpec(
                model_type=Model_type.C_2, predict_distance=False, model_version=version,
                model_parameters=model_parameters,
                include_distance_feature=include_distance_feature, sample_weight=sample_weight,
                fixed_size_encoding=fixed_size_encoding, flat_encoding=False,
                data_type=data_type),
            TrainModelSpec(
                model_type=Model_type.C_3, predict_distance=False, model_version=version,
                model_parameters=model_parameters,
                include_distance_feature=include_distance_feature, sample_weight=sample_weight,
                fixed_size_encoding=fixed_size_encoding, flat_encoding=False,
                data_type=data_type)
            ]

        predict_main(
            aligned, train_model_spec_list,
            prediction_table_path_prefix="{}_cleavage{}_v_{}_read_ts_{}{}".format(
                data_type, "_sw" if sample_weight else "", version,
                read_threshold_train, "_folds_{}".format(k_fold_number) if k_fold_number is not None else ""),
            read_threshold_train=read_threshold_train, k_fold_number=k_fold_number)


if __name__ == "__main__":
    main()
