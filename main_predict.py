"""
This module provides some examples of generating predictions from saved models.
In addition, it shows how we generated all the predictions we used in our evaluations.
"""
import tensorflow as tf

import train_and_predict_scripts.predict_config as predict_config
from train_and_predict_scripts.utilities import ensemble_predict


def main():
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Example of loading the models from file path, and predict on any dataset.
    # This is the optimal usage for quick prediction from saved model on new dataset.
    # Also computed average emsemble
    # In this example, we generate predictions on independent refined TrueOT dataset.
    ensemble_predict(
        ensemble_components_file_path_and_name_list=[
            "files/bulges/1_folds/5_revision_ensemble_{}_exclude_RHAMPseq_continue_from_change_seq/"
            "read_ts_0/cleavage_models/aligned/FullGUIDEseq/classification/c_2/"
            "ln_x_plus_one_trans/model_fold_0".format(i) for i in range(5)],
        dataset_df="files/datasets/Refined_TrueOT.csv")

    # Example of using the predict_config that contains configures of prediction scenarios we used.
    # See predict_config for all the different configurations.
    # In this example, we generate predictions on independent GUIDE-seq and refined TrueOT datasets.
    # Note that it does not compute the average ensemble prediction, and this need to be done manually.
    for ensemble_componet_i in range(5):
        for setting_number in (8, 10, 12, 14):
            predict_config.main(
                version="5_revision_ensemble_{}".format(ensemble_componet_i), setting_number=setting_number)


if __name__ == "__main__":
    main()

# To run other prediction scenarios we used, you will need to retrain the required models.
# Use train_1_fold.py and train_folds.py to do so.

# for setting_number in (1, 2, 3, 4, 5, 8, 10, 12, 14):
#     predict_config.main(version="5_revision", setting_number=setting_number)

# for setting_number in (16,):
#     predict_config.main(version="5_revision", setting_number=setting_number)

# for ensemble_componet_i in range(5):
#     for setting_number in (5, 9, 11, 13, 15):
#         predict_config.main(
#             version="5_revision_ensemble_{}".format(ensemble_componet_i), setting_number=setting_number)

# for ensemble_componet_i in range(5):
#     for setting_number in (17, 18, 19, 20):
#         predict_config.main(
#             version="5_revision_ensemble_{}".format(ensemble_componet_i), setting_number=setting_number)
