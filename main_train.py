"""
This module provides some examples of using the train_1_fold and train_folds without using the shell commands
"""
import tensorflow as tf

import train_1_fold
import train_folds
from OT_deep_score_src.general_utilities import Data_type, Model_task


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

    # Example of using the training on entire dataset -
    # TL Ensemble models from CHANGE-seq to Full GUIDE-seq dataset.
    # Excludes data of sgRNAs from RHAMP-seq.
    train_1_fold.train_handler(
        read_threshold=0, sample_weight=False, data_type=Data_type.FULL_GUIDE_SEQ,
        data_types_to_exclude=Data_type.RHAMP_SEQ, model_version="5_revision", num_ensembles=5,
        transfer_learning=True, model_tasks=(Model_task.CLASSIFICATION_TASK, Model_task.REGRESSION_TASK),
        model_type=None)

    # Example of using the training on in 10-fold-cross validation dataset -
    # TL Ensemble models from CHANGE-seq to GUIDE-seq dataset.
    train_folds.train_handler(
        read_threshold=0, sample_weight=False, num_ensembles=5, model_version="5_revision",
        data_type="TL", model_tasks=(Model_task.CLASSIFICATION_TASK, Model_task.REGRESSION_TASK))


if __name__ == "__main__":
    main()
