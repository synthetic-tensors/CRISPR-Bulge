"""
general utilities
"""
from concurrent.futures import ThreadPoolExecutor
import pathlib
import os

from enum import Enum

import numpy as np
import pandas as pd

N_CORES = os.cpu_count() if isinstance(os.cpu_count(), int) else 1

# ##############paths######################
HOME_DIR = str(pathlib.Path(__file__).parent.parent.absolute()) + "/"  # project path
FILES_DIR = HOME_DIR + "files/"
DATASETS_PATH = FILES_DIR + "datasets/"

# #################constants##################
SEED = 10


SG_RNA = "sgRNA"
SG_RNA_SEQ = "Align.sgRNA"
OFF_TARGET = "Align.off-target"
CHROM = "chrom"
CROM_START_LOCATION = "Align.chromStart"
CROM_END_LOCATION = "Align.chromEnd"
READS = "reads"
LABEL = "label"
DISTANCE = "Align.#Edit"
MISMATCHES = "Align.#Mismatches"
BULGES = "Align.#Bulges"
MAX_DISTANCE = 6

DEFAULT_READ_THRESHOLD = 100


# TODO: __eq__ is implemented for backward compatibility, consider to remove in future, and also hash


class Padding_type(Enum):
    """
    Represents padding types for sequence alignment.

    Values:
        NONE: No padding.
        GAP: Pad with gap characters.
    """

    NONE = None
    GAP = "gap_pad"
    # ZERO = "zero_pad"

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, __o) -> bool:
        return self is __o or self.value == __o

    def __hash__(self) -> int:
        return super().__hash__()


class Encoding_type(Enum):
    """
    Represents encoding types for sgRNA and OTS sequences.

    Values:
        ONE_HOT: One-hot encoding.
        CRISPR_NET: CRISPR-Net style encoding.
        FIXED_SIZE: Fixed-size representation (deprecated).
    """

    ONE_HOT = "oneHotEncoding"
    CRISPR_NET = "crisprNetEncoding"
    FIXED_SIZE = "fixEncoding"  # This is not really in use anymore

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, __o) -> bool:
        return self is __o or self.value == __o

    def __hash__(self) -> int:
        return super().__hash__()


class Data_trans_type(Enum):
    """
    Represents data transformation types.

    Values:
        NONE: No transformation.
        LOG1P: Logarithmic (log1p) transformation.
        LOG1P_MAX: Log1p followed by max scaling.
        STANDARD: Standard scaling.
        MAX: Max scaling.
        BOX_COX: Box-Cox transformation.
        YEO_JOHNSON: Yeo-Johnson transformation.
    """

    NONE = "no_trans"
    LOG1P = "ln_x_plus_one_trans"
    LOG1P_MAX = "ln_x_plus_one_and_max_trans"
    STANDARD = "standard_trans"
    MAX = "max_trans"
    BOX_COX = "box_cox_trans"
    YEO_JOHNSON = "yeo_johnson_trans"

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, __o) -> bool:
        return self is __o or self.value == __o

    def __hash__(self) -> int:
        return super().__hash__()


class Model_task(Enum):
    """
    Represents machine learning task types.

    Values:
        CLASSIFICATION_TASK: Classification of positive/negative outcomes.
        REGRESSION_TASK: Regression for predicting numerical values.
    """

    CLASSIFICATION_TASK = "classification"
    REGRESSION_TASK = "regression"

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, __o) -> bool:
        return self is __o or self.value == __o

    def __hash__(self) -> int:
        return super().__hash__()


class Data_type(Enum):
    """
    Represents different CRISPR datasets.

    Values include CHANGEseq, GUIDEseq, and others.
    """

    CHANGE_SEQ = "CHANGEseq"
    GUIDE_SEQ = "GUIDEseq"
    NEW_GUIDE_SEQ = "NewGUIDEseq"
    FULL_GUIDE_SEQ = "FullGUIDEseq"

    RHAMP_SEQ = "RHAMPseq"  # RHAMP-seq from the CHANGE-seq study 
    REFINED_TURE_OT = "Refined_TrueOT"
    CHEN_2017_GUIDE_SEQ = "Chen_2017_GUIDE_seq"
    TSAI_2015_GUIDE_SEQ = "Tsai_2015_GUIDE_seq"
    LISTGARTEN_2018_GUIDE_SEQ = "Listgarten_2018_GUIDE_seq"
    CRISPR_NET = "CrisprNet"

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, __o) -> bool:
        return self is __o or self.value == __o

    def __hash__(self) -> int:
        return super().__hash__()


class Model_type(Enum):
    """
    Represents types of predictive models.

    Values include XGBoost, SVM, MLP, and others.
    """

    XGBOOST = "xgboost"
    SVM = "SVM"
    SVM_LINEAR = "SVM-linear"
    SGD = "SGD"
    MLP = "MLP"
    ADABOOST = "adaboost"
    RF = "rf"
    LASSO = "lasso"
    CATBOOST = "catboost"
    C_1 = "c_1"
    C_2 = "c_2"
    C_3 = "c_3"
    D_1 = "d_1"
    D_2 = "d_2"
    D_3 = "d_3"
    D2C_2 = "d2c_2"
    D2C_2_1 = "d2c_2_1"
    D2C_3 = "d2c_3"

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, __o) -> bool:
        return self is __o or self.value == __o

    def __hash__(self) -> int:
        return super().__hash__()


class Xgboost_tf_type(Enum):
    """
    Represents transfer learning modes for XGBoost models.

    Values:
        NONE: No transfer learning.
        ADD, UPDATE: See XGBoost API for details.
    """

    NONE = None
    ADD = "add"
    UPDATE = "update"

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, __o) -> bool:
        return self is __o or self.value == __o

    def __hash__(self) -> int:
        return super().__hash__()


# ##############general functions###############
def folder_num_gen():
    """
    Generator function to yield incrementing folder numbers as strings.
    """
    i = 1
    while True:
        yield str(i)
        i += 1


def generate_next_folder_name(prefix_path):
    """
    Generates the next available folder name with a numerical suffix within a specified prefix path.

    Args:
        prefix_path (str): The base path for the new folder.

    Returns:
        str: The generated folder path.
    """
    for folder_num in folder_num_gen():
        folder_name = prefix_path + "/" + str(folder_num)
        if not os.path.exists(folder_name):
            return folder_name


def parallelize_dataframe(df, func, n_cores=N_CORES):
    """
    Parallelizes the application of a function to a DataFrame across multiple cores.

    Args:
        df (pd.DataFrame): The input DataFrame.
        func (function): The function to apply in parallel.
        n_cores (int, optional): The number of cores to utilize. Defaults to all available cores.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    if isinstance(n_cores, int):
        df_split = np.array_split(df, n_cores)
        with ThreadPoolExecutor() as executor:
            result = executor.map(func, df_split)
        df = pd.concat(result)
    else:
        df = func(df)
        return df
