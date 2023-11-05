"""
general utilizes
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
CHANGE_SEQ_PATH = DATASETS_PATH + "CHANGEseq.xlsx"
GUIDE_SEQ_PATH = DATASETS_PATH + "GUIDEseq.xlsx"


# #################constants##################
SEED = 10

# SG_RNA = "sgRNA"
# SG_RNA_SEQ = "Alignment sgRNA"
# OFF_TARGET = "Alignment off-target"
# DISTANCE = "Alignment distance"
# CHROM = "Chromosome"
# CROM_START_LOCATION = "Location"
# READS = "reads"
# LABEL = "label"
# MISMATCHES = "Alignment Mismatches"
# BULGES = "Alignment Bulge Size"
# MAX_DISTANCE = 6


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
    NONE = None
    GAP = "gap_pad"
    # ZERO = "zero_pad"

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, __o) -> bool:
        return self is __o or self.value == __o

    def __hash__(self) -> int:
        return super().__hash__()


class Data_trans_type(Enum):
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
    CLASSIFICATION_TASK = "classification"
    REGRESSION_TASK = "regression"

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, __o) -> bool:
        return self is __o or self.value == __o

    def __hash__(self) -> int:
        return super().__hash__()


class Data_type(Enum):
    CHANGE_SEQ = "CHANGEseq"
    GUIDE_SEQ = "GUIDEseq"
    TRUE_OT = "TrueOT"
    RHAMP_SEQ = "RHAMPseq"
    NEW_GUIDE_SEQ = "NewGUIDEseq"
    FULL_GUIDE_SEQ = "FullGUIDEseq"

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, __o) -> bool:
        return self is __o or self.value == __o

    def __hash__(self) -> int:
        return super().__hash__()


class Model_type(Enum):
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
    Xgboost "transfer learning" type
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
    i = 1
    while True:
        yield str(i)
        i += 1


def generate_next_folder_name(prefix_path):
    for folder_num in folder_num_gen():
        folder_name = prefix_path + "/" + str(folder_num)
        if not os.path.exists(folder_name):
            return folder_name


def parallelize_dataframe(df, func, n_cores=N_CORES):
    if isinstance(n_cores, int):
        df_split = np.array_split(df, n_cores)
        with ThreadPoolExecutor() as executor:
            result = executor.map(func, df_split)
        df = pd.concat(result)
    else:
        df = func(df)
        return df
