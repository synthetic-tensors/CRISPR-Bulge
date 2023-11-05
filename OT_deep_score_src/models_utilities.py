"""
This module contains the utilizes functions for training and training all the xgboost model variants
"""
import re
import random
import itertools
import pandas as pd
import numpy as np

from sklearn.preprocessing import PowerTransformer, FunctionTransformer, MaxAbsScaler, StandardScaler

from OT_deep_score_src import general_utilities
from OT_deep_score_src.general_utilities import Data_type, Data_trans_type, SEED, SG_RNA, OFF_TARGET, \
    DISTANCE, SG_RNA_SEQ, LABEL, Padding_type

from models.nuclea_seq_modeling.modeling import log10_crispr_specificity
from models.moff_modeling.modeling import GMT_score


random.seed(SEED)


def load_sg_rnas_list(data_type=Data_type.CHANGE_SEQ):
    """
    load and return the sgRNAs of data type
    """
    sg_rnas_s = pd.read_csv(
        general_utilities.DATASETS_PATH + '{}_sgRNAs_list.csv'.format(data_type),
        header=None).squeeze("columns")
    return list(sg_rnas_s)


def load_order_sg_rnas(data_type=Data_type.CHANGE_SEQ):
    """
    load and return the sgRNAs in certain order for the k-fold training
    """
    sg_rnas_s = pd.read_csv(
        general_utilities.DATASETS_PATH + '{}_sgRNAs_ordering.csv'.format(data_type),
        header=None).squeeze("columns")
    return list(sg_rnas_s)


def order_sg_rnas(data_type=Data_type.CHANGE_SEQ):
    """
    Create and return the sgRNAs in certain order for the k-fold training
    """
    if data_type not in [Data_type.CHANGE_SEQ,  Data_type.FULL_GUIDE_SEQ, Data_type.GUIDE_SEQ, Data_type.NEW_GUIDE_SEQ]:
        raise ValueError("Unsupported data type")

    dataset_df = pd.read_csv(
        general_utilities.DATASETS_PATH + '{}/include_on_targets/{}_CR_Lazzarotto_2020_dataset.csv'.format(
            data_type, data_type))
    sg_rnas = list(dataset_df[SG_RNA].unique())
    print("There are", len(sg_rnas), "unique sgRNAs in the", data_type, "dataset")

    # sort the sgRNAs and shuffle them
    sg_rnas.sort()
    random.shuffle(sg_rnas)

    # save the sgRNAs order into csv file
    sg_rnas_s = pd.Series(sg_rnas)
    # to csv - you can read this to Series using -
    # pd.read_csv("file_name.csv", header=None, squeeze=True)
    sg_rnas_s.to_csv(general_utilities.DATASETS_PATH + '{}_sgRNAs_ordering.csv'.format(data_type),
                     header=False, index=False)

    return sg_rnas


def create_nucleotides_to_position_mapping():
    """
    Return the nucleotides to position mapping
    """
    # matrix positions for ('A','A'), ('A','C'),...
    # tuples of ('A','A'), ('A','C'),...
    nucleotides_product = list(itertools.product(*(['ACGT-'] * 2)))
    # tuples of (0,0), (0,1), ...
    position_product = [(int(x[0]), int(x[1]))
                        for x in itertools.product(*(['01234'] * 2))]
    nucleotides_to_position_mapping = dict(
        zip(nucleotides_product, position_product))

    # tuples of ('N','A'), ('N','C'),...
    n_mapping_nucleotides_list = [('N', char) for char in ['A', 'C', 'G', 'T', '-']]
    # list of tuples positions corresponding to ('A','A'), ('C','C'), ...
    n_mapping_position_list = [nucleotides_to_position_mapping[(char, char)]
                               for char in ['A', 'C', 'G', 'T', '-']]

    nucleotides_to_position_mapping.update(
        dict(zip(n_mapping_nucleotides_list, n_mapping_position_list)))

    # tuples of ('A','N'), ('C','N'),...
    n_mapping_nucleotides_list = [(char, 'N') for char in ['A', 'C', 'G', 'T', '-']]
    # list of tuples positions corresponding to ('A','A'), ('C','C'), ...
    n_mapping_position_list = [nucleotides_to_position_mapping[(char, char)]
                               for char in ['A', 'C', 'G', 'T', '-']]
    nucleotides_to_position_mapping.update(
        dict(zip(n_mapping_nucleotides_list, n_mapping_position_list)))

    return nucleotides_to_position_mapping


def create_fixed_size_encoding_mapping():
    """
    Return the nucleotides to position mapping for the fixed siz encoding
    """
    sg_rna_off_target_nec_map = {("A", "A"): 0, ("A", "C"): 1, ("A", "G"): 2, ("A", "T"): 3, ("A", "-"): 4,
                                 ("C", "A"): 5, ("C", "C"): 6, ("C", "G"): 7, ("C", "T"): 8, ("C", "-"): 9,
                                 ("G", "A"): 10, ("G", "C"): 11, ("G", "G"): 12, ("G", "T"): 13, ("G", "-"): 14,
                                 ("T", "A"): 15, ("T", "C"): 16, ("T", "G"): 17, ("T", "T"): 18, ("T", "-"): 19,
                                 ("N", "A"): 0, ("N", "C"): 6, ("N", "G"): 12, ("N", "T"): 18,
                                 ("A", "N"): 0, ("C", "N"): 6, ("G", "N"): 12, ("T", "N"): 18
                                 }

    rna_bugle_map = {("-", "A"): 0, ("-", "C"): 1, ("-", "G"): 2, ("-", "T"): 3}

    return sg_rna_off_target_nec_map, rna_bugle_map


def one_hot_encoding(dataset_df, n_samples, seq_len, nucleotide_num):
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()

    one_hot_arr = np.zeros((n_samples, seq_len, nucleotide_num, nucleotide_num), dtype=np.int8)
    for i, (sg_rna_seq, off_seq) in enumerate(zip(dataset_df[SG_RNA_SEQ], dataset_df[OFF_TARGET])):
        if len(off_seq) != len(sg_rna_seq):
            raise ValueError("len(off_seq) != len(sg_rna_seq)")
        actual_seq_size = len(off_seq)
        if actual_seq_size > seq_len:
            raise ValueError("actual_seq_size > seq_len")

        size_diff = seq_len - actual_seq_size
        for j in range(seq_len):
            if j >= size_diff:
                # note that it is important to take (sg_rna_seq_j, off_seq_j) as old models did the same.
                matrix_positions = nucleotides_to_position_mapping[(sg_rna_seq[j-size_diff], off_seq[j-size_diff])]
                one_hot_arr[i, j, matrix_positions[0], matrix_positions[1]] = 1
    # reshape to [n_samples, seq_len, nucleotide_num**2]
    one_hot_arr = one_hot_arr.reshape((n_samples, seq_len, nucleotide_num**2))

    return one_hot_arr


def flat_one_hot_encoding(dataset_df, n_samples, include_sequence_features,
                          include_distance_feature, seq_len, nucleotide_num):
    if include_sequence_features:
        final_result = one_hot_encoding(
            dataset_df=dataset_df, n_samples=n_samples, seq_len=seq_len, nucleotide_num=nucleotide_num)
        final_result = final_result.reshape(n_samples, -1)
        if include_distance_feature:
            final_result = np.concatenate((final_result, np.expand_dims(dataset_df[DISTANCE].values, 1)), axis=1)
    else:
        # else - include_distance_feature must be True
        final_result = np.expand_dims(dataset_df[DISTANCE].values, 1)

    return final_result


def fixed_size_encoding_fun(sg_rna, off_target, sg_rna_off_target_nec_map, rna_bugle_map):
    sg_rna_gaps_positions = [m.start() for m in re.finditer("-", sg_rna)]
    # trim the sequences and remove position with gap in the sgRNA sequence
    sg_rna_trimmed = "".join([char for (i, char) in enumerate(sg_rna) if i not in sg_rna_gaps_positions])
    off_target_trimmed = "".join([char for (i, char) in enumerate(off_target) if i not in sg_rna_gaps_positions])

    # encode the trimmed parts
    alignment_encoding = np.zeros((23, 20), dtype=np.int8)
    for (i, (sg_rna_char, off_target_char)) in enumerate(zip(sg_rna_trimmed, off_target_trimmed)):
        alignment_encoding[i, sg_rna_off_target_nec_map[(sg_rna_char, off_target_char)]] = 1

    # encode the RNA bulges
    rna_bulges_encoding = np.zeros((23, 4), dtype=np.int8)
    for sg_rna_gaps_pos in sg_rna_gaps_positions:
        rna_bulges_encoding[sg_rna_gaps_pos-1, rna_bugle_map["-", (off_target[sg_rna_gaps_pos])]] = 1

    return np.concatenate((alignment_encoding, rna_bulges_encoding), axis=1)


def flat_fixed_size_encdoing(dataset_df, n_samples, include_sequence_features, include_distance_feature):
    sg_rna_off_target_nec_map, rna_bugle_map = create_fixed_size_encoding_mapping()
    if include_sequence_features:
        final_result = np.zeros((n_samples, 23*24+1),
                                dtype=np.int8) if include_distance_feature else \
            np.zeros((n_samples, 23*24), dtype=np.int8)
    else:
        final_result = np.zeros((n_samples, 1), dtype=np.int8)
    for i, (seq1, seq2) in enumerate(zip(dataset_df[SG_RNA_SEQ], dataset_df[OFF_TARGET])):
        if include_sequence_features:
            if include_distance_feature:
                final_result[i, :-1] = fixed_size_encoding_fun(
                    seq1, seq2, sg_rna_off_target_nec_map, rna_bugle_map).flatten()
                final_result[i, -1] = dataset_df[DISTANCE].iloc[i]
            else:
                final_result[i] = fixed_size_encoding_fun(
                    seq1, seq2, sg_rna_off_target_nec_map, rna_bugle_map).flatten()
        else:
            # if include_sequence_features is False then include_distance_feature must be True
            final_result[i, -1] = dataset_df[DISTANCE].iloc[i]

    return final_result


def nuclea_seq_score_prediction(sg_rna, off_target):
    """
    Nuclea-seq model score prediction for "WT"
    """
    # TODO: this is only a plaster, I'm not sure how the handle the gap in the sgRNA.
    # sg_rna_without_gaps = sg_rna.replace("-", "")
    # assume N in the PAM is the corresponding nucleotide in the off-target sequence
    # pam_seq = off_target[-3] + sg_rna[-2:] if sg_rna[-3] == "N" else sg_rna_without_gaps[-3:] - wrong
    pam_seq = "T" + off_target[-2:]  # Nuclea-seq was trained only with TGG
    return log10_crispr_specificity('WT', pam_seq, sg_rna[:-3], off_target[:-3])


def gmt_score_prediction(dataset_df):
    """
    assign MOFF model GMT score prediction
    """
    dataset_df = dataset_df.rename({SG_RNA: "sgRNA", OFF_TARGET: "off-target"}, axis=1)
    dataset_moff_df = dataset_df[["sgRNA", "off-target"]].drop_duplicates(subset=['sgRNA'])
    dataset_moff_df["off-target"] = dataset_moff_df["sgRNA"].values
    dataset_moff_df = GMT_score(dataset_moff_df)

    dataset_df["GMT"] = np.nan
    for sg_rna in dataset_df["sgRNA"].unique():
        dataset_df.loc[dataset_df["sgRNA"] == sg_rna, ['GMT']] = \
            dataset_moff_df.loc[dataset_moff_df["sgRNA"] == sg_rna, "GOP"].values[0]

    return dataset_df["GMT"].values


def remove_gaps(df):
    df[SG_RNA_SEQ] = df[SG_RNA_SEQ].str.replace("-", "")
    df[OFF_TARGET] = df[OFF_TARGET].str.replace("-", "")


def left_pad_sequence(df, seq_len=24, ch="-"):
    df[SG_RNA_SEQ] = df[SG_RNA_SEQ].str.rjust(seq_len, ch)
    df[OFF_TARGET] = df[OFF_TARGET].str.rjust(seq_len, ch)


def build_sequence_features_constraints(
        bulges, aligned, padding_type, include_distance_feature, include_sequence_features, fixed_size_encoding):
    """
    confirm some constraints
    """
    if not bulges and (not aligned or padding_type != Padding_type.NONE):
        raise ValueError("When bulges is False, then aligned must be True and there should not be any padding")

    if (not include_distance_feature) and (not include_sequence_features):
        raise ValueError("include_distance_feature and include_sequence_features can not be both False")
    if fixed_size_encoding and not aligned:
        raise ValueError("fixed size encoding must be applied when sequecnes are aligned")


def build_sequence_features_initial_set(dataset_df, bulges, padding_type, aligned, convert_to_n, seq_len):
    if bulges:
        nucleotide_num = 5
        seq_len = 24 if seq_len is None else seq_len  # for now, it is just of len of 24
    else:
        nucleotide_num = 4
        seq_len = 23 if seq_len is None else seq_len
    n_samples = len(dataset_df)
    if convert_to_n:
        # convert dataset_df[sg_rna_name] -3 position to 'N'
        print("Converting the [-3] positions in each sgRNA sequence to 'N'")
        dataset_df[SG_RNA_SEQ] = dataset_df[SG_RNA_SEQ].apply(lambda s: s[:-3] + 'N' + s[-2:])

    if not aligned:
        remove_gaps(dataset_df)
        # dataset_df[SG_RNA_SEQ] = dataset_df.apply(lambda row: remove_gaps(row[SG_RNA_SEQ]), axis=1)
        # dataset_df[OFF_TARGET] = dataset_df.apply(lambda row: remove_gaps(row[OFF_TARGET]), axis=1)

    if padding_type == Padding_type.GAP:
        left_pad_sequence(dataset_df)
        # dataset_df[SG_RNA_SEQ] = dataset_df.apply(lambda row: left_pad_sequence(row[SG_RNA_SEQ]), axis=1)
        # dataset_df[OFF_TARGET] = dataset_df.apply(lambda row: left_pad_sequence(row[OFF_TARGET]), axis=1)

    return dataset_df, n_samples, seq_len, nucleotide_num


def build_sequence_features(dataset_df,
                            include_distance_feature=False,
                            include_sequence_features=True,
                            include_gmt_score=False,
                            include_nuclea_seq_score=False,
                            bulges=False,
                            padding_type=Padding_type.NONE,
                            aligned=True,
                            fixed_size_encoding=False,
                            convert_to_n=False,
                            flat_encoding=True,
                            seq_len=None
                            ):
    """
    Build sequence features using the nucleotides to position mapping
    """
    build_sequence_features_constraints(
        bulges, aligned, padding_type, include_distance_feature,
        include_sequence_features, fixed_size_encoding)

    dataset_df, n_samples, seq_len, nucleotide_num = build_sequence_features_initial_set(
        dataset_df, bulges, padding_type, aligned, convert_to_n, seq_len)

    if flat_encoding:
        if fixed_size_encoding:
            final_result = flat_fixed_size_encdoing(dataset_df, n_samples, include_sequence_features,
                                                    include_distance_feature)
        else:
            final_result = flat_one_hot_encoding(
                dataset_df=dataset_df, n_samples=n_samples, include_sequence_features=include_sequence_features,
                include_distance_feature=include_distance_feature, seq_len=seq_len, nucleotide_num=nucleotide_num)

        # GMT and Nuclea-seq
        if include_gmt_score:
            gmt_scores = gmt_score_prediction(dataset_df.copy())
            final_result = np.concatenate((final_result, np.expand_dims(gmt_scores, 1)), axis=1)
        if include_nuclea_seq_score:
            nuclea_seq_score = dataset_df.apply(
                lambda row: nuclea_seq_score_prediction(row[SG_RNA_SEQ], row[OFF_TARGET]), axis=1).values
            final_result = np.concatenate((final_result, np.expand_dims(nuclea_seq_score, 1)), axis=1)
    else:
        if fixed_size_encoding:
            raise NotImplementedError()
        else:
            final_result = one_hot_encoding(
                dataset_df=dataset_df, n_samples=n_samples, seq_len=seq_len, nucleotide_num=nucleotide_num)

        scores = []
        if include_distance_feature:
            distances = dataset_df[DISTANCE].values
            # transformer = transformer_generator(data=distances, trans_type=Data_trans_type.STANDARD)
            # distances = transform(data=distances, transformer=transformer)
            scores.append(distances)
        if include_gmt_score:
            scores.append(gmt_score_prediction(dataset_df.copy()))
        if include_nuclea_seq_score:
            scores.append(dataset_df.apply(lambda row: nuclea_seq_score_prediction(
                row[SG_RNA_SEQ], row[OFF_TARGET]), axis=1).values)
        if scores:
            # make sure that scores items are of shape [n_samples, #features]
            scores = np.concatenate([x[:, None] for x in scores], axis=1)
            final_result = [final_result, scores]

    print("The features sizes are")
    if not isinstance(final_result, list):
        print(final_result.shape)
    else:
        for feature in final_result:
            print(feature.shape)
    return final_result


##########################################################################
def create_fold_sets(target_fold, targets, dataset_df,
                     exclude_sg_rnas_without_positives):
    """
    Create fold sets for train/test
    remove_targets_without_positives: only from the train test.
        It doesn't matter in the test set, as we can't evaluate the performance when evaluating per sgRNA.
        Moreover, we can always remove them in the evaluation stage.
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


##########################################################################
def build_sampleweight(y_values):
    """
    Sample weight according to class
    """
    vec = np.zeros((len(y_values)))
    for values_class in np.unique(y_values):
        vec[y_values == values_class] = np.sum(
            y_values != values_class) / len(y_values)
    return vec


##########################################################################
def transformer_generator(data, trans_type):
    """
    Create create data transformer
    """
    data = data.reshape(-1, 1)
    if trans_type == Data_trans_type.NONE:
        # identity transformer
        transformer = FunctionTransformer()
    elif trans_type == Data_trans_type.LOG1P:
        transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
    elif trans_type == Data_trans_type.LOG1P_MAX:
        transformer_1 = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        data = transformer_1.transform(data)
        transformer_2 = MaxAbsScaler()
        transformer_2.fit(data)  # type: ignore
        transformer = (transformer_1, transformer_2)
    elif trans_type == Data_trans_type.STANDARD:
        transformer = StandardScaler()
        transformer.fit(data)
    elif trans_type == Data_trans_type.MAX:
        transformer = MaxAbsScaler()
        transformer.fit(data)
    elif trans_type == Data_trans_type.BOX_COX:
        if np.all(data == data[0]):
            # if the input data is constant, the return identity transformer
            print("identity transformer (instead of box-cox) was returned since the input data is constant")
            transformer = FunctionTransformer()
        else:
            # we balance the negatives and positives and then fit the transformation.
            data = data[data > 0]
            data = data.reshape(-1, 1)
            data = np.concatenate([data, np.zeros(data.shape)])
            # we perform box-cox on data+1
            transformer_1 = FunctionTransformer(func=lambda x: x + 1, inverse_func=lambda x: x - 1)
            data = transformer_1.transform(data)
            transformer_2 = PowerTransformer(method='box-cox')
            transformer_2.fit(data)  # type: ignore
            transformer = (transformer_1, transformer_2)
    elif trans_type == Data_trans_type.YEO_JOHNSON:
        if np.all(data == data[0]):
            # if the input data is constant, the return identity transformer
            print("identity transformer (instead of yeo-johnson) was returned since the input data is constant")
            transformer = FunctionTransformer()
        else:
            # we balance the negatives and positives and then fit the transformation.
            data = data[data > 0]
            data = data.reshape(-1, 1)
            data = np.concatenate([data, np.zeros(data.shape)])
            transformer = PowerTransformer(method='yeo-johnson')
            transformer.fit(data)
    else:
        raise ValueError("Invalid trans_type")

    return transformer


def transform(data, transformer, inverse=False):
    """
    transform function
    """
    data = data.reshape(-1, 1)
    if not isinstance(transformer, (list, tuple)):
        transformer = [transformer]
    if not inverse:
        for transformer_i in transformer:
            data = transformer_i.transform(data)
    else:
        for transformer_i in transformer[::-1]:
            data = transformer_i.inverse_transform(data)

    return np.squeeze(data)
