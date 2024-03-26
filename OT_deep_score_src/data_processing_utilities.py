"""
This module contains the utilizes functions for data processing of the data (creating features and labels)
"""
import re
import itertools
import numpy as np

from sklearn.preprocessing import PowerTransformer, FunctionTransformer, MaxAbsScaler, StandardScaler

from OT_deep_score_src.general_utilities import Data_trans_type, OFF_TARGET, \
    DISTANCE, SG_RNA_SEQ, Padding_type, Encoding_type

from OT_deep_score_src.models_utilities import gmt_score_prediction, nuclea_seq_score_prediction


def create_nucleotides_to_encoding_mapping_crispr_net():
    """
    Creates a mapping of nucleotide pairs (sgRNA, off-target) to positions for the CRISPR-Net encoding scheme.

    Returns:
        dict: A dictionary where keys are tuples of nucleotides ("A", "T"), ("G", "C"), etc.,
              and values are their corresponding 7-dimensional encoding vectors.
    """
    nucleotides_to_encoding_mapping = {
        ("A", "A"): np.array([1, 0, 0, 0, 0, 0, 0]),
        ("A", "T"): np.array([1, 1, 0, 0, 0, 1, 0]),
        ("A", "G"): np.array([1, 0, 1, 0, 0, 0, 0]),
        ("A", "C"): np.array([1, 0, 0, 1, 0, 0, 0]),
        ("A", "-"): np.array([1, 0, 0, 0, 1, 0, 0]),
        ("A", "N"): np.array([1, 0, 0, 0, 0, 0, 0]),

        ("T", "A"): np.array([1, 1, 0, 1, 0, 0, 1]),
        ("T", "T"): np.array([0, 0, 0, 1, 0, 0, 0]),
        ("T", "G"): np.array([0, 1, 1, 0, 0, 0, 1]),
        ("T", "C"): np.array([0, 1, 0, 1, 0, 0, 1]),
        ("T", "-"): np.array([0, 1, 0, 0, 1, 1, 0]),
        ("T", "N"): np.array([0, 0, 0, 1, 0, 0, 0]),

        ("G", "A"): np.array([1, 0, 1, 0, 0, 0, 1]),
        ("G", "T"): np.array([0, 1, 1, 0, 0, 1, 0]),
        ("G", "G"): np.array([0, 0, 1, 0, 0, 0, 0]),
        ("G", "C"): np.array([0, 0, 1, 1, 0, 1, 0]),
        ("G", "-"): np.array([0, 0, 1, 0, 1, 1, 0]),
        ("G", "N"): np.array([0, 0, 1, 0, 0, 0, 0]),

        ("C", "A"): np.array([1, 0, 0, 1, 0, 0, 1]),
        ("C", "T"): np.array([0, 1, 0, 1, 0, 1, 0]),
        ("C", "G"): np.array([0, 0, 1, 1, 0, 0, 1]),
        ("C", "C"): np.array([0, 0, 0, 1, 0, 0, 0]),
        ("C", "-"): np.array([0, 0, 0, 1, 1, 1, 0]),
        ("C", "N"): np.array([0, 0, 0, 1, 0, 0, 0]),

        ("-", "A"): np.array([1, 0, 0, 0, 1, 0, 1]),
        ("-", "T"): np.array([0, 1, 0, 0, 1, 0, 1]),
        ("-", "G"): np.array([0, 0, 1, 0, 1, 0, 1]),
        ("-", "C"): np.array([0, 0, 0, 1, 1, 0, 1]),
        ("-", "-"): np.array([0, 0, 0, 0, 0, 0, 0]),
        ("-", "N"): np.array([0, 0, 0, 0, 0, 0, 0]),

        ("N", "A"): np.array([1, 0, 0, 0, 0, 0, 0]),
        ("N", "T"): np.array([0, 0, 0, 1, 0, 0, 0]),
        ("N", "G"): np.array([0, 0, 1, 0, 0, 0, 0]),
        ("N", "C"): np.array([0, 0, 0, 1, 0, 0, 0]),
        ("N", "-"): np.array([0, 0, 0, 0, 0, 0, 0]),
        ("N", "N"): np.array([0, 0, 0, 0, 0, 0, 0])
    }

    return nucleotides_to_encoding_mapping


def create_nucleotides_to_position_mapping():
    """
    Creates a mapping of nucleotide pairs (sgRNA, off-target) to their numerical positions.
    This mapping includes positions for "N" nucleotides (representing any nucleotide).

    Returns:
        dict: A dictionary where keys are tuples of nucleotides ("A", "T"), ("G", "C"), etc.,
              and values are tuples representing their (row, column) positions in a matrix.
    """
    # matrix positions for ("A","A"), ("A","C"),...
    # tuples of ("A","A"), ("A","C"),...
    nucleotides_product = list(itertools.product(*(["ACGT-"] * 2)))
    # tuples of (0,0), (0,1), ...
    position_product = [(int(x[0]), int(x[1]))
                        for x in itertools.product(*(["01234"] * 2))]
    nucleotides_to_position_mapping = dict(
        zip(nucleotides_product, position_product))

    # tuples of ("N","A"), ("N","C"),...
    n_mapping_nucleotides_list = [("N", char) for char in ["A", "C", "G", "T", "-"]]
    # list of tuples positions corresponding to ("A","A"), ("C","C"), ...
    n_mapping_position_list = [nucleotides_to_position_mapping[(char, char)]
                               for char in ["A", "C", "G", "T", "-"]]

    nucleotides_to_position_mapping.update(
        dict(zip(n_mapping_nucleotides_list, n_mapping_position_list)))

    # tuples of ("A","N"), ("C","N"),...
    n_mapping_nucleotides_list = [(char, "N") for char in ["A", "C", "G", "T", "-"]]
    # list of tuples positions corresponding to ("A","A"), ("C","C"), ...
    n_mapping_position_list = [nucleotides_to_position_mapping[(char, char)]
                               for char in ["A", "C", "G", "T", "-"]]
    nucleotides_to_position_mapping.update(
        dict(zip(n_mapping_nucleotides_list, n_mapping_position_list)))

    return nucleotides_to_position_mapping


def create_fixed_size_encoding_mapping():
    """
    Creates mappings for the the fixed-size encoding scheme.

    Returns:
        tuple:
            * sg_rna_off_target_nec_map (dict): Maps nucleotide pairs to indices for sequence alignment mismatches.
            * rna_bugle_map (dict): Maps RNA bulges (gaps) to indices.
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


def crispr_net_encoding(dataset_df, n_samples, seq_len):
    """
    Generates CRISPR-Net encodings for a dataset of sgRNA and off-target sequences.

    Args:
        dataset_df (pd.DataFrame): Dataset Dataframe. Must contain columns "SG_RNA_SEQ" and "OFF_TARGET".
        n_samples (int): Number of samples in the dataset.
        seq_len (int): Length of the sequences.

    Returns:
        numpy.ndarray: A 3D array of shape (n_samples, seq_len, 7) representing the encodings.
    """
    nucleotides_to_position_mapping = create_nucleotides_to_encoding_mapping_crispr_net()

    encoding_arr = np.zeros((n_samples, seq_len, 7), dtype=np.int8)
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
                encoding_arr[i, j, :] = nucleotides_to_position_mapping[(sg_rna_seq[j-size_diff], off_seq[j-size_diff])]

    return encoding_arr


def flat_crispr_net_encoding(dataset_df, n_samples, include_sequence_features, include_distance_feature, seq_len):
    """
    Generates flattened CRISPR-Net encodings, optionally including distance features.

    Args:
        dataset_df (pd.DataFrame): Dataset Dataframe. Must contain columns "SG_RNA_SEQ" and "OFF_TARGET".
        n_samples (int): Number of samples in the dataset.
        include_sequence_features (bool): If True, includes sequence encoding features.
        include_distance_feature (bool): If True, includes the distance feature
            (dataset_df Must contain columns "DISTANCE").
        seq_len (int): Length of the sequences.

    Returns:
        numpy.ndarray: A flattened array representing the encodings.
    """
    if include_sequence_features:
        final_result = crispr_net_encoding(dataset_df=dataset_df, n_samples=n_samples, seq_len=seq_len)
        final_result = final_result.reshape(n_samples, -1)
        if include_distance_feature:
            final_result = np.concatenate((final_result, np.expand_dims(dataset_df[DISTANCE].values, 1)), axis=1)
    else:
        # else - include_distance_feature must be True
        final_result = np.expand_dims(dataset_df[DISTANCE].values, 1)

    return final_result


def one_hot_encoding(dataset_df, n_samples, seq_len, nucleotide_num):
    """
    Creates a one-hot encoding of sgRNA and off-target sequences.

    Args:
        dataset_df (pd.DataFrame): Dataset Dataframe. Must contain columns "SG_RNA_SEQ" and "OFF_TARGET".
        n_samples (int): Total number of samples in the dataset.
        seq_len (int): Length of the sequences.
        nucleotide_num (int): Number of distinct nucleotides (5 when including bulges).

    Returns:
        np.ndarray: One-hot encoded array, shape: (n_samples, seq_len, nucleotide_num ** 2)
    """
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


def flat_one_hot_encoding(
        dataset_df, n_samples, include_sequence_features, include_distance_feature, seq_len, nucleotide_num):
    """
    Generates a flattened one-hot encoding of DNA sequences, optionally including a distance feature.

    Args:
        dataset_df (pd.DataFrame): Dataset Dataframe. Must contain columns "SG_RNA_SEQ" and "OFF_TARGET".
        n_samples (int): Total number of samples.
        include_sequence_features (bool): If True, includes sequence encoding features.
        include_distance_feature (bool): If True, includes the distance feature
            (dataset_df Must contain columns "DISTANCE").
        seq_len (int):  Length of the sequences.
        nucleotide_num (int): Number of distinct nucleotides (5 when including bulges).

    Returns:
        np.ndarray: Flattened array, potentially including the distance feature.
    """
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
    """
    Creates the fixed-size encoding of sgRNA and off-target sequences.

    Args:
        sg_rna (str): The sgRNA sequence.
        off_target (str): The off-target sequence.
        sg_rna_off_target_nec_map (dict): Maps nucleotide pairs to indices (for alignment encoding).
        rna_bugle_map (dict): Maps RNA bulge types to indices.

    Returns:
        np.ndarray: The fixed-size encoded representation.
    """
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
    """
    Creates a flattened, fixed-size encoding representation of DNA sequences, optionally including a distance feature.

    Args:
        dataset_df (pd.DataFrame): Dataset Dataframe. Must contain columns "SG_RNA_SEQ" and "OFF_TARGET".
        n_samples (int): Total number of samples.
        include_sequence_features (bool): If True, includes sequence encoding features.
        include_distance_feature (bool): If True, includes the distance feature
            (dataset_df Must contain columns "DISTANCE").

    Returns:
        np.ndarray: The flattened, fixed-size encoded representation.
    """
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


def remove_gaps(df):
    """
    Removes gap characters ("-") from "SG_RNA_SEQ" and "OFF_TARGET" columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
    """
    df[SG_RNA_SEQ] = df[SG_RNA_SEQ].str.replace("-", "")
    df[OFF_TARGET] = df[OFF_TARGET].str.replace("-", "")


def left_pad_sequence(df, seq_len=24, ch="-"):
    """
    Left-pads sequences in the "SG_RNA_SEQ" and "OFF_TARGET" columns of a DataFrame,
    extending them to a specified length.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        seq_len (int, optional): The desired sequence length. Defaults to 24.
        ch (str, optional): The character to use for padding. Defaults to "-".
    """
    df[SG_RNA_SEQ] = df[SG_RNA_SEQ].str.rjust(seq_len, ch)
    df[OFF_TARGET] = df[OFF_TARGET].str.rjust(seq_len, ch)


def build_sequence_features_constraints(
        bulges, aligned, padding_type, include_distance_feature, include_sequence_features, encoding_type):
    """
    Validates parameters used for building sequence features.

    Args:
        bulges (bool): Whether includes bulge data.
        aligned (bool): Whether sequences are aligned.
        padding_type (Padding_type): The type of padding used.
        include_distance_feature (bool): Whether to include the distance feature.
        include_sequence_features (bool): Whether to include sequence features.
        encoding_type (Encoding_type): The type of sequence encoding.

    Raises:
         ValueError: If invalid combinations of parameters are provided.
    """
    # TODO: confirm some constraints
    if not isinstance(encoding_type, Encoding_type):
        raise ValueError("encoding_type is not one of the valid encoding types")
    if not isinstance(padding_type, Padding_type):
        raise ValueError("padding_type is not one of the valid padding types")
    if not bulges and (not aligned or padding_type != Padding_type.NONE):
        raise ValueError("When bulges is False, then aligned must be True and there should not be any padding")
    if (not include_distance_feature) and (not include_sequence_features):
        raise ValueError("include_distance_feature and include_sequence_features can not be both False")
    if encoding_type == Encoding_type.FIXED_SIZE and not aligned:
        raise ValueError("fixed size encoding must be applied when sequecnes are aligned")
    if encoding_type == Encoding_type.CRISPR_NET and not bulges:
        raise ValueError("encoding_type cannot be CRISPR-Net encoding and without bulges encoding")


def build_sequence_features_initial_set(dataset_df, bulges, padding_type, aligned, convert_to_n, seq_len):
    """
    Prepares the dataset and sets parameters for feature encoding.

    Args:
        dataset_df (pd.DataFrame): Dataset DataFrame containing "SG_RNA_SEQ" and "OFF_TARGET" columns.
        bulges (bool): Whether to account for bulges in the encoding.
        padding_type (Padding_type): Specifies the type of padding to apply.
        aligned (bool):  Whether the sequences are aligned.
        convert_to_n (bool): Whether to convert the -3 position in sgRNA sequences to "N".
        seq_len (int):  The expected sequence length (used if not provided elsewhere).

    Returns:
        tuple:
            * dataset_df (pd.DataFrame): The potentially modified DataFrame
            * n_samples (int): Number of samples in the dataset.
            * seq_len (int): The final sequence length.
            * nucleotide_num (int): Number of distinct nucleotides.
    """

    if bulges:
        nucleotide_num = 5
        seq_len = 24 if seq_len is None else seq_len  # for now, it is just of len of 24
    else:
        nucleotide_num = 4
        seq_len = 23 if seq_len is None else seq_len
    n_samples = len(dataset_df)
    if convert_to_n:
        # convert dataset_df[sg_rna_name] -3 position to "N"
        print("Converting the [-3] positions in each sgRNA sequence to 'N'")
        dataset_df[SG_RNA_SEQ] = dataset_df[SG_RNA_SEQ].apply(lambda s: s[:-3] + "N" + s[-2:])

    if not aligned:
        remove_gaps(dataset_df)

    if padding_type == Padding_type.GAP:
        left_pad_sequence(dataset_df)

    return dataset_df, n_samples, seq_len, nucleotide_num


def build_sequence_features(dataset_df,
                            include_distance_feature=False,
                            include_sequence_features=True,
                            include_gmt_score=False,
                            include_nuclea_seq_score=False,
                            bulges=False,
                            padding_type=Padding_type.NONE,
                            aligned=True,
                            encoding_type=Encoding_type.ONE_HOT,
                            convert_to_n=False,
                            flat_encoding=True,
                            seq_len=None,
                            verbose=1
                            ):
    """
    Constructs sequence features, optionally including distance, GMT score, and Nuclea-seq score.

    Args:
        dataset_df (pd.DataFrame):  Dataset DataFrame containing "SG_RNA_SEQ", "OFF_TARGET",
                                    and potentially "DISTANCE" columns.
        include_distance_feature (bool, optional): Whether to include the distance feature. Defaults to False.
        include_sequence_features (bool, optional): Whether to include sequence features. Defaults to True.
        include_gmt_score (bool, optional): Whether to include the GMT score. Defaults to False.
        include_nuclea_seq_score (bool, optional): Whether to include the Nuclea-seq score. Defaults to False.
        bulges (bool, optional): Whether dataset include bulges. Defaults to False.
        padding_type (Padding_type, optional): Type of padding. Defaults to Padding_type.NONE.
        aligned (bool, optional): Whether sequences are aligned. Defaults to True.
        encoding_type (Encoding_type, optional): Type of sequence encoding. Defaults to Encoding_type.ONE_HOT.
        convert_to_n (bool, optional): Whether to convert the -3 position in sgRNA sequences to "N". Defaults to False.
        flat_encoding (bool, optional): Whether to use flattened encoding. Defaults to True.
        seq_len (int, optional): Expected sequence length. Defaults to None.
        verbose (int, optional): Controls verbosity of output. Defaults to 1.

    Returns:
        np.ndarray or list: The encoded features. If multiple feature types are included, a list is returned.
    """
    build_sequence_features_constraints(
        bulges, aligned, padding_type, include_distance_feature,
        include_sequence_features, encoding_type)

    dataset_df, n_samples, seq_len, nucleotide_num = build_sequence_features_initial_set(
        dataset_df, bulges, padding_type, aligned, convert_to_n, seq_len)

    if flat_encoding:
        if encoding_type == Encoding_type.FIXED_SIZE:
            final_result = flat_fixed_size_encdoing(dataset_df, n_samples, include_sequence_features,
                                                    include_distance_feature)
        elif encoding_type == Encoding_type.ONE_HOT:
            final_result = flat_one_hot_encoding(
                dataset_df=dataset_df, n_samples=n_samples, include_sequence_features=include_sequence_features,
                include_distance_feature=include_distance_feature, seq_len=seq_len, nucleotide_num=nucleotide_num)
        else:
            # encoding_type == Encoding_type.CRISPR_NET
            final_result = flat_crispr_net_encoding(
                dataset_df=dataset_df, n_samples=n_samples, include_sequence_features=include_sequence_features,
                include_distance_feature=include_distance_feature, seq_len=seq_len)

        # GMT and Nuclea-seq
        if include_gmt_score:
            gmt_scores = gmt_score_prediction(dataset_df.copy())
            final_result = np.concatenate((final_result, np.expand_dims(gmt_scores, 1)), axis=1)
        if include_nuclea_seq_score:
            nuclea_seq_score = dataset_df.apply(
                lambda row: nuclea_seq_score_prediction(row[SG_RNA_SEQ], row[OFF_TARGET]), axis=1).values
            final_result = np.concatenate((final_result, np.expand_dims(nuclea_seq_score, 1)), axis=1)
    else:
        if encoding_type == Encoding_type.FIXED_SIZE:
            raise NotImplementedError()
        elif encoding_type == Encoding_type.ONE_HOT:
            final_result = one_hot_encoding(
                dataset_df=dataset_df, n_samples=n_samples, seq_len=seq_len, nucleotide_num=nucleotide_num)
        else:
            # encoding_type == Encoding_type.CRISPR_NET
            final_result = crispr_net_encoding(
                dataset_df=dataset_df, n_samples=n_samples, seq_len=seq_len)

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

    if verbose == 1:
        print("The features sizes are")
        if not isinstance(final_result, list):
            print(final_result.shape)
        else:
            for feature in final_result:
                print(feature.shape)
    return final_result


##########################################################################
def build_sampleweight(y_values):
    """
    Calculates sample weights to balance data based on class distribution.

    Args:
        y_values (np.ndarray): An array containing class labels for samples.

    Returns:
        np.ndarray: An array of sample weights, where the weight of a sample is inversely proportional
                    to the frequency of its class.
    """
    vec = np.zeros((len(y_values)))
    for values_class in np.unique(y_values):
        vec[y_values == values_class] = np.sum(
            y_values != values_class) / len(y_values)
    return vec


##########################################################################
def transformer_generator(data, trans_type):
    """
    Creates a data transformer based on the specified transformation type.

    Args:
        data (np.ndarray): The input data (as a 1D array) to fit the transformer to.
        trans_type (Data_trans_type): The type of transformation desired.
            Options include:
                * "NONE": No transformation.
                * "LOG1P": Logarithmic (log1p) transformation.
                * "LOG1P_MAX": Logarithmic (log1p) followed by max absolute scaling.
                * "STANDARD": Standard scaling (centering and scaling to unit variance).
                * "MAX": Max absolute scaling.
                * "BOX_COX": Box-Cox transformation.
                * "YEO_JOHNSON": Yeo-Johnson transformation.

    Returns:
        Transformer: A fitted transformer object, ready to transform data.
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
            transformer_2 = PowerTransformer(method="box-cox")
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
            transformer = PowerTransformer(method="yeo-johnson")
            transformer.fit(data)
    else:
        raise ValueError("Invalid trans_type")

    return transformer


def transform(data, transformer, inverse=False):
    """
    Applies or reverses a data transformation.

    Args:
        data (np.ndarray): The input data (as a 1D array).
        transformer (Transformer or list/tuple of Transformers): The transformer(s) to apply.
        inverse (bool, optional): If True, reverses the transformation. Defaults to False.

    Returns:
        np.ndarray: The transformed (or inversely transformed) data.
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
