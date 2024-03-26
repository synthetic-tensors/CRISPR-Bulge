"""
This module generates synthetic off-target sequences with varying numbers of mismatches and bulges
for a given sgRNA. It utilizes alignment algorithm to confirm the alignment.
The generated off-targets are intended for use in downstream analysis, such as embedding creation.
"""

import itertools
import random
import pandas as pd
from tqdm import tqdm

from Bio import Align

from OT_deep_score_src.dataset_utilities import load_order_sg_rnas
from OT_deep_score_src.general_utilities import DATASETS_PATH, Data_type

DISABLE_TQDM = True


def extract_alignment_pattern(alignment):
    """
    Extracts the mismatch and gap pattern from a Biopython alignment object.

    Args:
        alignment (Bio.Align.PairwiseAlignment): A pairwise alignment object.

    Returns:
        str: A string representing the alignment pattern using ".", "-", and "|"
             to denote mismatches, gaps, and matches respectively.
    """
    alignment = alignment.format().split("\n")[1]
    alignment_pattern = ""
    for c in alignment:
        alignment_pattern += c if c in [".", "-", "|"] else ""
    return alignment_pattern


def find_alignments(sg_rna, off_target, k_mis, bulge_positions, dna_bulge=True):
    """
    Finds alignments between an sgRNA and an off-target sequence with specific mismatch and bulge constraints.

    Args:
        sg_rna (str): The sgRNA sequence.
        off_target (str): The off-target sequence.
        k_mis (int): The maximum number of allowed mismatches.
        bulge_positions (list): A list of positions where bulges are required in the alignment.
        dna_bulge (bool, optional): If True, searches for alignments with DNA bulges.
                                    Defaults to True.

    Returns:
        list: A list of Biopython alignment objects that meet the specified criteria.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 0
    aligner.mismatch_score = -1
    aligner.open_gap_score = -(k_mis + 0.5)
    aligner.extend_gap_score = -(k_mis + 0.5)
    alignments = aligner.align(sg_rna, off_target)
    # alignments = pairwise2.align.globalms(sg_rna, off_target, 0, -1, -k_mis, -k_mis)
    req_alignments = []
    for alignment in alignments:
        alignment_pattern = extract_alignment_pattern(alignment)
        ali_bulge_size = alignment_pattern.count("-")
        ali_mm = alignment_pattern.count(".")
        if k_mis == ali_mm and ali_bulge_size == len(bulge_positions) and all(
                [alignment[0 if dna_bulge else 1][pos] == "-" for pos in bulge_positions]):
            req_alignments.append(alignment)

    return req_alignments


def generate_off_target_k_mismaches(sg_rna, k):
    """
    Generates off-target sequences with a specified number of mismatches compared to the sgRNA.

    Args:
        sg_rna (str): The sgRNA sequence.
        k (int): The desired number of mismatches.

    Returns:
        list: A list of off-target sequences, each containing 'k' mismatches.
    """
    bases = ["A", "C", "G", "T"]
    positions = [i for i in range(0, 20)] + [21, 22]

    off_targets = []
    for mis_positions in itertools.combinations((positions), k):
        off_target = sg_rna
        for i in mis_positions:
            new_nec = random.choice([nec for nec in bases if nec != sg_rna[i]])
            off_target = off_target[:i] + new_nec + off_target[i+1:]
        off_target = off_target.replace("N", random.choice(bases))
        off_targets.append(off_target)

    return off_targets


def generate_off_target_k_mismaches_dna_bulge(sg_rna, k_mis):
    """
    Generates off-target sequences with a specified number of mismatches and a single DNA bulge.

    Args:
        sg_rna (str): The sgRNA sequence.
        k_mis (int): The desired number of mismatches.

    Returns:
        tuple:
            * sg_rnas (list): A list of aligned sgRNA sequences,
                accommodating the DNA bulge.
            * off_targets (list): A list of corresponding aligned off-target sequences.
    """
    bases = ["A", "C", "G", "T"]
    positions = [i for i in range(0, 20)] + [21, 22]
    off_targets = []
    sg_rnas = []
    for mis_positions in tqdm(list(itertools.combinations((positions), k_mis)), disable=DISABLE_TQDM):
        for dna_bulge_pos in range(1, 23):
            succeed = False
            while not succeed:
                # create mismatches
                off_target = sg_rna
                for i in mis_positions:
                    new_nec = random.choice([nec for nec in bases if nec != sg_rna[i]])
                    off_target = off_target[:i] + new_nec + off_target[i+1:]
                # add DNA bulge
                for new_nec in bases:
                    off_target_temp = off_target[:dna_bulge_pos] + new_nec + off_target[dna_bulge_pos:]
                    alignments = find_alignments(sg_rna, off_target_temp, k_mis, [dna_bulge_pos])
                    if alignments:
                        succeed = True
                        off_target_temp = off_target_temp.replace("N", random.choice(bases))
                        sg_rnas.append(sg_rna[:dna_bulge_pos] + "-" + sg_rna[dna_bulge_pos:])
                        off_targets.append(off_target_temp)
                        break
    return sg_rnas, off_targets


def generate_off_target_k_mismaches_rna_bulge(sg_rna, k_mis):
    """
    Generates off-target sequences with a specified number of mismatches and a single RNA bulge.

    Args:
        sg_rna (str): The sgRNA sequence.
        k_mis (int): The desired number of mismatches.

    Returns:
        list: A list of off-target sequences containing the RNA bulge and mismatches.
    """
    bases = ["A", "C", "G", "T"]
    positions = [i for i in range(0, 20)] + [21, 22]
    off_targets = []
    for mis_positions in tqdm(list(itertools.combinations((positions), k_mis)), disable=DISABLE_TQDM):
        for rna_bulge_pos in [pos for pos in positions if pos not in mis_positions]:
            succeed = False
            while not succeed:
                # create mismatches
                off_target = sg_rna
                for i in mis_positions:
                    new_nec = random.choice([nec for nec in bases if nec != sg_rna[i]])
                    off_target = off_target[:i] + new_nec + off_target[i+1:]
                # add RNA bulge
                for new_nec in bases:
                    off_target_temp = off_target[:rna_bulge_pos] + off_target[rna_bulge_pos + 1:]
                    alignments = find_alignments(sg_rna, off_target_temp, k_mis, [rna_bulge_pos], dna_bulge=False)
                    if alignments:
                        succeed = True
                        off_target_temp = off_target[:rna_bulge_pos] + "-" + off_target[rna_bulge_pos + 1:]
                        off_target_temp = off_target_temp.replace("N", random.choice(bases))
                        off_targets.append(off_target_temp)
                        break
    return off_targets


def main():
    """
    Coordinates the generation of synthetic off-target sites and saves them to CSV files.

    1. Loads a set of sgRNAs.
    2. Iterates over sgRNAs:
       * Generates off-targets with varying mismatches, DNA bulges, and RNA bulges.
       * Constructs a DataFrame for each sgRNA's off-target set.
       * Saves the DataFrame as a CSV file.
    """
    sg_rnas = load_order_sg_rnas(Data_type.CHANGE_SEQ)
    for sg_rna in sg_rnas:
        df = pd.DataFrame()
        for k in range(0, 7):
            off_targets = generate_off_target_k_mismaches(sg_rna, k)
            mis_k_df = pd.DataFrame(
                data={"off-target": off_targets, "sgRNA": sg_rna,
                      "Alignment Mismatches": k, "Alignment Bulge Size": 0,
                      "Alignment distance": k})
            df = pd.concat((df, mis_k_df))

        for k in range(0, 5):
            alinged_sg_rnas, off_targets = generate_off_target_k_mismaches_dna_bulge(sg_rna, k_mis=k)
            mis_k_df = pd.DataFrame(
                data={"off-target": off_targets, "sgRNA": alinged_sg_rnas,
                      "Alignment Mismatches": k, "Alignment Bulge Size": 1,
                      "Alignment distance": k + 1})
            df = pd.concat((df, mis_k_df))

        for k in range(0, 5):
            off_targets = generate_off_target_k_mismaches_rna_bulge(sg_rna, k_mis=k)
            mis_k_df = pd.DataFrame(
                data={"off-target": off_targets, "sgRNA": sg_rna,
                      "Alignment Mismatches": k, "Alignment Bulge Size": 1,
                      "Alignment distance": k + 1})
            df = pd.concat((df, mis_k_df))

        df.to_csv(DATASETS_PATH + "generated_off_targets_for_embedding/{}.csv".format(sg_rna), index=False)


if __name__ == "__main__":
    main()
