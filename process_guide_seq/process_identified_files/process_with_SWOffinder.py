import pandas as pd

import os
import shutil
from pathlib import Path
from os import listdir
from os.path import isfile, join

import subprocess


def process_df(expriment_df: pd.DataFrame, expriment_file_dist: str):
    """
    Converts the Window sequence into fasta file for the SWOffinder to work on the window sequence
    """
    expriment_df.dropna(subset=["WindowSequence"], inplace=True)
    expriment_df.reset_index(inplace=True, drop=True)
    expriment_df["WindowSequence"] = expriment_df["WindowSequence"].str.upper()
    expriment_sg_rna = expriment_df["TargetSequence"].iloc[0]
    with open(expriment_file_dist, "w") as f:
        window_seqs = expriment_df["WindowSequence"].values
        size = len(window_seqs)
        for i, window_seq in enumerate(window_seqs):
            f.write(">{}\n".format(i))
            f.write(window_seq)
            if i != size - 1:
                f.write("\n")
    return expriment_df, expriment_sg_rna


def process_expriments(expriments_name="NLM1"):
    expriments_dir_path = "identified_output_files/{}_identified".format(expriments_name)
    expriment_file_names = [f for f in listdir(expriments_dir_path) if isfile(join(expriments_dir_path, f))]

    for expriment_i in range(len(expriment_file_names)):
        # naming
        expriment_i_file_name = expriments_dir_path + "/" + expriment_file_names[expriment_i]
        expriment_i_fasta_output_dist_dir = "{}/{}_window_fasta/".format("output", expriments_name)
        expriment_i_fasta_output_dist = "{}{}.fa".format(
            expriment_i_fasta_output_dist_dir, expriment_file_names[expriment_i][:-4])
        expriment_i_filtered_output_dist = "{}/{}_distance_filtered/{}.csv".format(
            "output", expriments_name, expriment_file_names[expriment_i][:-4])
        expriment_i_identified_final_output_dist = "{}/{}_final/{}.csv".format(
            "output", expriments_name, expriment_file_names[expriment_i][:-4])
        Path(os.path.dirname(expriment_i_fasta_output_dist)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(expriment_i_filtered_output_dist)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(expriment_i_identified_final_output_dist)).mkdir(parents=True, exist_ok=True)

        expriment_i_df = pd.read_csv(expriment_i_file_name, sep="\t")
        expriment_i_df, expriment_sg_rna = process_df(expriment_i_df, expriment_i_fasta_output_dist)
        # run Smith-Waterman Off Target for confirm alignment
        java_file = "../SWOffinder/SmithWatermanOffTarget/*.java"
        # Compile Java program
        subprocess.run("javac -d bin {}".format(java_file), shell=True)
        # Run Java program
        run_cmd = "java -cp bin SmithWatermanOffTarget.SmithWatermanOffTargetSearchAlign {} {} {} " \
                  "6 6 4 1 1 false 50 NGG true".format(
                      expriment_i_fasta_output_dist, expriment_sg_rna, expriment_i_filtered_output_dist[:-4])
        subprocess.run(run_cmd, shell=True)
        # remove temp files
        shutil.rmtree(expriment_i_fasta_output_dist_dir)

        # remove more than one alignment in window
        expriment_filtered_df = pd.read_csv(expriment_i_filtered_output_dist)
        expriment_filtered_df["allowed_align_edit"] = expriment_filtered_df["#Bulges"] + \
            expriment_filtered_df["#Mismatches"]
        expriment_filtered_df = expriment_filtered_df.sort_values(["allowed_align_edit", "#Bulges"])
        expriment_filtered_df = expriment_filtered_df.drop_duplicates(subset="Chromosome", keep="first")

        # add the alignments found to the identified table
        expriment_i_df.loc[
            expriment_filtered_df["Chromosome"],
            ["Align-Strand", "Align-EndPosition", "Align-SiteSeqPlusMaxEditsBefore",
             "Align-#Edit", "Align-AlignedTarget", "Align-AlignedText",
             "Align-#Mismatches", "Align-#Bulges"]] = \
            expriment_filtered_df[["Strand", "EndPosition", "SiteSeqPlusMaxEditsBefore",
                                   "#Edit", "AlignedTarget", "AlignedText", "#Mismatches", "#Bulges"]].values

        expriment_i_df.to_csv(expriment_i_identified_final_output_dist, index=False)
        # remove temp files
        shutil.rmtree(os.path.dirname(expriment_i_filtered_output_dist))


def main():
    process_expriments(expriments_name="NLM1")
    process_expriments(expriments_name="NLM2")
    process_expriments(expriments_name="CRL")
    process_expriments(expriments_name="ML")
    process_expriments(expriments_name="Chen17")
    process_expriments(expriments_name="Listgarten")
    process_expriments(expriments_name="Tsai_2015")


if __name__ == "__main__":
    main()
