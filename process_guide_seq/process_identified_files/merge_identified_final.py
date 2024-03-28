import pandas as pd

from os import listdir
from os.path import isfile, join
WINDOW_SIZE = 25


def merge_expriments(expriments_dir_path="NLM1_identified_final", expriments_id=1):
    df_for_merge_lst = []
    expriment_file_names = [f for f in listdir(expriments_dir_path) if isfile(join(expriments_dir_path, f))]

    for expriment_i in range(len(expriment_file_names)):
        # naming
        expriment_i_file_name = expriments_dir_path + "/" + expriment_file_names[expriment_i]
        expriment_i_df = pd.read_csv(expriment_i_file_name)
        # filter rows without alignment
        expriment_i_df = expriment_i_df[~((expriment_i_df["Site_SubstitutionsOnly.Sequence"].isna()) &
                                        (expriment_i_df["Site_GapsAllowed.Sequence"].isna()) &
                                        (expriment_i_df["Align-AlignedText"].isna()))]

        expriment_i_df["run"] = expriments_id
        expriment_i_df["withGaps.#Bulges"] = expriment_i_df["Site_GapsAllowed.Deletions"] + \
            expriment_i_df["Site_GapsAllowed.Insertions"]
        expriment_i_df.loc[~expriment_i_df["Site_SubstitutionsOnly.Sequence"].isna(), "onlySub.#Bulges"] = 0

        # calculate start and end locations to the new alignment
        expriment_i_df["Align.chromStart"] = expriment_i_df["Position"] - WINDOW_SIZE + \
            expriment_i_df["Align-EndPosition"] - \
            expriment_i_df["Align-AlignedText"].str.replace("-", "").str.len()
        expriment_i_df["Align.chromEnd"] = expriment_i_df["Position"] - WINDOW_SIZE + \
            expriment_i_df["Align-EndPosition"]

        expriment_i_df = expriment_i_df[
            ["run", "Cell", "TargetSequence", "bi.sum.mi",
             "WindowChromosome", "WindowSequence",

             "Site_SubstitutionsOnly.Sequence",
             "Site_SubstitutionsOnly.NumSubstitutions", "onlySub.#Bulges",
             "Site_SubstitutionsOnly.Strand", "Site_SubstitutionsOnly.Start", "Site_SubstitutionsOnly.End",

             "Site_GapsAllowed.Sequence", "RealignedTargetSequence",
             "Site_GapsAllowed.Substitutions", "withGaps.#Bulges", "Site_GapsAllowed.Strand",
             "Site_GapsAllowed.Start", "Site_GapsAllowed.End",

             "Align-SiteSeqPlusMaxEditsBefore", "Align-AlignedText", "Align-AlignedTarget",
             "Align-#Mismatches", "Align-#Bulges", "Align-Strand",
             "Align.chromStart", "Align.chromEnd"
             ]]

        expriment_i_df = expriment_i_df.rename(
            {"Cell": "name", "TargetSequence": "sgRNA", "bi.sum.mi": "reads",
             "WindowChromosome": "chrom",

             "Site_SubstitutionsOnly.Sequence": "onlySub.off-target",
             "Site_SubstitutionsOnly.NumSubstitutions": "onlySub.#Mismatches",
             "Site_SubstitutionsOnly.Strand": "onlySub.strand",
             "Site_SubstitutionsOnly.Start": "onlySub.chromStart",
             "Site_SubstitutionsOnly.End": "onlySub.chromEnd",

             "Site_GapsAllowed.Sequence": "withGaps.off-target",
             "RealignedTargetSequence": "withGaps.sgRNA",
             "Site_GapsAllowed.Substitutions": "withGaps.#Mismatches",
             "Site_GapsAllowed.Strand": "withGaps.strand",
             "Site_GapsAllowed.Start": "withGaps.chromStart",
             "Site_GapsAllowed.End": "withGaps.chromEnd",


             "Align-SiteSeqPlusMaxEditsBefore": "Align.SiteSeqPlusMaxEditsBefore",
             "Align-AlignedText": "Align.off-target",
             "Align-AlignedTarget": "Align.sgRNA",
             "Align-#Mismatches": "Align.#Mismatches",
             "Align-#Bulges": "Align.#Bulges",
             "Align-Strand": "Align.strand"}, axis=1)

        df_for_merge_lst.append(expriment_i_df)

    return pd.concat(df_for_merge_lst)


def main():
    for dataset in ["ChangeSeq", "Chen17", "Listgarten", "Tsai_2015"]:
        if dataset == "ChangeSeq":
            df = pd.concat(
                [merge_expriments(
                    expriments_dir_path="output/NLM1_final", expriments_id="NLM1"),  # type: ignore
                 merge_expriments(
                    expriments_dir_path="output/NLM2_final", expriments_id="NLM2"),  # type: ignore
                 merge_expriments(
                    expriments_dir_path="output/CRL_final", expriments_id="CRL"),  # type: ignore
                 merge_expriments(
                    expriments_dir_path="output/ML_final", expriments_id="ML")])  # type: ignore
            df.reset_index(inplace=True, drop=True)
            df.to_csv("output/expriments_final.csv", index=False)
        elif dataset == "Chen17":
            df = merge_expriments(
                expriments_dir_path="output/Chen17_final", expriments_id="Chen17")  # type: ignore
            df.reset_index(inplace=True, drop=True)
            df.to_csv("output/expriments_final_Chen17.csv", index=False)
        elif dataset == "Listgarten":
            df = merge_expriments(
                expriments_dir_path="output/Listgarten_final", expriments_id="Listgarten")  # type: ignore
            df.reset_index(inplace=True, drop=True)
            df.to_csv("output/expriments_final_Listgarten.csv", index=False)
        elif dataset == "Tsai_2015":
            df = merge_expriments(
                expriments_dir_path="output/Tsai_2015_final", expriments_id="Tsai_2014")  # type: ignore
            df.reset_index(inplace=True, drop=True)
            df.to_csv("output/expriments_final_Tsai_2015.csv", index=False)


if __name__ == "__main__":
    main()
