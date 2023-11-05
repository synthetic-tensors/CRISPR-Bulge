# CRISPR-Bulge
The code developed for the preprint manuscript:
[Generating, modeling, and evaluating a large-scale set of CRISPR/Cas9 off-target sites with bulges](https://www.biorxiv.org/content/10.1101/2023.11.01.565099v1)

The code contains multiple options for training models used in the study, including a transfer learning model for CHANGE-seq to GUIDE-seq.

Datasets for training to be added soon (Require to overcome size limit on GitHub).

The CRISPR/Cas9 system is a highly accurate gene-editing technique, but it can also lead to unintended off-target sites (OTS). Consequently, many high-throughput assays have been developed to measure OTS in a genome-wide manner, and their data was used to train machine-learning models to predict OTS. However, these models are inaccurate when considering OTS with bulges due to limited data compared to OTS without bulges. Recently, CHANGE-seq, a new in vitro technique to detect OTS, was used to produce a dataset of unprecedented scale and quality. In addition, the same study produced in cellula GUIDE-seq experiments but none of these experiments included bulges. Here, we generated the most comprehensive GUIDE-seq dataset with bulges, and trained and evaluated state-of-the-art machine-learning models that consider OTS with bulges. We first reprocessed the publicly available experimental raw data of the CHANGE-seq study to generate 20 new GUIDE-seq datasets, and hundreds of OTS with bulges among the original and new GUIDE-seq experiments. We then trained multiple machine-learning models, and demonstrated their state-of-the-art performance both in vitro and in cellula overall and when focusing on OTS with bulges. Last, we visualized the key features learned by our models on OTS with bulges in a unique representation.


# Usage

Note: you must install the requirements below before running the code.

1. Unzip the datasets inside the files folder (See README inside the files folder).

2. To run the main script with an example for training models on the GUIDE-seq data in a 10-fold manner, run with the Python command:
```
python main.py
```

3. To run other example scripts, replace the imported script in `main.py` with the desired script. The available scripts are:
```
At folder train_and_predict_scripts folder:
1. train_10_folds_script_example - the script is currently in the main script. See above.
2. predict_10_folds_script_example - uses the models trained using the train_10_folds_script_example script to predict on the GUIDE-seq dataset.
3. predict_on_TrueOT_script - uses the trained models that were used in comparing CRISPR-Net and CRISPR-IP to predict on the TrueOT dataset. Models naming here and in the manuscript are different (See files folder README).
```
More examples will be provided soon.

# Requirements:
The code was tested with:\
Python interpreter == 3.10.10\
Python packages required for using CRISPR-Bulge (other versions may work as well):\
    numpy==1.23.5\
    pandas==2.0.2\
    scikit-learn==1.2.2\
    scipy==1.10.1\
    tensorflow==2.12.0\
    xgboost==2.0.1\
    catboost==1.2.2

Note: There might be other packages needed. Please contact us in case of any problem.