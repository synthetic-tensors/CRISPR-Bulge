# CRISPR-Bulge
The CRISPR/Cas9 system is a highly accurate gene-editing technique, but it can also lead to unintended off-target sites (OTS). Consequently, many high-throughput assays have been developed to measure OTS in a genome-wide manner, and their data was used to train machine-learning models to predict OTS. However, these models are inaccurate when considering OTS with bulges due to limited data compared to OTS without bulges. Recently, CHANGE-seq, a new in vitro technique to detect OTS, was used to produce a dataset of unprecedented scale and quality. In addition, the same study produced in cellula GUIDE-seq experiments but none of these experiments included bulges. Here, we generated the most comprehensive GUIDE-seq dataset with bulges, and trained and evaluated state-of-the-art machine-learning models that consider OTS with bulges. We first reprocessed the publicly available experimental raw data of the CHANGE-seq study to generate 20 new GUIDE-seq datasets, and hundreds of OTS with bulges among the original and new GUIDE-seq experiments. We then trained multiple machine-learning models, and demonstrated their state-of-the-art performance both in vitro and in cellula overall and when focusing on OTS with bulges. Last, we visualized the key features learned by our models on OTS with bulges in a unique representation.

The code developed for the preprint manuscript:
[Generating, modeling, and evaluating a large-scale set of CRISPR/Cas9 off-target sites with bulges](https://www.biorxiv.org/content/10.1101/2023.11.01.565099v1)

The code contains multiple options for training models used in the study, including a transfer learning model for CHANGE-seq to GUIDE-seq.


# Usage

## Notes
- You must install the requirements below before running the code.

- Unzip the datasets inside the files folder (See README inside the files folder).

- Models named here and in the manuscript are different:
```
c_1 = MLP-Embd
c_2 = GRU-Embd
c_3 = GRU
```

## Prediction
1. Optimal usage for quick prediction from saved models on a new dataset: To run ensemble prediction using saved models on a saved dataset as a `.csv` file use the `ensemble_prdict` function from `train_and_predict_scripts/utilities.py`. Here is an example that executed when running `python main_predict.py`:
```
ensemble_predict(
        ensemble_components_file_path_and_name_list=[
            "files/bulges/1_folds/5_revision_ensemble_{}_exclude_RHAMPseq_continue_from_change_seq/"
            "read_ts_0/cleavage_models/aligned/FullGUIDEseq/classification/c_2/"
            "ln_x_plus_one_trans/model_fold_0".format(i) for i in range(5)],
        dataset_df="files/datasets/Refined_TrueOT.csv")
```
- `ensemble_components_file_path_and_name_list`(list): A list of file paths and names of the pre-trained ensemble component models.
- `dataset_df` - Either a DataFrame containing the test data or a file path to a CSV file containing the test data.

2. Use `main(version, setting_number, model_types=None)` from `train_and_predict_scripts/predict_config.py` to apply the different prediction scenarios we used in our evaluations. Run `python main_predict.py` for an example.

## Training
1. For training on an entire dataset, use the command-line interface:
```
usage: train_1_fold.py [-h] [-th READ_THRESHOLD] [-sw] [-ens NUM_ENSEMBLES] [-ver MODEL_VERSION] [-tl] [-d_type DATA_TYPE] [-exc_type DATA_TYPES_TO_EXCLUDE] [-c] [-r] [-m_type MODEL_TYPE]

1 fold on dataset or on CHANGE-seq and then continue on GUIDE-seq data.

options:
  -h, --help            show this help message and exit
  -th READ_THRESHOLD, --read_threshold READ_THRESHOLD
                        Reads threshold.
  -sw, --sample_weight  Use Sample weight in training.
  -ens NUM_ENSEMBLES, --num_ensembles NUM_ENSEMBLES
                        Number of ensembles. Default is 1 - no ensembles.
  -ver MODEL_VERSION, --model_version MODEL_VERSION
                        String representing the model version, used to load parameters and save model.
  -tl, --transfer_learning
                        Continue training from CHANGE-seq to GUIDE-seq. XGBoost is not supported.
  -d_type DATA_TYPE, --data_type DATA_TYPE
                        If tl=True, data type to continue on, else, data type to train on. See code for supported options.
  -exc_type DATA_TYPES_TO_EXCLUDE, --data_types_to_exclude DATA_TYPES_TO_EXCLUDE
                        sgRNAs of data type to exclude. See code for supported options.
  -c, --train_classification
                        Train classification task models.
  -r, --train_regression
                        Train regression task models.
  -m_type MODEL_TYPE, --model_type MODEL_TYPE
                        train spesific model type can be either 'c_1', 'c_2', 'c_3', or 'xgboost'.Default is None, and all models types are trained
```
- You can modify the code to support new datasets you are willing to train on.


2. For training on an entire dataset in a cross-validation manner, use the command-line interface:
```
usage: train_folds.py [-h] [-th READ_THRESHOLD] [-sw] [-ens NUM_ENSEMBLES] [-ver MODEL_VERSION] [-d_type DATA_TYPE] [-c] [-r]

10 folds train on CHANGE-seq/GUIDE-seq (CH/GU) or continue training from CHANGE-seq to GUIDE-seq data (Transfer Learning models). Training options are not flexible, see train 1 fold for model felxible train

options:
  -h, --help            show this help message and exit
  -th READ_THRESHOLD, --read_threshold READ_THRESHOLD
                        Reads threshold.
  -sw, --sample_weight  Use Sample weight in training.
  -ens NUM_ENSEMBLES, --num_ensembles NUM_ENSEMBLES
                        Number of ensembles. Default is 1 - no ensembles.
  -ver MODEL_VERSION, --model_version MODEL_VERSION
                        String representing the model version, used to load parameters and save model.
  -d_type DATA_TYPE, --data_type DATA_TYPE
                        Train data type. Can be either CHANGEseq, GUIDEseq, or TL for transfer learning model
  -c, --train_classification
                        Train classification task models.
  -r, --train_regression
                        Train regression task models.
```

## Positional effect and embedding visualizations
The folder `evaluations` contains the notebooks `position_effect.ipynb` and `learned_representation_analysis.ipynb` for generating the interpretability visualizations we used in the manuscript.

## Processing the GUIDE-seq data
The folder `process_guide_seq` contains the DNA barcodes for demultiplexing the raw sequencing data of the GUIDE-seq experiments and some instructions for generating the datasets. Please read the README file in the folder.

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

- If you are willing to use AWS EC2 instances, you might want to work with the AMI `Deep Learning AMI GPU TensorFlow 2.12.0 (Amazon Linux 2) 20230529`.
- Note: Other packages might be needed. Please get in touch with us if you have any problems.