# CRISPR-Bulge
The code developed for the preprint manuscript:
[Generating, modeling, and evaluating a large-scale set of CRISPR/Cas9 off-target sites with bulges](URL_TBD)

The code contains multiple options for training models used in the study, including a transfer learning model for CHANGE-seq to GUIDE-seq.

Datasets for training to be added soon (Require to overcome size limit on GitHub).

The CRISPR/Cas9 system is a highly accurate gene-editing technique, but it can also lead to unintended off-target sites (OTS). Consequently, many high-throughput assays have been developed to measure OTS in a genome-wide manner, and their data was used to train machine-learning models to predict OTS. However, these models are inaccurate when considering OTS with bulges due to limited data compared to OTS without bulges. Recently, CHANGE-seq, a new in vitro technique to detect OTS, was used to produce a dataset of unprecedented scale and quality. In addition, the same study produced in cellula GUIDE-seq experiments but none of these experiments included bulges. Here, we generated the most comprehensive GUIDE-seq dataset with bulges, and trained and evaluated state-of-the-art machine-learning models that consider OTS with bulges. We first reprocessed the publicly available experimental raw data of the CHANGE-seq study to generate 20 new GUIDE-seq datasets, and hundreds of OTS with bulges among the original and new GUIDE-seq experiments. We then trained multiple machine-learning models, and demonstrated their state-of-the-art performance both in vitro and in cellula overall and when focusing on OTS with bulges. Last, we visualized the key features learned by our models on OTS with bulges in a unique representation.


# Usage
```
The main.py script calls for an example script for training models for GUIDE-seq data.

Datasets for training to be added soon (Require to overcome size limit on GitHub).

More examples are to be added soon.
```

# Requirements:
The code was tested with:\
Python interpreter == 3.10.10\
Python packages required for using CRISPR-Bulge (others versions may work as well):\
    numpy==1.23.5\
    pandas==2.0.2\
    scikit-learn==1.2.2\
    scipy==1.10.1\
    tensorflow==2.12.0\
    xgboost==2.0.1\
    catboost==1.2.2

Note: There might be other packages needed. Please contact us in case of any problem.
