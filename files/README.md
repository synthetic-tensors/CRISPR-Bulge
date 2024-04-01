# Datasets
You must unzip the `datasets` folder from `datasets.zip`.

Since the zipped file was uploaded using Git LFS, you need to clone the repository using Git LFS. Otherwise, you might download a corrupted file.

Notes:
- The FullGUIDEseq dataset combines the GUIDEseq (from the CHANGE-seq study) and NewGUIDEseq datasets.
- The folder also contains the dataset partition code for GUIDE-seq and CHANGE-seq.

# Trained models
The folder `bulges` contains the trained models that were used to compare CRISPR-Net, CRISPR-IP, etc., on the independent datasets.

Models named here and in the manuscript are different:
```
c_1 = MLP-Embd
c_2 = GRU-Embd
c_3 = GRU
```