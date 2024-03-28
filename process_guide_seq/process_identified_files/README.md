1. Realign using SWOffinder the OTS obtained from the GUIDE-seq pipeline for each sgRNA (in folder `identified_output_files`) using:
```
python process_with_SWOffinder.py
```

2. Merge the realigned tables of OTS of different sgRNAs into one table using:
```
python merge_identified_final.py
```