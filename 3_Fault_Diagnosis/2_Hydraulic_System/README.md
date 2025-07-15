This file contains the instructions to run the fault diagnosis experiment for the hydraulic system.

This directory contains multiple subdirectories, one per classification task. However, before navigating to the directories and running the code please complete the following steps first.

1) Obtain the .zip file of the data form the source referenced in the paper.
2) Extract the files into this directory
3) Run the Preprocess_Data.py file
4) Copy and past the generated .mat file into each of the subdirectories of this directory.

After completing the previous steps, please navigate to each of the subdirectories of this directory and perform the following steps.

1) Run the WDR_MSVM_{name}.py file as many repetitions as desired.
2) Run the Rename_Files.py file to rename the generated .mat files in sequential order.
3) Run the analyze_data.m file to generate the final results.