This file contains instructions to replicate the results of this experiment. Please follow the instructions below.

1) Navigate to the Data_Generation subdirectory.
2) Run the WDR_MSVM_script.m file to generate the data.
3) Run the Prepare_Data_Script.m file, and save the workspace with the name raw_data_1.mat in the same directory.
4) Run the extract_mats.m file, and save the resulting workspace in the directories named 4_Classes and 7_Classes with the name 'multi_pump_data.mat'
5) Navigate to the directory associated with the experiment of interest (either 4_Classes or 7_Classes).
6) Run the WDR_MSVM_pump_{number}.py file the desired number of repititions.
7) Run the Rename_Files.py file in order to rename the files in sequential order.
8) Run the analyze_data.mat file to obtain the final results.