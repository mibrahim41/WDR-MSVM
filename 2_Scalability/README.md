The code in this directory is used to run Experiment 4 and generate Figures 7a and 7b in Appendix C.

Instructions:
-First, you should run each of the provided Python files (except for rename_files.py) 50 different times. Each file writes a .mat file
to the directory, and generates its own simulation data. Thus, it does not require any addiitonal data files
to run, and it also does not require any parameter changes. Each Python file runs the experiment for one of the
experimental combinations tested Experiment 4.

-Then, you should run the rename_files.py file to rename all the generated .mat file in sequential order.

-Once all the .mat files are generated and renamed, you should run the following 3 .m files:
analyze_data_C.m
analyze_data_P.m
analyze_data_N.m
These files process the generated data, and generate the results to be plotted. Once each of files runs, please save the workspace
as a .mat file named C_data.mat, P_data.mat, and N_data.mat, respectively.

-Finally, run the plot_results.m file to generate the final plots.