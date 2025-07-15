The code in this directory is used to generate Figures 1a and 1b in the main paper and 2,3,4,5 in Appendix C.

Instructions:
-First, you should run the simulation_experiment.py python code. This does not require any additional files.
The experiment will run 50 times with randomized data each run.

-You must change parameters such as n_classes, n_features, n_informative, and weights_train 
to test all the different conditions stated in the paper.

-With each set of experimental conditions, the code will save a .mat file to your current directory.
You must change the name of the saved .mat file in the python code to reflect the current conditions.
This means change the number of classes, the number of features, and whether the train set is balanced or imbalanced.

-Once you have saved all the necessary .mat files, you should run simulation_experiment_plotting.m to generate plots.
If you did not test the exact conditions stated in the paper you might run into errors in this step due to file names.
Please make sure that the file names in the simulation_experiment_plotting.m as well as the subplot sizes match the 
conditions you tested.

-Additionally, the simulation_experiment_surface_plots_OVA.m and simulation_experiment_surface_plots_MSVM.m files can be used
to generate the surface plots in Appendix C.