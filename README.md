
## Running Kilosort and post processing the data

The directory kilosort_and_phy_pre_and_post contains scripts to run Kilosort and post process data after manual curation in phy. `denoise_and_run_ks` calls a function to remove noise (`removeLineNoise_SpectrumEstimation.m ` downloaded from [here](https://www.mathworks.com/matlabcentral/fileexchange/54228-remove-line-noise)) and `main_kilosort2_ks` and uses information from the configuration in `Chan14.mat` and `configFile16.m` to run Kilosort. 
 
`recalculate_contamination.m` is a script you can use to force recalculation of the contamination percentage after merging or splitting a cluster (it calls `set_cutoff_ks.m`).  

After running Kilosort and manually curating electrodes in Phy, the next script to run is `phy_post_process.m`.  This  
calls `populate_post_phy_data_structure`, and creates a data structure with the fields expected by further processes.   

## Getting info from the waveforms

The directory waveform_data_processing contains the script `process_wf.m`.  This script generates graphs of clusters' averaged waveforms (and some individual waveforms), calculates FWHM, the firing rate, and divides the neurons into INs and PNs.  Moving all this functionality into Python code is in process, but for the safety paper it was done with this code.

## Single spike data processing

The directory single_spike_data contains the Python code for analyzing the single spike data.  The virtual environment must be activated (navigate to the directory and `source venv/bin/activate` in bash).  Then you can execute the any of graph, spreadsheet, or figure making procedures from `main.py`.  (As a comment in `main` advises, you can see what's available in `procs.py`.  As of this writing, there's no proc to save the first figure yet.). 

## Significance testing

The directory r_scripts contains an R file that performs omnibus significance tests.  Thus far, other tests (like post hocs) have been performed by using the Python code to write and execute R files.

## Other

Utils contains functions that are called by multiple other scripts, and one file that writes out events in an experiment to clarify timings.  Archive contains files that are working their way towards deletion.  Misc contains files that contain some reasoning in progress and will probably eventually be trash.  
