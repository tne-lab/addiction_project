# addiction_project_private
Private Repo to share addiction project code for analysis pipeline development

## Scripts
### Analyze_ERP.py
Function: The main analysis script for ERP analysis. (Written by Manny)

Input: OEP recording session. Can also take in simulated data created by gen_sig.py for validation

Output: Averaged event data from an OEP recording in matplotlib plot. Subplots are created for each channel of interest.

### analyze_event_power.py 
Function: Takes parameters given by user to analyze frequency-specific Power around "events" in Open Ephys data


### power_analysis_tools.py
NOTE: Formerly "amplitude_longitudinal_analysis.py"

Input: 1) OEP Recording with 6 Phase Calculators tuned to Gamma Band "Amplitude" (Online Power). 2) gen_sig.py to generate simulated signal 3) LFP data to calculate Offline Power

Output: Heatmap showing amplitude² (Power), averaging every second (30k samples). Power spectral analysis. Some other tools included.

### gen_sig.py
Input: Parameters.

Output: Mimics OEP data and can be plugged into Manny's code for validation, among other things.

### oep_driver.py
Function: Drives OEP recordings of set length and records to specified location. Used in this context to compare online and offline PC Magnitude calculations.

### power_histogram.py
Input: OEP data

Output: Histogram of Power Data

### static_thresh_counter.py
Function: Runs through OEP data, iteratively decides on a threshold, and then tests that on a longer stretch of data. To start this was 5 minutes to set threshold, 15 minutes to test threshold.

### thresh_stepper.py
Function: Tests either gamma distribution or power distribution to step threshold up or down to reach an average of 1 stimulation per second.
