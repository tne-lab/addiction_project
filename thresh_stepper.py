### Jon Whear - August 2023 - whear003@umn.edu ###

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from open_ephys.analysis import Session
import gen_sig
​
# Parameters of the gamma distribution
shape_parameter = 2.0  # Shape parameter (k)
scale_parameter = 1.0  # Scale parameter (theta)
gen_gamma = False
threshold = 0 # Starting threshold
sample_rate = 30000
coi = 0 # Channel of interest
thresh_step = 1
thresh_decay = 1/30000
​
# Pwr constants
lowpass = 120 / (sample_rate/2)
highpass = 160 / (sample_rate/2)
butter_order = 3 # Order of butterworth bandpass filter (using gen_sig.py) - should be 6 according to OEP cpp code (nCoefs*2)
​
​
if gen_gamma: # Generate random samples from the gamma distribution
    num_samples = 30000 * 60
    samples = gamma.rvs(a=shape_parameter, scale=scale_parameter, size=num_samples)
​
else:
    directory = r'Z:\projmon\addiction_synaptic_plasticity\VideoSync Data\SPAM4_PrefTest_080223'
​
    # Create session object
    session = Session(directory)
    alldata = np.array(session.recordnodes[0].recordings[0].continuous[0].samples)  # datastream object is a list
    
    # Pull out coi
    channel_data = alldata[:, coi]
    samples = gen_sig.butter_filt(butter_order, lowpass, highpass, channel_data)
    num_samples = len(samples)
    print(samples)
    threshold = np.average(samples) # Set value to average of data?
​
thresh_list = []
stim_times = []
timeout = 0
​
def threshold_stimmer(samples): # Decision making for trigger
    global threshold, timeout
    for i in range(len(samples)):
        timeout += 1 # Allow timeout to continue 
        if samples[i] > threshold and timeout > 30000:
            threshold += thresh_step
            if timeout > 30000:
                stim_times.append(i)
                print(f'stim! at {i}')
                print(f'adding 1 second timout')
                timeout = 0 # Reset timeout value`
        else:
            threshold -= thresh_decay # decay
        thresh_list.append(threshold)
    return stim_times, thresh_list
​
​
def plot_threshold_stims(stim_times, thresh_list): # Create a histogram of the samples
    plt.plot(thresh_list)
    for i in range(len(stim_times)):
        plt.axvline(x=stim_times[i], color='black', ls='-', lw=0.5) # Assumes before = after!
    plt.show()
​
def print_stats(stim_times, thresh_list):
    print(f'Recording Seconds: {(len(samples)/sample_rate)}')
    print(f'Threshold Step: {thresh_step}   Threshold Decay: {thresh_decay}')
    print(f'Number of Stims {len(stim_times)}')
    print(f'Stimulations per Second: {len(stim_times)/(len(samples)/sample_rate)}')
    print(f'Averge Stim Threshold: {np.average(thresh_list)}')
​
stim_times, thresh_list = threshold_stimmer(samples)
plot_threshold_stims(stim_times, thresh_list)
print_stats(stim_times, thresh_list)
