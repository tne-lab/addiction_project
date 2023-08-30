### Jon Whear - August 2023 - whear003@umn.edu ###
import numpy as np
from scipy.stats import gamma
import time
import matplotlib.pyplot as plt
from open_ephys.analysis import Session
import gen_sig
​
# Parameters
sample_rate = 30000
coi = 0 # Channel of interest
threshold = 0 # Starting threshold - this should be upated to the mean of the data
target_stim_per_sec = 1
tolerance = 0.1 # Tolerance for difference from target crossings
threshold_step = 0.01
​
# Parameters of the gamma distribution (simulated data)
gen_gamma = False
gamma_shape = 2
gamma_scale = 0.5
num_seconds = 5 # parameters to change for gamma dist
​
# Parameters for power calculation (real data)
lowpass = 120 / (sample_rate/2)
highpass = 160 / (sample_rate/2)
butter_order = 3 # Order of butterworth bandpass filter (using gen_sig.py) - should be 6 according to OEP cpp code (nCoefs*2)
train_min = 5
test_min = 15
​
if gen_gamma: # Generate random samples from the gamma distribution
    num_samples = num_seconds * sample_rate 
    power_data = gamma.rvs(a=gamma_shape, scale=gamma_scale, size=num_samples)
​
else:
    directory = r'Z:\projmon\addiction_synaptic_plasticity\VideoSync Data\SPAF5_Reinstate1_082323\08-23-2023\2023-08-23_13-43-17'
​
    # Create session object
    session = Session(directory)
    print(directory)
    alldata = np.array(session.recordnodes[0].recordings[0].continuous[0].samples)  # datastream object is a list
    
    # Pull out coi
    channel_data = alldata[:, coi]
    cut_time = train_min*sample_rate*60 # when to swap from train to test
    power_data = gen_sig.butter_filt(butter_order, lowpass, highpass, channel_data[:cut_time])
    power_data_test = gen_sig.butter_filt(butter_order, lowpass, highpass, channel_data[cut_time:cut_time+test_min*60*sample_rate])
    num_samples = len(power_data)
    num_samples_test = len(power_data_test)
​
​
# Function to determine the stimulation threshold
def determine_threshold():
    # Calculate statistics of the power data
    power_mean = np.mean(power_data)
    power_std = np.std(power_data)
    
    # Determine the initial power threshold
    threshold = power_mean + power_std
    threshold = 2401.2
    
    # Iterate to find threshold for 1 crossing per second
    tolerance = 0.1  
    while True:
       
        crossings = 0
        for value in power_data:
            if value > threshold:
                crossings += 1
        crossings_per_second = crossings / num_seconds
        
        if abs(crossings_per_second - target_stim_per_sec) < tolerance:
            return threshold
        
        if crossings_per_second > target_stim_per_sec: # I think almost always it will start very low
            if crossings > 10000:
                threshold += threshold_step*10000 # macro tuning
            elif crossings > 5000:
                threshold += threshold_step*5000 # mid tuning
            elif crossings > 1000:
                threshold += threshold_step*1000 # gross tuning
            elif crossings > 500:
                threshold += threshold_step*200 # mid tuning
            elif crossings > 200:
                threshold += threshold_step*200 # mid tuning
            elif crossings > 100:
                threshold += threshold_step*100 # mid tuning
            elif crossings > 10:
                threshold += threshold_step*10 # mid tuning
            elif crossings > 5:
                threshold += threshold_step*5 # mid tuning
            else:
                threshold += threshold_step # fine tuning
        else:
            threshold -= threshold_step # fine tuning
        print(f'Crossings: {crossings}, Threshold: {threshold}')
​
# Function to run stimulation simulation
def run_stim_sim(threshold):
    stimulation_count = 0
    crossing_count = 0
    sample_rate = 30000
    timeout = 0.0 * sample_rate
​
    if gen_gamma:
        num_seconds = 60
        num_samples = num_seconds * sample_rate
        
        # Simulate gamma-distributed power data for a minute
        gamma_shape = 2
        gamma_scale = 0.5
        power_data = gamma.rvs(a=gamma_shape, scale=gamma_scale, size=num_samples)
    else:
        power_data = power_data_test
        num_samples = num_samples_test
        num_seconds = num_samples/30000
    
    for i in range(num_samples):
        if power_data[i] > threshold:
            crossing_count += 1
            stimulation_count += 1
            print(f"Stimulation delivered at time {i/sample_rate:.2f} seconds.")
            # Apply timeout
            for j in range(i, min(i + int(timeout), num_samples)):
                power_data[j] = 0
    
    print(f"Total stimulations delivered: {stimulation_count}")
    print(f"Average stimulations per second: {stimulation_count / num_seconds:.2f}")
​
# Main program
def main():
    threshold = determine_threshold()
    print(f"Determined threshold: {threshold}")
    run_stim_sim(threshold)
​
if __name__ == "__main__":
    main()
