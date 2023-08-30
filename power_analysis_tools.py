### Jon Whear - Summer 2023 - whear003@umn.edu ###

### Dependencies ###
import os
import sys
sys.path.append(r'C:\Users\Windows\anaconda3\envs\open-ephys-python-tools\Lib\site-packages')
import math
from open_ephys.analysis import Session
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pprint import pprint
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import gen_sig
import json

'''
This script is designed to pull out Amplitude (Power Correlate) data from OEP recordings over time.

Reasoning: We have calculated amplitude for entire recordings, but looking for variability over time will allow us to determine if these values change between recording/day/rat

Setup:
    - 6 Phase Calculators, each tuned to a specific 20Hz range of High Gamma
        - 80-100Hz (Channel 8)
        - 100-120Hz (Channel 9)
        - 120-140Hz (Channel 10)
        - 140-160Hz (Channel 11)
        - 160-180Hz (Channel 12)
        - 180-200Hz (Channel 13)

        To keep recordings lightweight, convention I have set for record node is set to record only channels 1-16. More/less should work, however, assuming you adjust settings for gamma band amplitude data

FILE LOCATIONS - WILL BE DIFFERENT FOR WINDOWS V. UNIX(Linux & MacOS) FILE ARCH.
## Windows 10 Lab Computer (TNEL) - Virginia's Bay ##
file_loc = r'C:\\Users\\Jon\\Desktop'

directory = '/Users/Jon/Desktop/OEP_Data/RAW_OEP_Sine_Wave_30s_2023-07-14_12-51-29'

## MacOS Monterey (Jon W's Laptop) Computer ##
file_loc = r'/Users/Jon/Desktop/OEP_Data' 

## Windows 10 Jon Computer (Jon W's) ##
file_loc = r'D:\\EPHYSDATA\\leah\MAG_TESTING\\SINE'
'''

### Constants ###
n_bin = 6 # Number of gamma band splits
start_channel = 5 # Channel to start on for amplitude data - Be careful! This will not always be the same
sample_rate = 30000
amp_avg_length = 1 # Length of time, in seconds, used to calculate average power for given range. Must be an integer.
power_min = 0 # Min Amp^2 (power) value
power_max = 8000 # Max value for plotting power in heatmap
lowpass = 120 / (sample_rate/2)
highpass = 160 / (sample_rate/2)
butter_order = 3 # Order of butterworth bandpass filter (using gen_sig.py) - should be 6 according to OEP cpp code (nCoefs*2)

file_loc = r'D:\EPHYSDATA\leah'

online_rec_name = 'SPAM3_2023-online'   # Data which has been run through OEP online Phase Calculator (PC) analysis
offline_rec_name = 'SPAM3_2023-offline' # Same recording, but hasn't been run through PC yet.
online_directory = os.path.join(file_loc, online_rec_name)
#online_directory = r'D:\EPHYSDATA\leah\SPAM3_Habituation3_071223\07-12-2023\2023-07-12_15-29-18'
online_directory = r'Z:\projmon\addiction_synaptic_plasticity\ERPs\SPAF5_Conditioning1_preERP_072923'
offline_directory = os.path.join(file_loc, offline_rec_name) #'/Users/Jon/Desktop/OEP_Data/RAW_OEP_Sine_Wave_30s_2023-07-14_12-51-29'

def open_session(n_bin, start_channel, sample_rate, directory):
    session = Session(directory)
    recording_cont = session.recordnodes[0].recordings[0].continuous[0]
    alldata = recording_cont.samples  # A list of lists
    alldata2 = np.array(alldata) # Convert to np array
    rec_sec = math.floor(len(alldata2)/sample_rate) # How many seconds is the recording?
    alldata3 = alldata2[:rec_sec*sample_rate, (start_channel-1):(start_channel+n_bin-1)] # 0 indexed, reduce to needed channels only and cut length to multiple of avg_sample_length integer
    return alldata3, rec_sec

def comp_analysis(magnitude, sample_rate, rec_sec):
    comp_channel = 0
    avg_data = pd.DataFrame(columns = ['freq','time', 'amp'])

    for i in range(rec_sec):
        inst_data = []
        amp_chan = magnitude[i*sample_rate:(i+1)*sample_rate] # only 1d data assuming only one coi
        amp_avg = np.average(amp_chan)
        inst_data.append(0) # What frequency
        inst_data.append(i+1) # What time, account for 0 index
        inst_data.append(amp_avg**2) # What amplitude TODO - I am calculating amp^2 of average, not of raw values. Not sure if this is ok
        avg_data = pd.concat([pd.DataFrame([inst_data], columns = avg_data.columns), avg_data], ignore_index=True) # pd.concat should be fastest? not 100% sure
   
    avg_data = avg_data.fillna(0) # It seems to take a second for values to settle / calc to begin
    return avg_data

def long_analysis(n_bin, sample_rate, rec_sec, alldata3):
    avg_data = pd.DataFrame(columns = ['freq','time', 'amp'])

    for i in range(rec_sec):
        for j in range(n_bin):
            inst_data = []
            amp_chan = alldata3[i*sample_rate:(i+1)*sample_rate, j] # The channel of interest and the associated 30k samples in that second.
            amp_avg = np.average(amp_chan)
            inst_data.append(j) # What frequency
            inst_data.append(i+1) # What time, account for 0 index
            inst_data.append(amp_avg**2) # *0.0000037682911550208444) # What amplitude TODO - I am calculating amp^2 of average, not of raw values. Not sure if this is ok
            avg_data = pd.concat([pd.DataFrame([inst_data], columns = avg_data.columns), avg_data], ignore_index=True) # pd.concat should be fastest? not 100% sure

    avg_data = avg_data.fillna(0) # It seems to take a second for values to settle / calc to begin
    return avg_data

def long_analysis_raw(alldata3):
    mag_data = alldata3[30000:32000]
    pwr_data = (mag_data**2)
    print(pwr_data[:100])
    return pwr_data

### Plotting ###
def plot_mag(avg_data):
    ax = avg_data.pivot("freq", "time", "amp") # Convert to wide-format data for sns heatmap - eg: X.pivot("y_axis", "x_axis", "ampliude")
    plot = sns.heatmap(ax, vmin = power_min, vmax = power_max) # plot heatmap
    y_labels = ['80-100Hz', '100-120Hz', '120-140Hz', '140-160Hz','160-180Hz','1800-200Hz']
    plot.set_yticklabels(y_labels, fontsize=7)
    plot.set(title="Amplitude² for: {} ({}s)".format(online_rec_name, rec_sec), xlabel = "Time (s)", ylabel= "Avg. Amp²")
    plt.savefig(online_directory + "{}_power_heatmap.png".format(online_rec_name))
    plt.show()
    print('Plotting done. Figure saved.')

def comp_mag(online_data, offline_data, comp_channel):
    online_data['freq'].replace(comp_channel, 'online', inplace = True)
    offline_data['freq'].replace(comp_channel, 'offline', inplace = True)
    avg_data = pd.concat([online_data, offline_data], ignore_index=True)
    ax = avg_data.pivot("freq", "time", "amp") # Convert to wide-format data for sns heatmap - eg: X.pivot("y_axis", "x_axis", "ampliude")
    plot = sns.heatmap(ax, vmin = power_min, vmax = power_max) # plot heatmap
    y_labels = ['Offline Power', 'Online Power'] # I think this used to be flipped?
    plot.set_yticklabels(y_labels, fontsize=7)
    plot.set(title="Amplitude² for: {} ({}s)".format(offline_rec_name, rec_sec), xlabel = "Time (s)", ylabel= "Avg. Amp²")
    plt.show()

def run_analysis_with_parameters():
    alldata3, rec_sec = open_session(n_bin, start_channel, sample_rate, amp_avg_length, online_directory)

    comp_channel = 0
    online_data = long_analysis(n_bin, sample_rate, amp_avg_length, rec_sec, alldata3)# [comp_channel]
    online_data = online_data.drop(online_data[(online_data.freq != comp_channel)].index)

    
    for lowpass in range(60, 210, 10):
        for highpass in range(lowpass + 140, 210, 10):
            # Perform analysis with the current lowpass and highpass values
            offline_data = comp_analysis(gen_sig.calc_mag(offline_directory, comp_channel, lowpass / (sample_rate/2), highpass / (sample_rate/2), butter_order), sample_rate, amp_avg_length, rec_sec)

            # Plot and savsece the figure
            plt.plot(online_data['time'], online_data['amp'])
            plt.plot(offline_data['time'], offline_data['amp'])
            plt.title(f'Lowpass: {lowpass} Hz, Highpass: {highpass} Hz')
            plt.xlabel('Time (s)')
            plt.ylabel('Average Amp²')
            image_name = f"plot_low_{lowpass}_high_{highpass}.png"
            image_path = os.path.join(file_loc, image_name)
            plt.savefig(image_path)
            plt.close()  # Close the current plot for the next iteration


def heatmap_loop(alldata3, seconds, directory):
    sample_rate = 30000
    length = sample_rate * seconds
    order = 2

    # SNS
    heat_min = 0
    heat_max = 800

    # For Bandpass Filter #
    start_freq = 30
    end_freq = 200
    step = 1
    n_bin = int((end_freq-start_freq)/step)

    if (end_freq-start_freq) % step != 0:
        raise Exception('frequency range must be divisible by step size to create integer for bin number. For Example (200-0)/10 = 20 bins')
    
    pass_list = list(range(start_freq, end_freq + 1, step))
    random_data = np.zeros((n_bin, length))
    ### GENERATE OEP DATA ###
    for i in range(n_bin):
        lowpass = pass_list[i] / (sample_rate/2)
        highpass = (pass_list[i]+step) / (sample_rate/2)
        gen_signal = alldata3[:, 0]
        # param = [1, seconds, sample_rate, f1, m1, f2, m2, spike_width, ma, spike_interval, delay] # This is kinda stupid but it works for now
        line = gen_sig.butter_filt(order, lowpass, highpass, gen_signal)
        random_data[i] = line
        print(f'loop {i} is COMPLETE!!!')

    avg_data = pd.DataFrame(columns = ['freq','time', 'amp'])

    for i in range(seconds):
        for j in range(n_bin):
            inst_data = []
            amp_chan = random_data[j, i*sample_rate:(i+1)*sample_rate] # The channel of interest and the associated 30k samples in that second.
            amp_avg = np.average(amp_chan)
            inst_data.append(j*step+start_freq) # What frequency
            inst_data.append(i+1) # What time, account for 0 index
            inst_data.append(amp_avg**2)
            avg_data = pd.concat([pd.DataFrame([inst_data], columns = avg_data.columns), avg_data], ignore_index=True) # pd.concat should be fastest? not 100% sure

    avg_data = avg_data.fillna(0) # It seems to take a second for values to settle / calc to begin
    ax = avg_data.pivot("freq", "time", "amp")

    ax.to_csv(os.path.join(directory, f'{online_rec_name}_startfreq({start_freq})_endfreq({end_freq}_step({step}).csv'))

    # Create a heatmap using seaborn
    plot = sns.heatmap(ax, cmap='viridis', vmin=heat_min, vmax=heat_max)
    plot.set(title="Amplitude² for: {} ({}s)".format(online_rec_name, seconds), xlabel="Time (ms)", ylabel="Frequency")
    plot_save_loc = os.path.join(directory, f'{online_rec_name}_startfreq({start_freq})_endfreq({end_freq}_step({step})_heatmax({heat_max})).png')
    plt.savefig(plot_save_loc)
    # plt.show()
    plt.close()

### Main ###
if __name__ == "__main__":
    alldata3, rec_sec = open_session(n_bin, start_channel, sample_rate, online_directory) # online Analysis
    heatmap_loop(alldata3)
    seconds = math.floor(len(alldata3)/sample_rate)
    x = input('Type 0 for single recording analysis. Type 1 for online vs. offline analysis: ')

    # Define desired analysis
    if x == '0':
        avg_data = long_analysis(n_bin, sample_rate, amp_avg_length, rec_sec, alldata3)
        plot_mag(avg_data)

    elif x == '1':
        comp_channel = 0
        online_data = long_analysis(n_bin, sample_rate, rec_sec, alldata3)# [comp_channel]
        online_data = online_data.drop(online_data[(online_data.freq != comp_channel)].index)

        offline_data = comp_analysis(gen_sig.calc_mag(offline_directory, comp_channel, lowpass, highpass, butter_order), sample_rate, rec_sec)
        print(f"online min {online_data['amp'].min()}")
        print(f"offline min {offline_data['amp'].min()}")
        plt.plot(online_data['time'], online_data['amp'])
        plt.plot(offline_data['time'], offline_data['amp'])
        plt.show()
        plt.close()

        comp_mag(online_data, offline_data, comp_channel)

    elif x == '2': # For seeing the raw data w/o any calculations
        # params
        coi = 0
        start_samp = 0
        stop_samp = 120000

        online_raw = alldata3[:, coi] # alldata is opened above

        # alldata4, rec_sec = open_session(n_bin, start_channel, sample_rate, offline_directory) # offline data loading
        # offline_raw = alldata4[:, coi]

        plt.plot(online_raw[start_samp:])
        # plt.plot(offline_raw[start_samp:stop_samp])
        plt.show()
        plt.close()

    elif x == '3':
        # Set the size of the array
        # n_bin = 10
        seconds = 5
        data = []
        sample_rate = 30000
        length = sample_rate * seconds
        rand_min = 1
        rand_max = 10
        order = 2

        # Signal Gen #
        f1 = 60
        m1 = 60
        f2 = 150
        m2 = 40
        spike_width = 0.1
        ma = 100
        spike_interval = 1
        delay = 10
        channels = 40

        # SNS
        heat_min = 0
        heat_max = 1

        # For Bandpass Filter #
        start_freq = 10
        end_freq = 200
        step = 10
        n_bin = int((end_freq-start_freq)/step)

        if (end_freq-start_freq) % step != 0:
            raise Exception('frequency range must be divisible by step size to create integer for bin number. For Example (200-0)/10 = 20 bins')
        
        pass_list = list(range(start_freq, end_freq + 1, step))
        order = 2 # It's 6th order in the cpp code for this freq range
        random_data = np.zeros((n_bin, length))
        ### GENERATE OEP DATA ###
        for i in range(n_bin):
            lowpass = pass_list[i] / (sample_rate/2)
            highpass = (pass_list[i]+step) / (sample_rate/2)
            gen_signal = gen_sig.signal_gen(1, seconds, sample_rate, f1, m1, f2, m2, spike_width, ma, spike_interval, delay) # channels = 1 so that it only does it once
            param = [1, seconds, sample_rate, f1, m1, f2, m2, spike_width, ma, spike_interval, delay] # This is kinda stupid but it works for now
            line = gen_sig.calc_mag(offline_directory, 0, lowpass, highpass, order, param=param, signal=gen_signal)
            random_data[i] = line
            print(f'loop {i} is COMPLETE!!!')

        avg_data = pd.DataFrame(columns = ['freq','time', 'amp'])

        for i in range(seconds):
            for j in range(n_bin):
                inst_data = []
                amp_chan = random_data[j, i*sample_rate:(i+1)*sample_rate] # The channel of interest and the associated 30k samples in that second.
                amp_avg = np.average(amp_chan)
                inst_data.append(j*step+start_freq) # What frequency
                inst_data.append(i+1) # What time, account for 0 index
                inst_data.append(amp_avg**2)
                avg_data = pd.concat([pd.DataFrame([inst_data], columns = avg_data.columns), avg_data], ignore_index=True) # pd.concat should be fastest? not 100% sure

        avg_data = avg_data.fillna(0) # It seems to take a second for values to settle / calc to begin
        ax = avg_data.pivot("freq", "time", "amp")

        ax.to_csv(os.path.join(offline_directory, f'{online_rec_name}_startfreq({start_freq})_endfreq({end_freq}_step({step})).csv'))

        # Create a heatmap using seaborn
        plot = sns.heatmap(ax, cmap='viridis', vmin=heat_min, vmax=heat_max)
        plot.set(title="Amplitude² for: {} ({}s)".format(online_rec_name, seconds), xlabel="Time (ms)", ylabel="Frequency")
        plt.show()
        plt.close()

    elif x == '4':
        # Set the size of the array
        # n_bin = 10
        # seconds = 30
        data = []
        sample_rate = 30000
        length = sample_rate * seconds
        order = 2

        # SNS
        heat_min = 0
        heat_max = 800

        # For Bandpass Filter #
        start_freq = 150
        end_freq = 190
        step = 1
        n_bin = int((end_freq-start_freq)/step)

        if (end_freq-start_freq) % step != 0:
            raise Exception('frequency range must be divisible by step size to create integer for bin number. For Example (200-0)/10 = 20 bins')
        
        pass_list = list(range(start_freq, end_freq + 1, step))
        random_data = np.zeros((n_bin, length))
        ### GENERATE OEP DATA ###
        for i in range(n_bin):
            lowpass = pass_list[i] / (sample_rate/2)
            highpass = (pass_list[i]+step) / (sample_rate/2)
            gen_signal = alldata3[:, 0]
            # param = [1, seconds, sample_rate, f1, m1, f2, m2, spike_width, ma, spike_interval, delay] # This is kinda stupid but it works for now
            line = gen_sig.butter_filt(order, lowpass, highpass, gen_signal)
            random_data[i] = line
            print(f'loop {i} is COMPLETE!!!')

        avg_data = pd.DataFrame(columns = ['freq','time', 'amp'])

        for i in range(seconds):
            for j in range(n_bin):
                inst_data = []
                amp_chan = random_data[j, i*sample_rate:(i+1)*sample_rate] # The channel of interest and the associated 30k samples in that second.
                amp_avg = np.average(amp_chan)
                inst_data.append(j*step+start_freq) # What frequency
                inst_data.append(i+1) # What time, account for 0 index
                inst_data.append(amp_avg**2)
                avg_data = pd.concat([pd.DataFrame([inst_data], columns = avg_data.columns), avg_data], ignore_index=True) # pd.concat should be fastest? not 100% sure

        avg_data = avg_data.fillna(0) # It seems to take a second for values to settle / calc to begin
        ax = avg_data.pivot("freq", "time", "amp")

        ax.to_csv(os.path.join(online_directory, f'{online_rec_name}_startfreq({start_freq})_endfreq({end_freq}_step({step}).csv'))

        # Create a heatmap using seaborn
        plot = sns.heatmap(ax, cmap='viridis', vmin=heat_min, vmax=heat_max)
        plot.set(title="Amplitude² for: {} ({}s)".format(online_rec_name, seconds), xlabel="Time (ms)", ylabel="Frequency")
        plot_save_loc = os.path.join(online_directory, f'{online_rec_name}_startfreq({start_freq})_endfreq({end_freq}_step({step})_heatmax({heat_max})).png')
        plt.savefig(plot_save_loc)
        plt.show()
        plt.close()

    elif x == '5': # For loading in data which has already been analyzed
        # SNS
        heat_min = 0
        heat_max = 100

        # For Bandpass Filter #
        start_freq = 100
        end_freq = 200
        step = 20

        ax = pd.read_csv('Z:\\projmon\\addiction_synaptic_plasticity\\ERPs\\SPAF5_Conditioning4_preERP_080123_analyze_pwr_startfreq(100)_endfreq(200_step(20).csv')
        # ax = pd.read_csv(os.path.join(online_directory, f'{online_rec_name}_startfreq({start_freq})_endfreq({end_freq}_step({step}).csv'))

        print(ax)
        plot = sns.heatmap(ax, cmap='viridis', vmin=heat_min, vmax=heat_max)
        plot.set(title="Amplitude² for: {} ({}s)".format(online_rec_name, seconds), xlabel="Time (ms)", ylabel="Frequency")
        plot_save_loc = os.path.join(online_directory, f'{online_rec_name}_startfreq({start_freq})_endfreq({end_freq}_step({step})_heatmax({heat_max})).png')
        plt.savefig(plot_save_loc)
        plt.show()
        plt.close()

    else:
        raise Exception('Input not recognized!')
