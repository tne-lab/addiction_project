### Jon Whear - July 2023 - whear003@umn.edu ###
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy.signal import hilbert, filtfilt, butter
####
import sys
sys.path.append(r'C:\Users\Windows\anaconda3\envs\open-ephys-python-tools\Lib\site-packages')
from open_ephys.analysis import Session
from tkinter import Tk  # GUI stuff
from tkinter.filedialog import askopenfilename, askdirectory
import json

def signal_gen(channels, seconds, sample_rate, f1, m1, f2, m2, spike_width, ma, spike_interval, delay):
    ### Create Sine Signals ###
    s1 = np.pi * 2 * f1
    s2 = np.pi * 2 * f2

    ### Create the spike signal ###
    spike_signal = np.zeros(sample_rate * seconds)
    spike_center =  delay + (spike_width/2)
    spike_start = int(spike_center - (spike_width/2))
    spike_end = int(spike_center + (spike_width/2)) 
    for i in range(int(seconds / spike_interval)):
        spike_signal[spike_start + i * int(spike_interval * sample_rate):spike_end + i * int(spike_interval * sample_rate)] = ma

    spike_signal = gaussian_filter(spike_signal, sigma=0) # Smooth square to a more realistic curve w/ Gaussian Filter
    plt.plot(spike_signal)
    # plt.show()
    plt.close()

    ### Combine the signals ###
    signal_1 = m1 * np.sin(np.arange(0, seconds*s1, s1/sample_rate))
    signal_2 = m2 * np.sin(np.arange(0, seconds*s2, s2/sample_rate))
    signal_f = signal_1 + signal_2 + spike_signal
    signal_out = signal_f
    for i in range(channels-1):
        signal_out = np.column_stack((signal_out, signal_f))

    return signal_out

def event_gen(seconds, sample_rate, spike_interval):
    event_log = pd.DataFrame(columns=['line', 'sample_number', 'timestamp', 'processor_id', 'stream_index', 'stream_name', 'state'])

    for i in range(seconds):
        inst_data = [
            '1',  # line
            i * sample_rate * spike_interval,  # sample_number
            i,  # timestamp
            100,  # processor_id
            0,  # stream index
            'Rhythm Data',  # stream_name
            1  # state
        ]
        event_log.loc[len(event_log)] = inst_data

        inst_data = [
            '1',  # line
            i * sample_rate * spike_interval + 600,  # sample_number
            i + (600 / sample_rate),  # timestamp
            100,  # processor_id
            0,  # stream index
            'Rhythm Data',  # stream_name
            0  # state
        ]
        event_log.loc[len(event_log)] = inst_data
    return event_log

def calc_mag(directory, coi, lowpass, highpass, order, signal=None, param=None):
    if param is not None:
        channels = param[0]
        seconds = param[1]
        sample_rate = param[2]
        f1 = param[3]
        m1 = param[4]
        f2 = param[5]
        m2 = param[6]
        spike_width = param[7]
        ma = param[8]
        spike_interval = param[9]
        delay = param[10]
        
    if signal is not None: # Leaving as this for now until other options are added
        if isinstance(signal, np.ndarray): # Check for correct data type before continuing
            alldata2 = signal_gen(channels, seconds, sample_rate, f1, m1, f2, m2, spike_width, ma, spike_interval, delay)
            print("""Creating signal with default properties: \nChannels = {}\nSeconds = {}\nSample Rate = {}\nFrequency 1 (Hz) = {}\nMagnitude 1 = {}\nFrequency 2 = {}\nMagnitude 2 = {}\nSpike Width = {}\nSpike Magnitude = {}\nSpike Interval = {}\nSpike Delay = {}""".format(channels, seconds, sample_rate, f1, m1, f2, m2, spike_width, ma, spike_interval, delay))
        else:
            raise ValueError("Invalid data_type. Use 'np.ndarray' for numpy array data_type.")
        
    else:
        session = Session(directory)
        recording_cont = session.recordnodes[0].recordings[0].continuous[0]
        alldata = recording_cont.samples  # A list of lists

        # convert to numpy array
        alldata2 = np.array(alldata)
        alldata2 = alldata2[:, coi] # Should be a lot faster than running HT on all channels...

    magnitude = butter_filt(order, lowpass, highpass, alldata2)
    return magnitude

def butter_filt(order, lowpass, highpass, alldata2):
    ## Bandpass - (lowpass, highpass are in Hz). Order is for butterworth bandpass filter
    [b, a] = butter(order, [lowpass, highpass], btype = 'band') # Butterworth digital and analog filter
    # alldata3_cut = filtfilt(b, a, alldata2_cut)
    alldata3 = filtfilt(b, a, alldata2)

    # datastream object is a list
    datastream = alldata3

    # Apply the Hilbert Transform
    analytic_signal = hilbert(datastream)

    # Compute the instantaneous amplitude (magnitude)
    magnitude = np.abs(analytic_signal)
    return magnitude
    

### RUNNING CODE ###
if __name__ == "__main__":
    # shared variables #
    seconds = 3
    sample_rate = 30000
    spike_interval = 1

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

    # For Bandpass Filter #
    lowpass = 120 / (sample_rate/2)
    highpass = 160 / (sample_rate/2)
    order = 2 # It's 6th order in the cpp code for this freq range

    ### GENERATE OEP DATA ###
    gen_signal = signal_gen(channels, seconds, sample_rate, f1, m1, f2, m2, spike_width, ma, spike_interval, delay)
    print(gen_signal)

    event_data = event_gen(seconds, sample_rate, spike_interval)
    print(event_data)
    directory = r'D:\\EPHYSDATA\\leah\\MAG_TESTING\\SINE\\RAW_OEP_Sine_Wave_30s_2023-07-14_12-51-29'
    online_magnitude = calc_mag(directory, 0, lowpass, highpass, order, signal = gen_signal)
    plt.plot(online_magnitude[:5000]) # first 5k data
    plt.show()
