### Jon Whear - August 2023 - whear003@umn.edu ###

import os
import sys
sys.path.append(r'C:\Users\Windows\anaconda3\envs\open-ephys-python-tools\Lib\site-packages')
from open_ephys.analysis import Session
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pprint import pprint
from tkinter import Tk, filedialog
from scipy import signal
import matplotlib.pyplot as plt
import gen_sig

# Constants. Some could be user input or inferred from data
SAMPLES_BEFORE = 100  # samples
SAMPLES_AFTER = 100  # samples
CHANNELS = 16  # Should usually be 16, could be a [list] if not all wires used
X_LABEL_NUM = 5 # Number of labels. Should be odd so that 0 is in the middle.
PLOTS_PER_ROW = 4 
NOTCH_FILTER = False  # Toggle notch filter on/off
GEN_SIG = True
NOTCH_FILTER_FREQ = 60.0  # In Hz, frequency to be filtered out (Notch)
Q = 1.0  # Quality Factor, lower numbers cut out a wider frequency band
SAMPLERATE = 30000

def open_session():
    Tk().withdraw()  # we don't want a full GUI, block root window
    directory = filedialog.askdirectory(title='Path to Data ')
    directory = directory.replace('/', '\\')

    # Create session object
    session = Session(directory)
    recordnode = session.recordnodes[0]
    recording = recordnode.recordings[0]

    # pandas dataframe of events timestamps, samples, states
    eventtable = recording.events
    print(f"OEP EVENTS: {eventtable}")
    mask = eventtable['state'] == 1  # Finds all event ON times
    numberExperiments = len(session.recordnodes[0].recordings)  # How many experiments for this date
    print(f"numberExperiments: ({numberExperiments})")
    zerotimes = eventtable[mask]['sample_number']
    return directory, SAMPLERATE, session, zerotimes

def filter_oep_data(SAMPLERATE):
    b_notch, a_notch = signal.iirnotch(NOTCH_FILTER_FREQ, Q, SAMPLERATE)  # Design a notch filter using signal.iirnotch
    return b_notch, a_notch

# MAIN ANALYSIS FUNCTION
def average_oep_data(index, zerotimes):
    alldata = np.array(session.recordnodes[0].recordings[index].continuous[0].samples)  # datastream object is a list
    order = 2

    # For Bandpass Filter
    start_freq = 100
    end_freq = 200
    step = 20
    avg_bin = 10 # Number of samples averaged into each bin
    y_bins = int((end_freq-start_freq)/step)
    
    if (end_freq-start_freq) % step != 0:
        raise Exception('frequency range must be divisible by step size to create integer for bin number. For Example (200-0)/10 = 20 bins')
    
    pass_list = list(range(start_freq, end_freq + 1, step))

    pwr_data = [] # pd.DataFrame(columns = ['chan', 'freq', 'time', 'pwr'])
    for channel in range(CHANNELS):
        for index, zero in enumerate(zerotimes[:-1]):  # don't use last trace because it's often truncated
            trace_start = zero - SAMPLES_BEFORE
            trace_end = zero + SAMPLES_AFTER

            # Extract data between the timestamps as ndarray
            cd = alldata[trace_start:trace_end, channel]  # channel data

            ### GENERATE OEP DATA ###
            avg_pwr = []
            for i in range(y_bins):
                lowpass = pass_list[i] / (SAMPLERATE/2)
                highpass = (pass_list[i]+step) / (SAMPLERATE/2)
                trace_pwr = gen_sig.butter_filt(order, lowpass, highpass, cd)**2
                x_bins = int(len(trace_pwr)/avg_bin)

            for j in range(x_bins):
                inst_avg = trace_pwr[j*avg_bin:(j+1)*avg_bin]
                pwr_avg = np.average(inst_avg)
                avg_pwr.append(pwr_avg)
                
            for k in range(len(avg_pwr)): # Channel data is denoted by which line data is on
                inst_data = [channel, i * step + start_freq, k, avg_pwr[k]]
                pwr_data.append(inst_data)
                
            print(f'Analyzing - Channel: {channel+1} of {CHANNELS}, Zerotime: {index+1} of {len(zerotimes)}, n_bin: {i+1} of {y_bins}')
    
    pwr_data = pd.DataFrame(pwr_data, columns = ['chan', 'freq', 'time', 'pwr'])
    pwr_data.to_csv(os.path.join(directory, f'{directory}_analyze_pwr_startfreq({start_freq})_endfreq({end_freq}_step({step}).csv'))

    return pwr_data, avg_bin

def plot16(oLFP_data, recnum, path, SAMPLERATE, avg_bin):
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    fig.subplots_adjust(top=5)

    for CHANNEL in range(CHANNELS):
        # Filter data for the current channel
        oLFP_data_channel = oLFP_data[oLFP_data['chan'] == CHANNEL][['freq', 'time', 'pwr']]
        oLFP_data_pivot = oLFP_data_channel.pivot_table(index='freq', columns='time', values='pwr')
        heatmap_data = oLFP_data_pivot.values.tolist()

        heat_min = 0 # For Heatmap
        heat_max = 20 # For Heatmap

        # Create the subplot for the heatmap
        r = CHANNEL // PLOTS_PER_ROW
        c = CHANNEL % PLOTS_PER_ROW
        ax[r][c].imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=heat_min, vmax=heat_max)
        title = f'Channel {CHANNEL + 1}'
        ax[r][c].set_title(title)
        ax[r][c].set_xlabel('Time (ms)')
        ax[r][c].set_ylabel('Frequency Band')

        # Convert samples to milliseconds
        pwr_len = len(heatmap_data[0])
        sample_ticks = np.arange(0, pwr_len, pwr_len/(pwr_len/X_LABEL_NUM))

        ms_ticks_lists = np.arange(-pwr_len/2, pwr_len/2, pwr_len/(pwr_len/X_LABEL_NUM))
        ms_ticks =  ms_ticks_lists / SAMPLERATE * 1000 * avg_bin # To account for averaging before

        ax[r][c].set_xticks(sample_ticks)
        ax[r][c].set_xticklabels(["{:.1f}".format(ms_tick) for ms_tick in ms_ticks])

        # Draw a vertical line at stimulation time
        ax[r][c].axvline(x=pwr_len/2, color='black', ls='-', lw=0.5) # Assumes before = after!

        # Figure row,column housekeeping
        c += 1
        if c % PLOTS_PER_ROW == 0:
            r += 1
            c = 0

    # Save the figure to a file
    os.chdir(path)
    filename = os.path.basename(path)
    print(path, filename)
    savename = f'{filename}_heatmap.png'
    TEMP_TITLE = filename + " EXPERIMENT " + str(recnum + 1) + " Opto " + " ms"
    fig.suptitle(TEMP_TITLE, fontsize=24) # Need recording number too
    plt.tight_layout()
    plt.show()
    fig.savefig(savename, dpi=300, format='png', metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None
                )
    plt.close()
    return fig

# Run analysis function for each experiment in Session
directory, SAMPLERATE, session, zerotimes = open_session()
b_notch, a_notch = filter_oep_data(SAMPLERATE)
pwr_data, avg_bin = average_oep_data(0, zerotimes)

# Plot heatmap per channel
results = plot16(pwr_data, 0, directory, SAMPLERATE, avg_bin)
print("*** DONE WITH RECORDING ")
