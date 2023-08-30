### Dependencies ###
import sys
sys.path.append(r'C:\Users\Windows\anaconda3\envs\open-ephys-python-tools\Lib\site-packages')
from open_ephys.analysis import Session
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pprint import pprint
from scipy import signal
import matplotlib.pyplot as plt

### OEP Tools Array Setup ###
directory = r'C:\Users\Jon\Desktop\MANNY_POWER_2023-07-05_23-59-18' # for example
session = Session(directory)
recording_cont = session.recordnodes[0].recordings[0].continuous[0]
alldata = recording_cont.samples  # A list of lists
alldata2 = np.array(alldata) # Convert to np array
alldata3 = alldata2[:, 3] # 0 indexed, I picked 4 - this ch. is amplitude

### Constants ###
n_bin = 50 # Number of bins in hist
sample_rate = 30000

### Global Stats ###
mean = np.mean(alldata3)
print('Mean: {}'.format(mean))
std = np.std(alldata3)
print('Std: {}'.format(std))

### Functions ###
def create_hist_plot(data):
    plt.hist(data, n_bin)
    plt.show()

def calc_rate(sigma):
    cutoff = mean + std*sigma
    cutoff_data = alldata3[alldata3 > cutoff]
    cutoff_len = len(cutoff_data)
    data_len = len(alldata3)
    stim_per_sec = cutoff_len/data_len*sample_rate
    print('{} Sigma: {} Stimulation Events Per Second'.format(sigma, stim_per_sec))

### Run Analysis ###
for i in np.arange(1, 8, 0.5):
    calc_rate(i)

create_hist_plot(alldata3)
