### Jon Whear - August 2023 - whear003@umn.edu ###
import pyximport; pyximport.install()
import zmqClasses
import stimmer
import daqAPI
import time

'''
Script used to drive OEP recordings for PC Mag vs. Offline Mag Calculations
Adapted from Simple_CL.py (Mark S.)
''' 

### Constants ###
OE_Address = "localhost"  # localhost or ip address of other computer
record_dir = r"D:\EPHYSDATA\leah" # Change for computer of interest
rec_min = 0.5 # In minutes
condition = 'SPAM3 FreeRun Samples (w PC)' # e.g. 'PC_MAG_CHx_Data_Name' 'RAW_Data_Name' 

### OEP ZMQ Config ###
snd = zmqClasses.SNDEvent(OE_Address,5556, recordingDir = record_dir)

### Record ###
print('starting raw pre')
snd.send(snd.STOP_REC)
snd.changeVars(prependText = condition)
snd.send(snd.START_REC)
time.sleep(rec_min*60) # length of rec
snd.send(snd.STOP_REC)
