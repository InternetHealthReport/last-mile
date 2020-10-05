from scipy import signal
import numpy as np
import logging

class Welch(object):
    def __init__(self, nperseg=256, fs=1.0, minRMS=5, noverlap=None):

        self.nperseg = nperseg
        self.fs = fs
        self.pspec = {}
        self.minRMS = minRMS
        self.noverlap = noverlap

    
    def analyze(self, timeseries):
        logging.debug("Frequency analysis...")

        f, Pxx_spec = signal.welch(
                timeseries, self.fs, nperseg=self.nperseg,  scaling="spectrum", 
                detrend="linear", noverlap=self.noverlap, average="median")
        # f, Pxx_spec = signal.periodogram(timeseries, self.fs, scaling="spectrum", detrend="linear")

        # midHigh = np.argwhere(f>1.0/(7*24))
        # f = f[midHigh]
        # Pxx_spec = Pxx_spec[midHigh]
        
        if np.sqrt(Pxx_spec.max())>self.minRMS:
            # self.pspec[link] = (f, Pxx_spec)
            return (f, Pxx_spec)
 
