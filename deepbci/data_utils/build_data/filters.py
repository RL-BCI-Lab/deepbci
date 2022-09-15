import numpy as np
from scipy.signal import butter, lfilter, iirnotch, filtfilt
import scipy.linalg

"""
filtfilt is preferred typically, except when doing online filtering then 
use lfilter.

https://dsp.stackexchange.com/questions/19084/applying-filter-in-scipy-signal-use-lfilter-or-filtfilt

"""

def MNF(mixsig):
    xs = np.roll(mixsig.T, 1, axis=0)
    dx = mixsig.T - xs
    #print('xs shape', xs.shape)
    #print('dx shape', dx.shape)

    X2 = np.dot(mixsig, mixsig.T)
    dX2 = np.dot(dx.T, dx)
    #print('X2 shape', X2.shape)
    #print('dX2 shape', dX2.shape)

    #V: selected eigenvalues, in ascending order, each repeated according to its multiplicity.
    #D: The normalized selected eigenvector corresponding to the eigenvalue w[i] is the column v[:,i].
    D, V = scipy.linalg.eigh(dX2, X2) # Matlab returns: V = Right eigenvectors D = eigenvalues
    V = np.round(V, 4) # round to same decimal place as matlab
    diag_D = np.zeros((V.shape[0], V.shape[0]), float)
    np.fill_diagonal(diag_D, np.flip(D, axis=0))
    #print("V shape", V.shape)
    #print("D shape", diag_D.shape)

    Phi = np.dot(mixsig.T, V)
    #U, S, V = gsvd()
    #Phi = np.dot(mixsig.T, V)
    #print('Phi shape', Phi.shape)
    return Phi #diag_D, V

class Filters():
    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order, axis):
        nyq = 0.5 * fs # max frequency, half the smp rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')

        return filtfilt(b, a, data, axis=axis)

    @staticmethod
    def butter_lowpass_filter(data, highcut, fs, order, axis):
        nyq = 0.5 * fs # max frequency, half the smp rate
        high = highcut / nyq
        b, a = butter(order, high, btype='lowpass')

        return filtfilt(b, a, data, axis=axis)

    @staticmethod
    def butter_highpass_filter(data, lowcut, fs, order, axis):
        nyq = 0.5 * fs # max frequency, half the smp rate
        low = lowcut / nyq
        b, a = butter(order, low, btype='highpass')

        return filtfilt(b, a, data, axis=axis)

    @staticmethod
    def notch_filter(data, fs, f0, Q):
        w0 = f0/(fs/2)
        b,a = iirnotch(w0, Q)
        
        return filtfilt(b, a, data)