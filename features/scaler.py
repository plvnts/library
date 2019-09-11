import numpy as np

def minmax_scaler(data, split=0.55):
    return (data.T/np.max(data[:int(data.shape[0]*0.55)], axis=0)).T

def standard_scaler(data, split=0.55):
    return (data-np.mean(data[:int(data.shape[0]*0.55)], axis=0))/np.std(data[:int(data.shape[0]*0.55)], axis=0)
