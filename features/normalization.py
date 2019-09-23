import numpy as np

def minmax_scaler(data, split=0.55):
    return data/np.max(data[:int(data.shape[0]*split)], axis=0)

def standard_scaler(data, split=0.55):
    return (data-np.mean(data[:int(data.shape[0]*split)], axis=0))/np.std(data[:int(data.shape[0]*split)], axis=0)

def sigmoid_scaler(data):
    return 1/(1+np.exp(-data))

def logit_scaler(data, b):
    return 1/(1+(data/(1-data))**-b)
