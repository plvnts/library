import numba as nb
import numpy as np

@nb.jit(forceobj=True)
def percentile_1d(x, window):
    """
    https://stackoverflow.com/questions/53130434/rolling-comparison-between-a-value-and-a-past-window-with-percentile-quantile
    """
    return np.pad(np.array([np.less_equal(x[i:i+window],x[i+window]).sum() for i in range(len(x)-window)],dtype=float)/window, (window, 0), 'constant')

def percentile_2d(x, window, axis=0):
    out = np.zeros(x.shape)
    if axis == 0:
        for i in np.arange(x.shape[1]):
            out[:,i] = percentile_1d(x[:,i], window)
    elif axis == 1:
        for i in np.arange(x.shape[0]):
            out[i] = percentile_1d(x[i], window)
    return out

def percentile_last(x, p, axis=0):
    sorted = (x[:,-p-1:] if axis == 1 else x[-p-1:]).argsort(kind='mergesort', axis=axis).argsort(kind='mergesort', axis=axis)
    return (sorted[:,-1] if axis == 1 else sorted[-1])/p
