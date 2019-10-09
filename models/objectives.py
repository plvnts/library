import numpy as np

def dist_1d_normalized(X, y):
    """
    Calculates distances between two 1d vectors
    Normalizes data
    """
    X_vals = X/np.max(np.abs(X))
    X_vals[np.isnan(X_vals)] = 0
    X_vals[np.isinf(X_vals)] = 0
    y_vals = y/np.max(np.abs(y))
    y_vals[np.isnan(y_vals)] = 0
    y_vals[np.isinf(y_vals)] = 0
    return np.mean(np.abs(X_vals-y_vals))

def dist_1d_standardized(X, y):
    """
    Calculates log distances between two 1d vectors
    Standardizes data
    """
    X_vals = (X-np.mean(X))/np.std(X)
    X_vals[np.isnan(X_vals)] = 0
    X_vals[np.isinf(X_vals)] = 0
    y_vals = (y-np.mean(y))/np.std(y)
    y_vals[np.isnan(y_vals)] = 0
    y_vals[np.isinf(y_vals)] = 0
    return np.mean(np.abs(X_vals-y_vals))

def dist_2d_normalized(X, y):
    """
    Calculates distances between 2d vectors and 1d vectors along horizontal axis
    Normalizes data
    """
    X_vals = (X/np.max(np.abs(X), axis=0)).T
    X_vals[np.isnan(X_vals)] = 0
    X_vals[np.isinf(X_vals)] = 0
    y_vals = y/np.max(np.abs(y), axis=0)
    y_vals[np.isnan(y_vals)] = 0
    y_vals[np.isinf(y_vals)] = 0
    return np.mean(np.abs(X_vals-y_vals).T, axis=0)

def dist_2d_standardized(X, y):
    """
    Calculates distances between 2d vectors and 1d vectors along horizontal axis
    Standardizes data
    """
    X_vals = ((X-np.mean(X, axis=0))/np.std(X, axis=0)).T
    X_vals[np.isnan(X_vals)] = 0
    X_vals[np.isinf(X_vals)] = 0
    y_vals = (y-np.mean(y, axis=0))/np.std(y, axis=0)
    y_vals[np.isnan(y_vals)] = 0
    y_vals[np.isinf(y_vals)] = 0
    return np.mean(np.abs(X_vals-y_vals).T, axis=0)

def ranks_rolling_1d(x, window):
    """
    Calculates rolling window rank of a 1-dim np.array
    """
    x = np.pad(x, (window-1, 0), 'constant') # Pad with window-1
    shape = x.shape[:-1] + (x.shape[-1] - window + 1, window)
    strides = x.strides + (x.strides[-1],)
    vals = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    ranks = vals.argsort(kind='mergesort', axis=1).argsort(kind='mergesort', axis=1)
    return ranks[:,-1]/window+1/window

def rank_obj(y_true, y_pred, window=500, noise_cutoff=0.5):
    y_true_ranks = ranks_rolling_1d(np.abs(y_true), window)
    y_pred_ranks = ranks_rolling_1d(np.abs(y_pred), window)
    y_true_ranks[y_true_ranks < noise_cutoff] = 0
    results = np.log1p(y_pred_ranks*y_true_ranks)*np.sign(y_true)*np.sign(y_pred)
    return np.sum(results)/len(y_true_ranks[y_true_ranks != 0])

def corr_obj(y_true, y_pred, noise_cutoff=0.5):
    return np.corrcoef(y_true, y_pred)[0][1]

def dist_obj(y_true, y_pred):
    return 1/dist_1d_standardized(y_true, y_pred)
