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
