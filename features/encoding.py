import numpy as np
import pandas as pd

def binary(series):
    return (series > 0).astype(int)

def quantile(series, q):
    return pd.qcut(series, q, labels=False, retbins=True)

def sigma(series, sig):
    nbins = np.ceil((np.nanmax(series) - np.nanmin(series)) / sig)
    out = pd.Series(index=series.index, name='sigma')
    for i in range(nbins):
        out[(series >= i*sig) & (series < (i+1)*sig)] = 0
    return out

def percentile(series, min_periods=0, side=None):
    series = series if not side else series[series>0] if side == 'BUY' else series[series<0]
    idx = np.searchsorted(np.sort(np.abs(series)), abs(series[-1]), side='right')
    return idx/len(series) if len(series) >= min_periods else 0
