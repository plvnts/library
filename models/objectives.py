import numpy as np
import pandas as pd

def dist_1d_standardized(X, y):
    X_vals = (X-np.mean(X))/np.std(X)
    X_vals[np.isnan(X_vals)] = 0
    X_vals[np.isinf(X_vals)] = 0
    y_vals = (y-np.mean(y))/np.std(y)
    y_vals[np.isnan(y_vals)] = 0
    y_vals[np.isinf(y_vals)] = 0
    return np.mean(np.abs(X_vals-y_vals))

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

def ranks_rolling_2d(X, window):
    """
    Calculates rolling window rank of a 2-dim np.array series along axis 0 (per column)
    """
    results = np.zeros(X.shape)
    for c in np.arange(X.shape[1]):
        results[:,c] = ranks_rolling_1d(X[:,c], window)
    return results

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

def baseline_obj(signals_loc, ohlc_loc, period=50, noise_cutoff=0.5, plot=True):
    ohlc = joblib.load(ohlc_loc)[['close']]
    ohlc['entry_price'] = ohlc['close']
    ohlc['exit_price'] = ohlc['close'].shift(-period)
    ohlc['long_r'] = (ohlc['exit_price']-ohlc['entry_price'])/ohlc['entry_price']
    ohlc['short_r'] = (ohlc['entry_price']-ohlc['exit_price'])/ohlc['entry_price']
    ohlc['long_r'] = (ohlc['long_r']-np.mean(ohlc['long_r']))/np.std(ohlc['long_r'])
    ohlc['short_r'] = (ohlc['short_r']-np.mean(ohlc['short_r']))/np.std(ohlc['short_r'])

    signals_df = joblib.load(signals_loc)
    signals_df['time'] = signals_df['time'].dt.tz_localize(None).shift(-1)
    signals_df = signals_df.merge(ohlc, how='left', left_on='time', right_index=True)
    signals_df = signals_df[signals_df['percentile']>noise_cutoff]
    signals_df.fillna(0, inplace=True)

    signals_df['cr'] = np.where(signals_df['tpred']>=0, signals_df['long_r'], signals_df['short_r'])
    signals_df['cr'] = signals_df['cr']*signals_df['percentile']
    combined_metric = np.sum(signals_df['cr'])/len(signals_df)*1000
    signals_df['rl'] = np.where(signals_df['tpred']>=0, signals_df['long_r'], 0)
    signals_df['rl'] = signals_df['rl']*signals_df['percentile']
    long_metric = np.sum(signals_df['rl'])/len(signals_df)*1000
    signals_df['sr'] = np.where(signals_df['tpred']>=0, 0, signals_df['short_r'])
    signals_df['sr'] = signals_df['sr']*signals_df['percentile']
    short_metric = np.sum(signals_df['sr'])/len(signals_df)*1000

    if plot:
        sns.set(rc={'figure.figsize':(16,8)})
        sns.lineplot(x=np.arange(len(signals_df))[::100], y=np.cumsum(signals_df['cr'])[::100])
    return combined_metric, long_metric, short_metric
