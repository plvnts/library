def get_model_baseline(signals_loc, ohlc_loc, holding_duration):
    """
    Returns a baseline metric for model based signals generated every tick/second/minute/hour/day
    Signals need to be subset of ohlc, containing same time column
    """
    signals = joblib.load(signals_loc) # A list of signals from the model for each tick/second/minute/hour/day
    ohlc = joblib.load(ohlc_loc)[['time','close']] # OHLC values for each tick/second/minute/hour/day

    ohlc['entry_price'] = ohlc['close']
    ohlc['exit_price'] = ohlc['close'].shift(-holding_duration)
    ohlc['long_r'] = (ohlc['exit_price']-ohlc['entry_price'])/ohlc['entry_price']
    ohlc['short_r'] = (ohlc['entry_price']-ohlc['exit_price'])/ohlc['entry_price']

    signals = signals.merge(ohlc, how='left', on='time')
    signals['r'] = np.where(signals['side']=='LONG', signals['long_r'], signals['short_r'])
    signals['r'] = signals['r']*signals['confidence'] # Multiply with confidence for better weighting
    total_seconds = (signals['time'].iloc[-1]-signals['time'].iloc[0]).total_seconds()
    signals = signals[signals['confidence']>0.5] # Remove noisy/random signals
    metric = signals['r'].sum()/total_seconds*60*60*24*365 # Annualized ROI sum

    return metric
    # sns.set(rc={'figure.figsize':(16,8)})
    # sns.lineplot(x=np.arange(len(signals))[::100], y=np.cumsum(signals['r'])[::100])
