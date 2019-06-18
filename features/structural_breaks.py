import numpy as np
import pandas as pd

def recursive_residuals(x, y, skip=None, lamda=0.0, alpha=0.95):
    '''calculate recursive ols with residuals and cusum test statistic
    Parameters
    ----------
    x : feature dataframe
    y : target variable series
    skip : int or None
        number of observations to use for initial OLS, if None then skip is
        set equal to the number of regressors (columns in exog)
    lamda : float
        weight for Ridge correction to initial (X'X)^{-1}
    alpha : {0.95, 0.99}
        confidence level of test, currently only two values supported,
        used for confidence interval in cusum graph
    Returns
    -------
    rresid : array
        recursive ols residuals
    rparams : array
        recursive ols parameter estimates
    rypred : arrayHomm and Breitung [2012]
        recursive prediction of endogenous variable
    rresid_standardized : array
        recursive residuals standardized so that N(0,sigma2) distributed, where
        sigma2 is the error variance
    rresid_scaled : array
        recursive residuals normalize so that N(0,1) distributed
    rcusum : array
        cumulative residuals for cusum test
    rcusumci : array
        confidence interval for cusum test, currently hard coded for alpha=0.95
    Notes
    -----
    It produces same recursive residuals as other version. This version updates
    the inverse of the X'X matrix and does not require matrix inversion during
    updating. looks efficient but no timing
    Confidence interval in Greene and Brown, Durbin and Evans is the same as
    in Ploberger after a little bit of algebra.
    References
    ----------
    jplv to check formulas, follows Harvey
    BigJudge 5.5.2b for formula for inverse(X'X) updating
    Greene section 7.5.2
    Brown, R. L., J. Durbin, and J. M. Evans. “Techniques for Testing the
    Constancy of Regression Relationships over Time.”
    Journal of the Royal Statistical Society. Series B (Methodological) 37,
    no. 2 (1975): 149-192.
    '''

    index = y.index
    x, y = np.array(x), np.array(y)
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    nobs, nvars = x.shape
    if skip is None:
        skip = nvars
    rparams = np.nan * np.zeros((nobs,nvars))
    rresid = np.nan * np.zeros((nobs))
    rypred = np.nan * np.zeros((nobs))
    rvarraw = np.nan * np.zeros((nobs))


    #intialize with skip observations
    x0 = x[:skip]
    y0 = y[:skip]
    #add Ridge to start (not in jplv
    XTXi = np.linalg.inv(np.dot(x0.T, x0)+lamda*np.eye(nvars))
    XTY = np.dot(x0.T, y0) #xi * y   #np.dot(xi, y)
    #beta = np.linalg.solve(XTX, XTY)
    beta = np.dot(XTXi, XTY)
    #print('beta', beta
    rparams[skip-1] = beta
    yipred = np.dot(x[skip-1], beta)
    rypred[skip-1] = yipred
    rresid[skip-1] = y[skip-1] - yipred
    rvarraw[skip-1] = 1 + np.dot(x[skip-1],np.dot(XTXi, x[skip-1]))
    for i in range(skip,nobs):
        xi = x[i:i+1,:]
        yi = y[i]
        #xxT = np.dot(xi.T, xi)  #xi is 2d 1 row
        xy = (xi*yi).ravel() # XTY is 1d  #np.dot(xi, yi)   #np.dot(xi, y)
        #print(xy.shape, XTY.shape
        #print(XTX
        #print(XTY

        # get prediction error with previous beta
        yipred = np.dot(xi, beta)
        rypred[i] = yipred
        residi = yi - yipred
        rresid[i] = residi

        #update beta and inverse(X'X)
        tmp = np.dot(XTXi, xi.T)
        ft = 1 + np.dot(xi, tmp)

        XTXi = XTXi - np.dot(tmp, tmp.T) / ft  #BigJudge equ 5.5.15

        #print('beta', beta
        beta = beta + (tmp*residi / ft).ravel()  #BigJudge equ 5.5.14
#        #version for testing
#        XTY += xy
#        beta = np.dot(XTXi, XTY)
#        print((tmp*yipred / ft).shape
#        print('tmp.shape, ft.shape, beta.shape', tmp.shape, ft.shape, beta.shape
        rparams[i] = beta
        rvarraw[i] = ft



    i = nobs
    #beta = np.linalg.solve(XTX, XTY)
    #rparams[i] = beta

    rresid_scaled = rresid/np.sqrt(rvarraw)   #this is N(0,sigma2) distributed
    nrr = nobs-skip
    #sigma2 = rresid_scaled[skip-1:].var(ddof=1)  #var or sum of squares ?
            #Greene has var, jplv and Ploberger have sum of squares (Ass.:mean=0)
    #Gretl uses: by reverse engineering matching their numbers
    sigma2 = rresid_scaled[skip:].var(ddof=1)
    rresid_standardized = rresid_scaled/np.sqrt(sigma2) #N(0,1) distributed
    rcusum = np.concatenate((rresid_standardized[:(skip-1)], rresid_standardized[(skip-1):].cumsum()))
    #confidence interval points in Greene p136 looks strange. Cleared up
    #this assumes sum of independent standard normal, which does not take into
    #account that we make many tests at the same time
    #rcusumci = np.sqrt(np.arange(skip,nobs+1))*np.array([[-1.],[+1.]])*stats.norm.sf(0.025)
    if alpha == 0.95:
        a = 0.948 #for alpha=0.95
    elif alpha == 0.99:
        a = 1.143 #for alpha=0.99
    elif alpha == 0.90:
        a = 0.850
    else:
        raise ValueError('alpha can only be 0.9, 0.95 or 0.99')

    #following taken from Ploberger,
    crit = a*np.sqrt(nrr)
    rcusumci = (a*np.sqrt(nrr) + 2*a*np.arange(0,nobs-skip)/np.sqrt(nrr)) \
                 * np.array([[-1.],[+1.]])
    #return (rresid, rparams, rypred, rresid_standardized, rresid_scaled, rcusum, rcusumci)
    return pd.Series(rcusum, index=index, name='rcusum')


def chu_stin_white(y):
    """
    Computes the maximum and minimum standardized departures of log-price yt for t in [0, t)
    Follows Homm and Breitung [2012]
    :param y: target variable series y
    :return: test statistic series
    """

    s_max = pd.Series(index=y.index, name='s_max')
    s_min = pd.Series(index=y.index, name='s_min')

    y = np.log(y).values
    t = np.arange(1, len(y))
    sigma = np.concatenate(([np.nan], np.sqrt((1/t) * np.nancumsum(np.diff(y)**2))))

    for i in t:
        s = (y[i] - y[:i]) * (1 / (sigma[i] * np.sqrt(i - t[:i] + 1)))
        s_max.iloc[i] = np.nanmax(s)
        s_min.iloc[i] = np.nanmin(s)
    return pd.concat((s_min, s_max), axis=1)
