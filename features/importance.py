import numpy as np

def colcorr_calculate(X, y):
    """
    Efficient column-wise correlation
    https://github.com/ikizhvatov/efficient-columnwise-correlation
    https://stackoverflow.com/questions/19401078/efficient-columnwise-correlation-coefficient-calculation
    """
    len_shape = len(y.shape)
    y = y if len_shape > 1 else y.reshape(-1, 1)
    (n, t) = X.shape      # n traces of t samples
    (n_bis, m) = y.shape  # n predictions for each of m candidates

    DO = X - (np.einsum("nt->t", X, optimize='optimal') / np.double(n)) # compute O - mean(O)
    DP = y - (np.einsum("nm->m", y, optimize='optimal') / np.double(n)) # compute P - mean(P)

    cov = np.einsum("nm,nt->mt", DP, DO, optimize='optimal')

    varP = np.einsum("nm,nm->m", DP, DP, optimize='optimal')
    varO = np.einsum("nt,nt->t", DO, DO, optimize='optimal')
    tmp = np.einsum("m,t->mt", varP, varO, optimize='optimal')

    return cov / np.sqrt(tmp) if len_shape > 1 else (cov / np.sqrt(tmp))[0]
