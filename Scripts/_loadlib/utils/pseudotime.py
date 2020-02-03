import numpy as np
from math import factorial
import pandas as pd
from scipy.interpolate import splrep, splev
import sc_tools as sat

def smooth_data(X, k=3, method='interpolate', window_size=5):
    fp_v = []
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(X, np.ndarray):
        x_val = np.arange(np.shape(X)[0])
        if np.ndim(X) == 1:
            if method == 'interpolate':
                X_out, fp = interpolate(x_val, X, k=k)
            elif method == 'sg':
                X_out = sat.sg_filter(X, order=k, window_size=window_size)
                fp = 0
            fp_v.append(fp)
        else:
            X_out = np.zeros(np.shape(X))
            for col_id in np.arange(np.shape(X)[1]):
                if method == 'interpolate':
                    X_out[:, col_id], fp = interpolate(x_val, X[:,col_id], k=k)
                elif method == 'sg':
                    X_out[:, col_id] = sat.sg_filter(X[:,col_id].values, order=k, window_size=window_size)
                    fp = 0
                fp_v.append(fp)
    elif isinstance(X, pd.DataFrame):
        X_out = X.copy()
        x_val = np.arange(np.shape(X)[0])
        for eid, col in enumerate(X.T.values):
            if method == 'interpolate':
                X_out.iloc[:, eid], fp = interpolate(x_val, col, k=k)
            elif method == 'sg':
                X_out.iloc[:, eid] = sat.sg_filter(col, order=k, window_size=window_size)
                fp = 0
            # X_out.iloc[:,eid], fp = interpolate(x_val, col, k=k)
            fp_v.append(fp)
    else:
        raise TypeError('The type of X is not understood, it must be either array, list, or pandas dataframe')
    return (X_out, np.array(fp_v))

def interpolate(x, y, k=4):
    w = np.repeat(1/np.std(y), np.shape(x)[0])
    tck, fp, ier, msg = splrep(x, y, k=k, w=w, full_output=True)
    new_y = splev(x, tck)
    return (new_y, fp)

