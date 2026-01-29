# This code was written in collaboration with Dr. Michael Weber (TROPOS)
import pymannkendall as mk
import numpy as np
import statsmodels.api as sm


def process_array_slope(double[:,:,:] Y, double[:,:] X, double[:, :] slope, double[:, :] p_value,
                        double[:, :] intercept, double[:, :] trend, double[:, :] tau, double[:, :] significance):
   
    cdef size_t i, j, y_lon, x_lon
    cdef double[:] y_sub
    
    x_lat = Y.shape[1]
    y_lon = Y.shape[2]

    for i in range(y_lon):
        for j in range(x_lat):
            y_sub = Y[:,j,i]

            y_arr = np.array(y_sub, dtype=float)
            x_arr = np.array(X, dtype=float)

            mask = ~np.isnan(y_arr)
            n = mask.sum()

            y_clean = y_arr[mask]
            x_clean = x_arr[mask]

            if n >= 10 and not np.allclose(y_clean, y_clean[0]):
                result = mk.original_test(y_clean)
                intercept[j,i] = result.intercept
                slope[j,i] = result.slope
                tau[j,i] = result.Tau
                p_value[j,i] = result.p

                if result.h==False:
                    signif = np.nan
                else:
                    signif = 0.0001 # assign arbitrary small amount
                significance[j,i] = signif

                h = result.trend
                if h == 'increasing':
                    hh = 1
                if h == 'decreasing':
                    hh = -1
                if h == 'no trend':
                    hh = 0
                trend[j,i] = hh

            else:
                slope[j,i] = np.nan
                p_value[j,i] = np.nan
                intercept[j, i] = np.nan
                trend[j,i] = np.nan
                tau[j,i] = np.nan
                significance[j, i] = np.nan

def process_array_slope_per_ice(double[:,:,:] Y, double[:,:,:] X, double[:, :] slope, double[:, :] p_value, double[:, :] intercept):

    cdef size_t i, j, y_lon, x_lon
    cdef double[:] y_sub
    cdef double[:] x_sub

    x_lat = Y.shape[1]
    y_lon = Y.shape[2]

    for i in range(y_lon):
        for j in range(x_lat):
            y_sub = Y[:,j,i]
            x_sub = X[:,j,i]

            y_arr = np.array(y_sub, dtype=float)
            x_arr = np.array(x_sub, dtype=float)

            masky = ~np.isnan(y_arr)
            ny = masky.sum()
            maskx = ~np.isnan(x_arr)
            nx = maskx.sum()

            y_clean = y_arr[masky]
            x_clean = x_arr[maskx]

            if ny >= 2 and not np.allclose(y_clean, y_clean[0]) and nx >= 2 and not np.allclose(x_clean, x_clean[0]) :
                result = sm.OLS(np.array(y_clean), np.array(sm.add_constant(x_clean)), missing='drop').fit()
                intercept[j,i] = result.params[0]
                slope[j, i] = result.params[1]
                pval = result.pvalues[1]

                if pval > 0.05:
                    pval = np.nan
                p_value[j,i] = pval

            else:
                slope[j,i] = np.nan
                p_value[j,i] = np.nan
                intercept[j, i] = np.nan
