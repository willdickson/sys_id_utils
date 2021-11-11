import numpy as np
import scipy.optimize as op 


def create_tf(b, a):
    """ 
    Creates a transfer function given vectors of values for the numerator, b[i], and 
    denominator, a[i], 

             b[0]*s**n + b[1]*s**(n-1) + ... + b[n]
      f(s) = --------------------------------------
             a[0]*s**m + a[1]*s**(m-1) + ... + a[m]

    Arguments:
      b = transfer function numerator coefficients
      a = transfer function denominator coefficients

    Return:
     tf = transfer function

    """
    def tf(s):
        num = 0.0*s
        den = 0.0*s
        for i in range(b.shape[0]):
            num += b[i]*s**(b.shape[0]-i-1)
        for i in range(a.shape[0]):
            den += a[i]*s**(a.shape[0]-i-1)
        return num/den
    return tf


def fit_tf(freq, gain, phase, n, m, maxiter=100_000, popsize=300, tol=1.0e-6,
        bounds=None, disp=False, dc_unity_gain=False):
    """
    Fits a rational transfer function, using least squares fit, to arrays gain
    and phase data as a function of frequency. 

    Arguments:
      freq     = array of test frequenceis
      gain     = array of gains
      phase    = array of phase lags
      n        = degree of numerator polynomial in transfer function
      m        = degree of denominator polynomial in transfer function

    Keword Arguments
      maxiter  = maximum number of iterations to perform during optimization
      popsize  = multiplier for setting population size during optimization 
      tol      = Relative tolerance for convergence
      bounds   = bounds for transfer function coefficients
      disp     = True/False, if True diplays evaluated function at each iteration
      dc_unity_gain = True/False, if True dc gain is fixed to unity. 

      Note, if dc_unity_gain is False than len(bounds) = m+n otherwise
      len(bounds) = m+n+1. 

    Return:
      b = transfer function numerator coefficients
      a = transfer function denominator coefficients
    """

    if bounds is None:
        if dc_unity_gain:
            bounds = [(0, 1.0e6) for x in range(n+m)]
        else:
            bounds = [(0, 1.0e6) for x in range(n+m+1)]

    def extract_ba(param):
        """ 
        Exracts the values for transfer function numerator, b[i], and denominator a[i]. 
        """
        if dc_unity_gain:
            b = param[:(n+1)] 
            a = np.ones((m+1,))
            a[1:-1] = param[(n+1):] 
            a[-1] = b[-1]
        else:
            b = param[:(n+1)] 
            a = np.ones((m+1,))
            a[1:] = param[(n+1):] 
        return b, a

    def cost_func(param, freq, gain, n, m):
        """ 
        Create transfer function,  evaluate at sample frequencies and return cost.  
        """
        b, a = extract_ba(param) 
        tf = create_tf(b, a)
        s = 2.0*np.pi*1.0j*freq
        tf_vals = tf(s)
        gain_pred = np.absolute(tf_vals)
        phase_pred = np.arctan2(np.imag(tf_vals), np.real(tf_vals))
        return ((gain - gain_pred)**2).sum() + (((phase - phase_pred)/(2.0*np.pi))**2).sum()

    # Find the best fit using global optimization method (differential evolution).
    res = op.differential_evolution(
            cost_func, 
            bounds, 
            args=(freq,gain,n,m), 
            maxiter=maxiter, 
            popsize=popsize, 
            tol=tol, 
            disp=disp
            )
    b, a = extract_ba(res.x)

    return b, a 



