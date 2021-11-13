import numpy as np
import scipy as sp

def fit_p_yaw_model(t, inp_sig, out_sig, filt_win=9, filt_ord=3, method='SLSQP'):  
    """
    Finds constrained least squares fit of proportional control yaw dynamics model
    to the proviced input and output data.

    Arguments:
      t        = time values
      inp_sig  = input signal
      out_sig  = output signal

    Keyword Arguments:
      filt_win   = Savitzky-Golay filter window
      filt_ord   = Savitzky-Golay filter order 
      method     = optimizatoin method (SLSQP, COBYLA, trust-constr)

    """
    # Calculate state information from input/output data. 
    dt = t[1] - t[0]
    out_sig_filt, acc_sig, err_sig = p_model_state_from_io(dt, inp_sig,
            out_sig, filt_win=filt_win, filt_ord=filt_ord)

    # Create cost function
    A = np.zeros((t.shape[0], 2))
    A[:,0] = -out_sig
    A[:,1] = err_sig
    b = acc_sig
    def cost_func(x):
        return ((np.dot(A,x) - b)**2).sum()

    # Solve system of equations
    x0 = np.random.rand(2)
    constraints = sp.optimize.LinearConstraint(np.eye(2), np.zeros((2,)), np.full((2,), np.inf))
    res = sp.optimize.minimize(cost_func, x0, method=method, constraints=constraints)

    # Extract model parameters 
    d, gp = res.x[0], res.x[1]

    # Collect information for accessing optimization results 
    info = {
            'out_sig_filt'    : out_sig_filt,
            'acc_sig'         : acc_sig,
            'err_sig'         : err_sig,
            'minimize_result' : res,
            }
    return d, gp, info


def fit_pi_yaw_model(t, inp_sig, out_sig, filt_win=9, filt_ord=3, method='SLSQP'):
    """
    Finds constrained least squares fit of proportional control yaw dynamics model
    to the proviced input and output data.

    Arguments:
      t        = time values
      inp_sig  = input signal
      out_sig  = output signal

    Keyword Arguments:
      filt_win   = Savitzky-Golay filter window
      filt_ord   = Savitzky-Golay filter order 
      method     = optimizatoin method (SLSQP, COBYLA, trust-constr)

    """
    pass



def p_model_state_from_io(dt, inp_sig, out_sig, filt_win=9, filt_ord=3):
    """
    Calculates the state values for the lpi control yaw dynamics model for a
    specific choice of integrator leak coefficient.

    Arguments:
      dt         = time step
      inp_sig    = input signal
      out_sig    = output signal

    Keyword Arguments:
      filt_win   = Savitzky-Golay filter window
      filt_ord   = Savitzky-Golay filter order 

    Return:
      out_sig_filt = filtered output 
      acc_sig      = acceleration signal
      err_sig      = error signal
    """
    out_sig_filt = sp.signal.savgol_filter(out_sig, filt_win, filt_ord, deriv=0)
    acc_sig = sp.signal.savgol_filter(out_sig, filt_win, filt_ord, deriv=1)
    acc_sig = acc_sig/dt 
    err_sig = inp_sig - out_sig
    return out_sig_filt, acc_sig, err_sig


def pi_model_state_from_io(dt, inp_sig, out_sig, filt_win=9, filt_ord=3):
    """
    Calculates the state values for the lpi control yaw dynamics model for a
    specific choice of integrator leak coefficient.

    Arguments:
      dt         = time step
      inp_sig    = input signal
      out_sig    = output signal

    Keyword Arguments:
      filt_win   = Savitzky-Golay filter window
      filt_ord   = Savitzky-Golay filter order 

    Return:
      out_sig_filt = filtered output 
      acc_sig      = acceleration signal
      err_sig      = error signal
      int_sig      = integral signal
    """
    return lpi_model_state_from_io(dt, inp_sig, out_sig, 0.0, filt_win=filt_win, filt_ord=filt_ord)


def lpi_model_state_from_io(dt, inp_sig, out_sig, leak_coeff, filt_win=9, filt_ord=3):
    """
    Calculates the state values for the lpi control yaw dynamics model for a
    specific choice of integrator leak coefficient.

    Arguments:
      dt         = time step
      inp_sig    = input signal
      out_sig    = output signal
      leak_coeff = integrator leak coefficient

    Keyword Arguments:
      filt_win   = Savitzky-Golay filter window
      filt_ord   = Savitzky-Golay filter order 

    Return:
      out_sig_filt = filtered output 
      acc_sig      = acceleration signal
      err_sig      = error signal
      int_sig      = integral signal
    """
    out_sig_filt = sp.signal.savgol_filter(out_sig, filt_win, filt_ord, deriv=0)
    acc_sig = sp.signal.savgol_filter(out_sig, filt_win, filt_ord, deriv=1)
    acc_sig = acc_sig/dt 
    err_sig = inp_sig - out_sig
    int_sig = leaky_integrator(leak_coeff, dt, err_sig) 
    return out_sig_filt, acc_sig, err_sig, int_sig


def leaky_integrator(leak_coeff, dt, err_sig):
    """
    Calculate leaky integral of error signal.

    di/dt = err_sig - leak_coeff*i

    Arguments:
      leak_coeff = leaky integrator coefficient
      dt         = time step
      err_sig    = error signal

    Return:
      int_sig    = leaky integral of error 
    """

    int_sig = np.zeros(err_sig.shape)
    for i in range(1, int_sig.shape[0]):
        int_sig[i] = int_sig[i-1] + dt*(err_sig[i-1] - leak_coeff*int_sig[i-1])
    return int_sig


