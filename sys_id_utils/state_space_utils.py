import numpy as np
import scipy as sp
from . import simulate

def fit_p_yaw_model(t, inp_sig, out_sig, method='min', op_param=None):  
    """
    Finds constrained least squares fit of proportional control yaw dynamics model
    to the provided input and output data.

    Arguments:
      t        = time values
      inp_sig  = input signal
      out_sig  = output signal

    Keyword Arguments:
      method     = optimizatoin method (SLSQP, COBYLA, trust-constr)
      op_param   = parameters specific to selected optimization method 

    """
    if op_param is None:
        user_op_param = {}
    else:
        user_op_param = dict(op_param) 

    def diffeq_cost_func(x):
        model_param = {
                'inertia'  : 1.0, 
                'damping'  : x[0],
                'pro_gain' : x[1],
                }
        sys = simulate.create_p_ss_model(model_param)
        _, out_sig_sim, state_signal = sp.signal.lsim(sys, inp_sig, t, X0=np.array([out_sig[0]]))
        cost = ((out_sig - out_sig_sim)**2).sum()
        print(cost)
        return cost

    constraints = sp.optimize.LinearConstraint(np.eye(2), np.zeros((2,)), np.full((2,), np.inf))

    if method in ('minimize', 'min'):

        x0 = np.random.rand(2)
        minimize_kwargs = {
                'method': 'SLSQP',
                'constraints' : constraints
                }
        minimize_kwargs.update(user_op_param)
        res = sp.optimize.minimize(diffeq_cost_func, x0, method='SLSQP', constraints=constraints)

    elif method in ('basinhopping', 'bh'):

        x0 = np.random.rand(2)
        bh_kwargs = {
                'disp'             : False,
                'stepsize'         : 2.0,
                'T'                : 10.0,
                'minimizer_kwargs' : {'method': 'SLSQP', 'constraints': constraints},
                }
        bh_kwargs.update(user_op_param)
        res = sp.optimize.basinhopping(diffeq_cost_func, x0, **bh_kwargs)

    elif method in ('differential_evolution', 'de'):

        try:
            de_bounds = user_op_param['bounds']
            del user_op_param['bounds']
        except KeyError:
            de_bounds = [(0.0, 1.0e6) for i in range(2)]

        de_kwargs = {
                'disp'        : False,
                'maxiter'     : 100_000,
                'popsize'     : 15, 
                'tol'         : 1.0e-6,
                'constraints' : constraints,
                }
        de_kwargs.update(user_op_param)
        res = sp.optimize.differential_evolution(diffeq_cost_func, de_bounds, **de_kwargs)

    else:
        raise ValueError('unknown optimization method {}'.format(method))

    # Extract model parameters 
    d, gp = res.x[0], res.x[1]
    return d, gp, res 


def fit_pi_yaw_model(t, inp_sig, out_sig, method='min', op_param=None):
    """
    Finds constrained least squares fit of proportional control yaw dynamics model
    to the proviced input and output data.

    Arguments:
      t        = time values
      inp_sig  = input signal
      out_sig  = output signal

    Keyword Arguments:
      method     = optimization method (minimize (min) , basinhopping (bh), differential_evolution (de))
      op_param   = parameters specific to selected optimization method 

    """
    if op_param is None:
        user_op_param = {}
    else:
        user_op_param = dict(op_param) 

    def diffeq_cost_func(x):
        model_param = {
                'inertia'   : 1.0, 
                'damping'  : x[0],
                'pro_gain' : x[1],
                'int_gain' : x[2],
                }
        sys = simulate.create_pi_ss_model(model_param)
        dt = t[1] - t[0]
        _, out_sig_sim, state_signal = sp.signal.lsim(sys, inp_sig, t, X0=np.array([out_sig[0]]))
        cost = ((out_sig - out_sig_sim)**2).sum()
        return cost

    constraints = sp.optimize.LinearConstraint(np.eye(3), np.zeros((3,)), np.full((3,), np.inf))

    if method in ('minimize', 'min'):

        x0 = np.random.rand(3)
        minimize_kwargs = {
                'method': 'SLSQP',
                'constraints' : constraints
                }
        minimize_kwargs.update(user_op_param)
        res = sp.optimize.minimize(diffeq_cost_func, x0, method='SLSQP', constraints=constraints)

    elif method in ('basinhopping', 'bh'):

        x0 = np.random.rand(3)
        bh_kwargs = {
                'disp'             : False,
                'stepsize'         : 2.0,
                'T'                : 10.0,
                'minimizer_kwargs' : {'method': 'SLSQP', 'constraints': constraints},
                }
        bh_kwargs.update(user_op_param)
        res = sp.optimize.basinhopping(diffeq_cost_func, x0, **bh_kwargs)

    elif method in ('differential_evolution', 'de'):

        try:
            de_bounds = user_op_param['bounds']
            del user_op_param['bounds']
        except KeyError:
            de_bounds = [(0.0, 1.0e6) for i in range(3)]

        de_kwargs = {
                'disp'        : False,
                'maxiter'     : 100_000,
                'popsize'     : 15, 
                'tol'         : 1.0e-6,
                'constraints' : constraints,
                }
        de_kwargs.update(user_op_param)
        res = sp.optimize.differential_evolution(diffeq_cost_func, de_bounds, **de_kwargs)

    else:
        raise ValueError('unknown optimization method {}'.format(method))

    # Extract model parameters 
    d, gp, gi = res.x[0], res.x[1], res.x[2] 
    return d, gp, gi, res 


def fit_lpi_yaw_model(t, inp_sig, out_sig, method='min', op_param=None):
    """
    Finds constrained least squares fit of proportional control yaw dynamics model
    to the provided input and output data.

    Arguments:
      t        = time values
      inp_sig  = input signal
      out_sig  = output signal

    Keyword Arguments:
      method     = optimization method (minimize (min) , basinhopping (bh), differential_evolution (de))
      op_param   = parameters specific to selected optimization method 

    """
    if op_param is None:
        user_op_param = {}
    else:
        user_op_param = dict(op_param) 

    def diffeq_cost_func(x):
        model_param = {
                'inertia'  : 1.0, 
                'damping'  : x[0],
                'pro_gain' : x[1],
                'int_gain' : x[2],
                'int_leak' : x[3],
                }
        sys = simulate.create_lpi_ss_model(model_param)
        _, out_sig_sim, state_signal = sp.signal.lsim(sys, inp_sig, t, X0=np.array([out_sig[0]]))
        cost = ((out_sig - out_sig_sim)**2).sum()
        return cost

    constraints = sp.optimize.LinearConstraint(np.eye(4), np.zeros((4,)), np.full((4,), np.inf))

    if method in ('minimize', 'min'):

        x0 = np.random.rand(4)
        minimize_kwargs = {
                'method': 'SLSQP',
                'constraints' : constraints
                }
        minimize_kwargs.update(user_op_param)
        res = sp.optimize.minimize(diffeq_cost_func, x0, method='SLSQP', constraints=constraints)

    elif method in ('basinhopping', 'bh'):

        x0 = np.random.rand(4)
        bh_kwargs = {
                'disp'             : False,
                'stepsize'         : 2.0,
                'T'                : 10.0,
                'minimizer_kwargs' : {'method': 'SLSQP', 'constraints': constraints},
                }
        bh_kwargs.update(user_op_param)
        res = sp.optimize.basinhopping(diffeq_cost_func, x0, **bh_kwargs)

    elif method in ('differential_evolution', 'de'):

        try:
            de_bounds = user_op_param['bounds']
            del user_op_param['bounds']
        except KeyError:
            de_bounds = [(0.0, 1.0e6) for i in range(4)]

        de_kwargs = {
                'disp'        : False,
                'maxiter'     : 100_000,
                'popsize'     : 15, 
                'tol'         : 1.0e-6,
                'constraints' : constraints,
                }
        de_kwargs.update(user_op_param)
        res = sp.optimize.differential_evolution(diffeq_cost_func, de_bounds, **de_kwargs)

    else:
        raise ValueError('unknown optimization method {}'.format(method))

    # Extract model parameters 
    d, gp, gi, c = res.x[0], res.x[1], res.x[2], res.x[3]
    return d, gp, gi, c, res 



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


