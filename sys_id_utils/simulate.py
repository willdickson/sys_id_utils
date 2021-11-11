import numpy as np
import scipy.signal

def create_lpi_ss_model(model_param):
    """
    Creates a state space model LPI yaw dynamics model 
    """
    I  = model_param['inertia']
    d  = model_param['damping']
    gp = model_param['pro_gain']
    gi = model_param['int_gain']
    b  = model_param['int_leak']
    A = np.array([[-(d + gp)/I, gi/I], [-1.0, -b]])
    B = np.array([[gp/I], [1.0]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    sys = scipy.signal.StateSpace(A,B,C,D)
    return sys


def lpi_yaw_model(t_start, t_final, num_pts, model_param, input_param):
    """
    Simulate LPI yaw dynamics model for fly yaw dynamics.  

    Arguments:
      t_start = simulation start time 
      t_final = simulation stop  time 
      num_pts = number of time steps
      model_param = dictionary of model params
      input_param = dictionary of input signal params

    Return:
      t             = array of simulation time points
      input_signal  = array of input signal values
      output_signal = array of output signal values 
      state_signal  = array of state values

    """
    t = np.linspace(t_start, t_final, num_pts)
    sys = create_lpi_ss_model(model_param)

    # Create input signal
    if input_param['type'] == 'chirp':
        method = input_param['method']
        phi = np.rad2deg(np.pi/2.0)
        f0 = input_param['min_freq']
        f1 = input_param['max_freq']
        t1 = t[-1]
        input_signal = scipy.signal.chirp(t, f0, t1, f1, method=method, phi=phi)
    elif input_param['type'] == 'step':
        u0 = input_param['begin_value']
        u1 = input_param['final_value']
        t_step = input_param['t_step']
        input_signal = np.zeros(t.shape,dtype=np.float)
        input_signal[t <= t_step] = input_param['value_start']
        input_signal[t <  t_step] = input_param['value_final'] 
    else:
        raise ValueError('unknown input signal type {}'.format(input_param['type']))

    # Add simulated noise to input
    if model_param['noise'] is not None:
        input_signal_w_noise = input_signal + model_param['noise']*np.random.randn(input_signal.shape[0])

    # Simulate system
    _, output_signal, state_signal = scipy.signal.lsim(sys, input_signal_w_noise, t)

    return t, input_signal, output_signal, state_signal
    


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    t0 = 0.0
    t1 = 20.0
    num_pts = 10000

    model_param = {
            'inertia'  : 3.0,
            'damping'  : 1.0,
            'pro_gain' : 50.0,
            'int_gain' : 50.0,
            'int_leak' : 0.0, 
            'noise'    : 0.0,
            }

    input_param_chrip = {
            'type'     : 'chirp',
            'method'   : 'logarithmic',
            'min_freq' : 0.1, 
            'max_freq' : 10.0,
            }

    input_param_step = {
            'type'        : 'step',
            'value_start' :  0.0, 
            'value_final' :  1.0,
            't_step'      :  10.0,  
            }

    t, input_sig, output_sig, state = lpi_yaw_model(t0, t1, num_pts, model_param, input_param_chrip)

    fig, ax = plt.subplots(1,1)
    h_input, = ax.plot(t,input_sig,'b')
    h_output, = ax.plot(t, output_sig ,'r')
    ax.set_xlabel('t (sec)')
    ax.set_ylabel('velocity (rad/sec)')
    ax.grid(True)
    plt.figlegend((h_input, h_output), ('input', 'output'), 'upper right')
    plt.show()




            


