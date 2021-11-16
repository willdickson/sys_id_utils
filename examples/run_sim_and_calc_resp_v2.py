import warnings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys_id_utils

num_pts = 2000 
t0 = 0.0   # Start time
t1 = 20.0  # End time

# Yaw dynamics + controller model parameters
model_param = {
        'inertia'  : 1.0,
        'damping'  : 1.0,
        'pro_gain' : 10.0,
        'int_gain' : 0.0,
        'int_leak' : 0.0, 
        'noise'    : 0.1,
        }

# Input signal parameters for chirp function
input_param_chrip = {
        'type'     : 'chirp',
        'method'   : 'logarithmic',
        'min_freq' : 0.1, 
        'max_freq' : 10.0,
        }

# Input signal parameters for step function
input_param_step = {
        'type'        : 'step',
        'value_start' :  0.0, 
        'value_final' :  1.0,
        't_step'      :  10.0,  
        }

nperseg = num_pts/8 # Number of points per segment for power spectral density calculation
f_cutoff = 8.0      # Cut off frequency for analysis

fit = True  # If True fits parametric model of transfer function to frequency response

# Create input and output data
t, input_sig, output_sig, state = sys_id_utils.lpi_yaw_model(t0, t1, num_pts, model_param, input_param_chrip)
f_sample = 1.0/(t[1] - t[0])

# Compute gain and phase as funtion of frequency from input/output data - as if we were performing an experiment.
f, gain_db, phase_deg = sys_id_utils.freq_response(input_sig, output_sig, f_sample, f_cutoff, nperseg)

# Create state space model to get theoretical frequncy response 
ss_model = sys_id_utils.create_lpi_ss_model(model_param)
f_model = np.linspace(0.0, f_cutoff, num_pts)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _, gain_db_model, phase_deg_model = sp.signal.bode(ss_model,w=f_model*2.0*np.pi)

# Fit state space models
if fit:
    if 1:
        d_fit, gp_fit, fit_info = sys_id_utils.fit_p_yaw_model(t, input_sig, output_sig, op_param={'disp':True})
        print('fit: d {}, gp {}'.format(d_fit, gp_fit))
        model_param_fit = {
                'inertia'  : model_param['inertia'],
                'damping'  : model_param['inertia']*d_fit,
                'pro_gain' : model_param['inertia']*gp_fit,
                'int_gain' : 0.0,
                'int_leak' : 0.0, 
                'noise'    : 0.0,
                }
    if 0: 
        d_fit, gp_fit, gi_fit, fit_info = sys_id_utils.fit_pi_yaw_model(t, input_sig, output_sig, op_param={'disp':True})
        print('fit: d {}, gp {}, gi {}'.format(d_fit, gp_fit, gi_fit))
        model_param_fit = {
                'inertia'  : model_param['inertia'],
                'damping'  : model_param['inertia']*d_fit,
                'pro_gain' : model_param['inertia']*gp_fit,
                'int_gain' : model_param['inertia']*gi_fit,
                'int_leak' : 0.0, 
                'noise'    : 0.0,
                }

    if 0:
        d_fit, gp_fit, gi_fit, c_fit, fit_info = sys_id_utils.fit_lpi_yaw_model(t, input_sig, output_sig, op_param={'disp':True})
        print('fit: d {}, gp {}, gi {}, c {}'.format(d_fit, gp_fit, gi_fit, c_fit))
        model_param_fit = {
                'inertia'  : model_param['inertia'],
                'damping'  : model_param['inertia']*d_fit,
                'pro_gain' : model_param['inertia']*gp_fit,
                'int_gain' : model_param['inertia']*gi_fit,
                'int_leak' : c_fit, 
                'noise'    : 0.0,
                }


    ss_model_fit = sys_id_utils.create_lpi_ss_model(model_param_fit)
    f_fit = f_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, gain_db_fit, phase_deg_fit = sp.signal.bode(ss_model_fit,w=f_fit*2.0*np.pi)

# Plot input and output data
fig1, ax1 = plt.subplots(1,1)
h_input, = ax1.plot(t,input_sig,'b')
h_output, = ax1.plot(t, output_sig ,'r')
ax1.set_xlabel('t (sec)')
ax1.set_ylabel('velocity (rad/sec)')
ax1.grid(True)
plt.figlegend((h_input, h_output), ('input', 'output'), 'upper right')

# Plot frequency response (Bode plot)
fig2, ax2 = plt.subplots(2,1,sharex=True)
fig2.suptitle('Frequency Response')

ax2[0].semilogx(f_model, gain_db_model,'b')
if fit:
    ax2[0].semilogx(f_fit, gain_db_fit,'g')
ax2[0].semilogx(f, gain_db,'or')
ax2[0].grid(True, which='both', axis='both')
ax2[0].set_ylabel('gain (dB)')

ax2[1].semilogx(f_model, phase_deg_model,'b')
if fit:
    ax2[1].semilogx(f_fit, phase_deg_fit, 'g')
ax2[1].semilogx(f, phase_deg,'or')
ax2[1].grid(True, which='both', axis='both')
ax2[1].set_ylabel('phase lag (deg)')
ax2[1].set_xlabel('f (Hz)')

plt.show()
