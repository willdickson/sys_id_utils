import warnings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys_id_utils

num_pts = 10000
t0 = 0.0   # Start time
t1 = 20.0  # End time

# Yaw dynamics + controller model parameters
model_param = {
        'inertia'  : 3.0,
        'damping'  : 1.0,
        'pro_gain' : 50.0,
        'int_gain' : 50.0,
        'int_leak' : 0.0, 
        'noise'    : 0.2,
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

fit = False  # If True fits parametric model of transfer function to frequency response
fit_nb = 1   # degree of transfer function numerator
fit_na = 2   # degree of transfer function denominator

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

# Fit transfer function
if fit:
    gain = 10**(gain_db/20.0)
    phase_rad = np.deg2rad(phase_deg)
    b, a = sys_id_utils.fit_tf(f, gain, phase_rad, fit_nb, fit_na, disp=True)
    tf_estimate = sp.signal.TransferFunction(b,a)
    f_fit = f_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, gain_db_fit, phase_deg_fit = sp.signal.bode(tf_estimate, w=f_fit*2.0*np.pi)

# Plot input and output data
fig, ax = plt.subplots(1,1)
h_input, = ax.plot(t,input_sig,'b')
h_output, = ax.plot(t, output_sig ,'r')
ax.set_xlabel('t (sec)')
ax.set_ylabel('velocity (rad/sec)')
ax.grid(True)
plt.figlegend((h_input, h_output), ('input', 'output'), 'upper right')

# Plot frequency response (Bode plot)
fig2, (ax1, ax2) = plt.subplots(2,1,sharex=True)
fig2.suptitle('Frequency Response')

ax1.semilogx(f_model, gain_db_model,'b')
if fit:
    ax1.semilogx(f_fit, gain_db_fit,'g')
ax1.semilogx(f, gain_db,'or')
ax1.grid(True, which='both', axis='both')
ax1.set_ylabel('gain (dB)')

ax2.semilogx(f_model, phase_deg_model,'b')
if fit:
    ax2.semilogx(f_fit, phase_deg_fit, 'g')
ax2.semilogx(f, phase_deg,'or')
ax2.grid(True, which='both', axis='both')
ax2.set_ylabel('phase lag (deg)')
ax2.set_xlabel('f (Hz)')

plt.show()
