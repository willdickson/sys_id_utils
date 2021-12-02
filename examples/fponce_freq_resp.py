import warnings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys_id_utils

# Load data from file
#filename_list = ['fly_example_data.txt', 'fly_example_data_unfilt.txt']
filename_list = ['fly_example_data_unfilt.txt']
color_list = ['r', 'g']

fig1, ax1 = plt.subplots(1,1,num=1)
fig2, (ax2, ax3) = plt.subplots(2,1,sharex=True,num=2)
fig3, (ax4, ax5, ax6) = plt.subplots(3,1,sharex=True,num=3)

for filename, color in zip(filename_list, color_list):

    print(filename)

    data = np.loadtxt(filename)
    t = data[:,0]
    t = np.linspace(t[0], t[-1], t.shape[0])
    input_sig = np.deg2rad(data[:,1])
    output_sig = np.deg2rad(data[:,2])
    num_pts = t.shape[0]
    f0 = 0.1
    f1 = 6.5
    t1 = 20.0

    dt = t[1] - t[0]
    nperseg = num_pts/5
    f_sample = 1.0/(t[1] - t[0])
    f_cutoff = 4.0 
    
    # Compute gain and phase as funtion of frequency
    f, gain_db, phase_deg = sys_id_utils.freq_response(input_sig, output_sig, f_sample, f_cutoff, nperseg)
    gain = 10**(gain_db/20.0)
    phase_rad = np.deg2rad(phase_deg)
    
    f_co, val_co = sp.signal.coherence(input_sig, output_sig, fs=f_sample, nperseg=nperseg)
    mask = f_co <= f_cutoff
    f_co = f_co[mask]
    val_co = val_co[mask]
    f_chirp = f0*(f1/f0)*(t/t1)

    # Fit transfer function
    fit = True 
    f_fit = np.linspace(f0,f_cutoff,1000)
    line_style = '{}'.format(color)

    # Fit state space models
    if fit:
        if 0:
            print(t.shape, input_sig.shape, output_sig.shape)
            d_fit, gp_fit, fit_info = sys_id_utils.fit_p_yaw_model(t, input_sig, output_sig, op_param={'disp':True})
            print('fit: d {}, gp {}'.format(d_fit, gp_fit))
            model_param_fit = {
                    'inertia'  : 1.0,
                    'damping'  : d_fit,
                    'pro_gain' : gp_fit,
                    'int_gain' : 0.0,
                    'int_leak' : 0.0, 
                    'noise'    : 0.0,
                    }
        if 1: 
            d_fit, gp_fit, gi_fit, fit_info = sys_id_utils.fit_pi_yaw_model(t, input_sig, output_sig, op_param={'disp':True})
            print('fit: d {}, gp {}, gi {}'.format(d_fit, gp_fit, gi_fit))
            model_param_fit = {
                    'inertia'  : 1.0,
                    'damping'  : d_fit,
                    'pro_gain' : gp_fit,
                    'int_gain' : gi_fit,
                    'int_leak' : 0.0, 
                    'noise'    : 0.0,
                    }
    
        if 0:
            d_fit, gp_fit, gi_fit, c_fit, fit_info = sys_id_utils.fit_lpi_yaw_model(t, input_sig, output_sig, op_param={'disp':True})
            print('fit: d {}, gp {}, gi {}, c {}'.format(d_fit, gp_fit, gi_fit, c_fit))
            model_param_fit = {
                    'inertia'  : 1.0,
                    'damping'  : d_fit,
                    'pro_gain' : gp_fit,
                    'int_gain' : gi_fit,
                    'int_leak' : c_fit, 
                    'noise'    : 0.0,
                    }
    
    
        ss_model_fit = sys_id_utils.create_lpi_ss_model(model_param_fit)
        #f_fit = f_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, gain_db_fit, phase_deg_fit = sp.signal.bode(ss_model_fit,w=f_fit*2.0*np.pi)
    
    # Plot input and output signals
    h_input, = ax1.plot(f_chirp,input_sig,'b')
    h_output, = ax1.plot(f_chirp, output_sig ,line_style)
    ax1.set_xlabel('freq(Hz)')
    ax1.set_ylabel('velocity (rad/sec)')
    ax1.grid(True)
    fig1.legend((h_input, h_output), ('input', 'output'), 'upper right')
    
    
    # Plot frequency response (Bode plot)
    line_style = 'o{}'.format(color)

    fig2.suptitle('Frequency Response')
    if fit:
        ax2.semilogx(f_fit, gain_db_fit, 'k')
    ax2.semilogx(f, gain_db, line_style)
    ax2.grid(True, which='both', axis='both')
    ax2.set_ylabel('ggain (dB)')
    if fit:
        ax3.semilogx(f_fit, phase_deg_fit, 'k')
    ax3.semilogx(f, phase_deg, line_style)
    ax3.grid(True, which='both', axis='both')
    ax3.set_ylabel('phase lag (deg)')
    ax3.set_xlabel('f (Hz)')
    
    # Plot frequency response (Bode plot)
    fig3.suptitle('Frequency Response')
    ax4.plot(f, gain,line_style)
    ax4.grid(True, which='both', axis='both')
    ax4.set_ylabel('gain')
    
    ax5.plot(f, phase_deg,line_style)
    ax5.grid(True, which='both', axis='both')
    ax5.set_ylabel('phase (deg)')
    
    ax6.plot(f_co, val_co, line_style)
    ax6.grid(True, which='both', axis='both')
    ax6.set_ylabel('coherence')
    ax6.set_xlabel('f (Hz)')
    ax6.set_ylim(0,1.1)


plt.show()
