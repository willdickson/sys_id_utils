import numpy as np
import matplotlib.pyplot as plt
import sys_id_utils

t0 = 0.0
t1 = 20.0
num_pts = 10000
save_data = True 
filename = 'example_data.txt'

model_param = {
        'inertia'  : 3.0,
        'damping'  : 1.0,
        'pro_gain' : 50.0,
        'int_gain' : 50.0,
        'int_leak' : 0.0, 
        'noise'    : 0.2,
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

t, input_sig, output_sig, state = sys_id_utils.lpi_yaw_model(t0, t1, num_pts, model_param, input_param_chrip)

if save_data:
    data = np.zeros((num_pts,5))
    data[:,0] = t
    data[:,1] = input_sig
    data[:,2] = output_sig
    data[:,3:] = state
    np.savetxt(filename, data)


# Plot input and output signals
fig1, ax = plt.subplots(1,1)
h_input, = ax.plot(t,input_sig,'b')
h_output, = ax.plot(t, output_sig ,'r')
ax.set_xlabel('t (sec)')
ax.set_ylabel('velocity (rad/sec)')
ax.grid(True)
plt.figlegend((h_input, h_output), ('input', 'output'), 'upper right')

plt.show()

