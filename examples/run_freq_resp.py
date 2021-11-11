import numpy as np
import matplotlib.pyplot as plt
import sys_id_utils

# Load data from file
filename = 'example_data.txt'
data = np.loadtxt(filename)
t = data[:,0]
input_sig = data[:,1]
output_sig = data[:,2]
num_pts = t.shape[0]

nperseg = num_pts/8
f_sample = 1.0/(t[1] - t[0])
f_cutoff = 8.0 

# Compute gain and phase as funtion of frequency
f, gain_db, phase_deg = sys_id_utils.freq_response(input_sig, output_sig, f_sample, f_cutoff, nperseg)

# Plot input and output signals
fig1, ax = plt.subplots(1,1)
h_input, = ax.plot(t,input_sig,'b')
h_output, = ax.plot(t, output_sig ,'r')
ax.set_xlabel('t (sec)')
ax.set_ylabel('velocity (rad/sec)')
ax.grid(True)
plt.figlegend((h_input, h_output), ('input', 'output'), 'upper right')


# Plot frequency response (Bode plot)
fig2, (ax1, ax2) = plt.subplots(2,1,sharex=True)
fig2.suptitle('Frequency Response')
ax1.semilogx(f, gain_db,'or')
ax1.grid(True, which='both', axis='both')
ax1.set_ylabel('gain (dB)')
ax2.semilogx(f, phase_deg,'or')
ax2.grid(True, which='both', axis='both')
ax2.set_ylabel('phase lag (deg)')
ax2.set_xlabel('f (Hz)')


plt.show()

