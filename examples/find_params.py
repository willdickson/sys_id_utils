import warnings
import numpy as np
import scipy as sp
import scipy.signal as sig
import matplotlib.pyplot as plt
import sys_id_utils

def leaky_integrator(leak, dt, err_sig):
    int_sig = np.zeros(err_sig.shape)
    for i in range(1, int_sig.shape[0]):
        int_sig[i] = int_sig[i-1] + dt*(err_sig[i-1] - leak*int_sig[i-1])
    return int_sig


filename = 'fly_example_data_unfilt.txt'
leak_array = np.linspace(0.0,40,500)
data = np.loadtxt(filename)

cost_array = np.zeros(leak_array.shape)
sys_fit_list = []

for i, leak in enumerate(leak_array):

    t = data[:,0]
    inp_sig = data[:,1]
    out_sig = data[:,2]
    num_pts = t.shape[0]
    f0 = 0.1
    f1 = 6.5
    t1 = 20.0
        
    nperseg = num_pts/6
    f_sample = 1.0/(t[1] - t[0])
    f_cutoff = 3.0
        
    f_exp, gain_db_exp, phase_deg_exp = sys_id_utils.freq_response(inp_sig, out_sig, f_sample, f_cutoff, nperseg)
    
    f_chirp = f0*(f1/f0)*(t/t1)
    
    filt_win = 9 
    filt_ord = 3
    out_sig_filt = sig.savgol_filter(out_sig, filt_win, filt_ord, deriv=0)
    acc_sig = sig.savgol_filter(out_sig, filt_win, filt_ord, deriv=1)
    dt = t[1] - t[0]
    acc_sig = acc_sig/dt 
    out_sig_rec = acc_sig.cumsum()*dt
    out_sig_rec = out_sig_rec - out_sig_rec[0] + out_sig_filt[0]
    err_sig = inp_sig - out_sig
    int_sig = leaky_integrator(leak, dt, err_sig) 
    
    mask = f_chirp < f_cutoff
    t = t[mask]
    inp_sig = inp_sig[mask]
    out_sig = out_sig[mask]
    out_sig_filt = out_sig_filt[mask]
    out_sig_rec = out_sig_rec[mask]
    acc_sig = acc_sig[mask]
    err_sig = err_sig[mask]

    #int_sig = err_sig.cumsum()*dt
    int_sig = int_sig[mask]
    
    #Solve linear system for dynamic parameters
    A = np.zeros((t.shape[0], 3))
    A[:,0] = out_sig
    A[:,1] = int_sig
    A[:,2] = inp_sig 
    b = acc_sig
    result = np.linalg.lstsq(A,b,rcond=None)
    param = result[0]
    resid = result[1]
    cost = (np.sqrt(resid)/t.shape[0])[0]
    cost_array[i] = cost
    print('cost = {}'.format(cost))

    A = np.array([[param[0], param[1]], [-1.0, -leak]])
    B = np.array([[param[2]], [1.0]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    sys_fit = sig.StateSpace(A,B,C,D)
    sys_fit_list.append(sys_fit)

fig0, ax0 = plt.subplots(1,1)
ax0.plot(leak_array, cost_array)
ax0.set_xlabel('leak param')
ax0.set_ylabel('mean residual')
ax0.grid(True)

ind_best = cost_array.argmin()
sys_fit = sys_fit_list[ind_best]
print(sys_fit)


f_fit = np.linspace(0.0, 4.0, 1000)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _, gain_db_fit, phase_deg_fit = sp.signal.bode(sys_fit, w=f_fit*2.0*np.pi)

fig1, ax1 = plt.subplots(4,1)
ax1[0].plot(t,inp_sig,'g')
ax1[0].plot(t,out_sig,'b')
ax1[0].plot(t,out_sig_filt,'r')
#ax1[0].plot(t,out_sig_rec,'g')
ax1[0].set_ylabel('vel (deg/sec)')
ax1[0].grid(True)
ax1[1].plot(t, acc_sig, 'r')
ax1[1].set_ylabel('acc (deg/sec**2)')
ax1[1].grid(True)
ax1[2].plot(t, err_sig, 'r')
ax1[2].set_ylabel('err (deg/sec)')
ax1[2].grid(True)
ax1[3].plot(t,int_sig, 'r')
ax1[3].grid(True)
ax1[3].set_ylabel('ierr (deg)')
ax1[3].set_xlabel('t (sec)')

fig2, ax2 = plt.subplots(2,1)
fig2.suptitle('Frequency Response')
ax2[0].semilogx(f_fit, gain_db_fit, 'b')
ax2[0].semilogx(f_exp, gain_db_exp, 'or')
ax2[0].grid(True, which='both', axis='both')
ax2[0].set_ylabel('ggain (dB)')
ax2[1].semilogx(f_fit, phase_deg_fit, 'b')
ax2[1].semilogx(f_exp, phase_deg_exp, 'or')
ax2[1].grid(True, which='both', axis='both')
ax2[1].set_ylabel('phase lag (deg)')
ax2[1].set_xlabel('f (Hz)')
plt.show()
