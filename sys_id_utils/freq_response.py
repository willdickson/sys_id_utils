import numpy as np
import scipy.signal


def freq_response(x, y, sample_freq, freq_cutoff, nperseg, gain_in_db=True, phase_in_deg=True):
    """
    Computes the frequency response (gain and phase) from the input data (x) and
    output data (y).

    Arguments:
      x   = array of input data
      y   = array of output data
      sample_freq  = sample frequency 
      cutoff_freq  = cutoff frequency 
      nperseg = length of each segment

    Keyword Arguements:
      gain_in_db  = True/False. If True returns gain in db. 
      gain_in_deg = True/False. If True phase in returned in degrees otherwise
      phase is returned in rads. 

    Return:
      freq  = array of frequencies
      gain  = array of gains
      phase = array of phase lags

    """

    fxx_cpsd, pxx_cpsd = scipy.signal.csd(x, x, sample_freq, 
            window='hann', 
            nperseg=nperseg,
            average='mean'
            )
    fxy_cpsd, pxy_cpsd = scipy.signal.csd(x, y, sample_freq, 
            window='hann', 
            nperseg=nperseg, 
            average='mean'
            )
    freq = fxx_cpsd

    gain = np.absolute(pxy_cpsd)/pxx_cpsd
    if gain_in_db:
        gain = 20.0*np.log10(gain)
        
    phase = np.arctan2(np.imag(pxy_cpsd), np.real(pxy_cpsd))
    if phase_in_deg:
        phase = np.rad2deg(phase)

    mask = freq <= freq_cutoff

    return freq[mask], gain[mask], phase[mask]
