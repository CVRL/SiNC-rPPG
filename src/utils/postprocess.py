import numpy as np
from scipy import signal


def standardize(wave):
    wave = (wave - np.mean(wave, axis=0)) / np.std(wave, axis=0)
    return wave


def overlap_add(pred_arrs, fpc, step, normed=True, hanning=True):
    plen = int(len(pred_arrs)*step + (fpc-step))
    oadd_arr = np.zeros(plen)
    hanning_window = np.hanning(fpc)
    for j, subj_win in enumerate(pred_arrs):
        ## Define the location of the packet
        start = int(j*step)
        end = start+fpc
        ## Make packet centered at 0 and standardized
        if normed:
            if hanning:
                packet = hanning_window * standardize(subj_win)
            else:
                packet = standardize(subj_win)
        else:
            ## This avoids scaling the pixel-wise waveforms
            bias = np.mean(subj_win, axis=0)
            if hanning:
                packet = (hanning_window * (subj_win - bias)) + bias
            else:
                packet = subj_win
        ## Overlap add the packet
        oadd_arr[start:end] = oadd_arr[start:end] + packet
    ## Center and standardize entire waveform
    if normed:
        oadd_arr = standardize(oadd_arr)
    return oadd_arr


def predict_HR(wave, fps=30, window_size=300, stride=1, maf_width=-1, pad_to_input=False):
    HR = sliding_bpm(wave, fps, window_size=window_size, stride=stride, pad_to_input=pad_to_input)
    if maf_width > 0:
        HR = smooth(HR, maf_width)
    return HR


def smooth(HR, maf_width):
    return np.convolve(np.pad(HR, maf_width//2, 'edge'), np.ones((maf_width))/maf_width, mode='valid')


def estimate_bpm(wave, fps=30, periodogram_window='hamming', nfft=5400, low_hz=0.66666, high_hz=3):
    window = signal.get_window(periodogram_window, wave.shape[0])
    freq, density = signal.periodogram(wave, window=window, fs=fps, nfft=nfft)
    idcs = np.where((freq >= low_hz) & (freq <= high_hz))[0]
    freq = freq[idcs]
    density = density[idcs]
    pulse_freq = freq[np.argmax(density)]
    HR = pulse_freq * 60
    return HR


def resize_to_input(HRs, n, stride, window_size):
    '''
        Returns: signal expanded to size n
    '''
    HRs = np.asarray(HRs)
    HRs = np.repeat(HRs, stride)
    diff = n - HRs.shape[0]
    pad = int(diff / 2)
    first_win = np.repeat(HRs[0], pad)
    last_win = np.repeat(HRs[-1], pad + (diff % 2 == 1))
    HRs = np.hstack((first_win, HRs, last_win))
    return HRs


def sliding_bpm(sig, fps=30, window_size=300, stride=1, low_hz=0.66667, high_hz=3, nfft=5400, pad_to_input=False):
    '''
        Returns: Heart rate prediction with the same size as
                 the input signal.
    '''
    window_size = int(window_size)
    n = sig.shape[0]
    window_count = ((n - window_size) / stride) + 1
    start_idcs = (stride * np.arange(window_count)).astype(int)

    HRs = []
    for s_idx in start_idcs:
        e_idx = s_idx + window_size
        sig_window = sig[s_idx:e_idx]
        sig_window = standardize(sig_window)
        HR = estimate_bpm(sig_window, fps, nfft=nfft, low_hz=low_hz, high_hz=high_hz)
        HRs.append(HR)

    if pad_to_input:
        HRs = resize_to_input(HRs, n, stride, window_size)
        assert(HRs.shape[0] == sig.shape[0])

    return HRs


if __name__ == '__main__':
    main()

