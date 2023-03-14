import torch
import torch.fft as fft

EPSILON = 1e-5
BP_LOW=2/3
BP_HIGH=3.0
BP_DELTA=0.1

def select_loss(arg_obj):
    losses = arg_obj.losses

    criterion_funcs = {
            'ipr': IPR_SSL, ## bandwidth loss
            'snr': SNR_SSL, ## sparsity loss
            'emd': EMD_SSL, ## variance loss
            'snrharm': SNR_harmonic_SSL, ## sparsity with harmonics (not recommended)
            'normnp': NP_SUPERVISED ## supervised negative pearson loss
    }

    criterions = {}
    if losses == "supervised":
        criterions['supervised'] = criterion_funcs[arg_obj.supervised_loss]
    elif losses == "supervised_priors":
        criterions['supervised'] = criterion_funcs[arg_obj.supervised_loss]
        criterions['bandwidth'] = criterion_funcs[arg_obj.bandwidth_loss]
        criterions['sparsity'] = criterion_funcs[arg_obj.sparsity_loss]
    else:
        if 'b' in losses:
            criterions['bandwidth'] = criterion_funcs[arg_obj.bandwidth_loss]
        if 's' in losses:
            criterions['sparsity'] = criterion_funcs[arg_obj.sparsity_loss]
        if 'v' in losses:
            criterions['variance'] = criterion_funcs[arg_obj.variance_loss]

    return criterions


def select_validation_loss(arg_obj):
    validation_loss = arg_obj.validation_loss
    criterion_funcs = {
            'ipr': IPR_SSL, ## bandwidth loss
            'snr': SNR_SSL, ## sparsity loss
            'emd': EMD_SSL, ## variance loss
            'snrharm': SNR_harmonic_SSL, ## sparsity with harmonics (not recommended)
            'normnp': NP_SUPERVISED ## supervised negative pearson loss
    }

    criterions = {}
    if validation_loss == "supervised":
        criterions['supervised'] = criterion_funcs[arg_obj.supervised_loss]
    elif validation_loss == "supervised_priors":
        criterions['supervised'] = criterion_funcs[arg_obj.supervised_loss]
        criterions['bandwidth'] = criterion_funcs[arg_obj.bandwidth_loss]
        criterions['sparsity'] = criterion_funcs[arg_obj.sparsity_loss]
    else:
        if 'b' in validation_loss:
            criterions['bandwidth'] = criterion_funcs[arg_obj.bandwidth_loss]
        if 's' in validation_loss:
            criterions['sparsity'] = criterion_funcs[arg_obj.sparsity_loss]
    return criterions


def _IPR_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, device=None):
    use_freqs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    zero_freqs = torch.logical_not(use_freqs)
    use_energy = torch.sum(psd[:,use_freqs], dim=1)
    zero_energy = torch.sum(psd[:,zero_freqs], dim=1)
    denom = use_energy + zero_energy + EPSILON
    ipr_loss = torch.mean(zero_energy / denom)
    return ipr_loss


def IPR_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, device=None):
    if speed is None:
        ipr_loss = _IPR_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, device=device)
    else:
        batch_size = psd.shape[0]
        ipr_losses = torch.ones((batch_size,1)).to(device)
        for b in range(batch_size):
            low_hz_b = low_hz * speed[b]
            high_hz_b = high_hz * speed[b]
            ipr_losses[b] = _IPR_SSL(freqs, psd[b].view(1,-1), low_hz=low_hz_b, high_hz=high_hz_b, device=device)
        ipr_loss = torch.mean(ipr_losses)
    return ipr_loss


def _EMD_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, normalized=False, bandpassed=False, device=None):
    ''' Squared earth mover's distance to uniform distribution.
    '''
    if not bandpassed:
        freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz)
    if not normalized:
        psd = normalize_psd(psd)
    B,T = psd.shape
    psd = torch.sum(psd, dim=0) / B
    expected = ((1/T)*torch.ones(T)).to(device) #uniform distribution
    emd_loss = torch.mean(torch.square(torch.cumsum(psd, dim=0) - torch.cumsum(expected, dim=0)))
    return emd_loss


def EMD_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, normalized=False, bandpassed=False, device=None):
    ''' Squared earth movers distance to uniform distribution.
    '''
    if speed is None:
        emd_loss = _EMD_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, normalized=normalized, bandpassed=bandpassed, device=device)
    else:
        B = psd.shape[0]
        expected = torch.zeros_like(freqs).to(device)
        for b in range(B):
            speed_b = speed[b]
            low_hz_b = low_hz * speed_b
            high_hz_b = high_hz * speed_b
            supp_idcs = torch.logical_and(freqs >= low_hz_b, freqs <= high_hz_b)
            uniform = torch.zeros_like(freqs)
            uniform[supp_idcs] = 1 / torch.sum(supp_idcs)
            expected = expected + uniform.to(device)
        lowest_hz = low_hz*torch.min(speed)
        highest_hz = high_hz*torch.max(speed)
        bpassed_freqs, psd = ideal_bandpass(freqs, psd, lowest_hz, highest_hz)
        bpassed_freqs, expected = ideal_bandpass(freqs, expected[None,:], lowest_hz, highest_hz)
        expected = expected[0] / torch.sum(expected[0]) #normalize expected psd
        psd = normalize_psd(psd) # treat all samples equally
        psd = torch.sum(psd, dim=0) / B # normalize batch psd
        emd_loss = torch.mean(torch.square(torch.cumsum(psd, dim=0) - torch.cumsum(expected, dim=0)))
    return emd_loss


def _SNR_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, freq_delta=BP_DELTA, normalized=False, bandpassed=False, device=None):
    ''' We treat this as a dynamic IPR dependent on the maximum predicted frequency.
        Arguments:
            freq_delta (float): pad for maximum frequency window we integrate over in Hertz
    '''
    if not bandpassed:
        freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz)
    signal_freq_idx = torch.argmax(psd, dim=1)
    signal_freq = freqs[signal_freq_idx].view(-1,1)
    freqs = freqs.repeat(psd.shape[0],1)
    low_cut = signal_freq - freq_delta
    high_cut = signal_freq + freq_delta
    band_idcs = torch.logical_and(freqs >= low_cut, freqs <= high_cut).to(device)
    signal_band = torch.sum(psd * band_idcs, dim=1)
    noise_band = torch.sum(psd * torch.logical_not(band_idcs), dim=1)
    denom = signal_band + noise_band + EPSILON
    snr_loss = torch.mean(noise_band / denom)
    return snr_loss


def SNR_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, freq_delta=BP_DELTA, normalized=False, bandpassed=False, device=None):
    ''' We treat this as a dynamic IPR dependent on the maximum predicted frequency.
        Arguments:
            freq_delta (float): pad for maximum frequency window we integrate over in Hertz
    '''
    if speed is None:
        snr_loss = _SNR_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, freq_delta=freq_delta, normalized=normalized, bandpassed=bandpassed, device=device)
    else:
        batch_size = psd.shape[0]
        snr_losses = torch.ones((batch_size,1)).to(device)
        for b in range(batch_size):
            low_hz_b = low_hz * speed[b]
            high_hz_b = high_hz * speed[b]
            snr_losses[b] = _SNR_SSL(freqs, psd[b].view(1,-1), low_hz=low_hz_b, high_hz=high_hz_b, freq_delta=freq_delta, normalized=normalized, bandpassed=bandpassed, device=device)
        snr_loss = torch.mean(snr_losses)
    return snr_loss


def _SNR_harmonic_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, freq_delta=BP_DELTA, normalized=False, bandpassed=False, device=None):
    ''' This sparsity loss incorporates the power in the second harmonic.
        We treat this as a dynamic IPR dependent on the maximum predicted frequency.
        Arguments:
            freq_delta (float): pad for maximum frequency window we integrate over in Hertz
    '''
    if not bandpassed:
        freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz)
    signal_freq_idx = torch.argmax(psd, dim=1)
    signal_freq = freqs[signal_freq_idx].view(-1,1)
    freqs = freqs.repeat(psd.shape[0],1)
    # First harmonic
    low_cut1 = signal_freq - freq_delta
    high_cut1 = signal_freq + freq_delta
    band_idcs = torch.logical_and(freqs >= low_cut1, freqs <= high_cut1).to(device)
    signal_band = torch.sum(psd * band_idcs, dim=1)
    # Second harmonic
    low_cut2 = 2*signal_freq - freq_delta
    high_cut2 = 2*signal_freq + freq_delta
    harm_idcs = torch.logical_and(freqs >= low_cut2, freqs <= high_cut2).to(device)
    harm_band = torch.sum(psd * harm_idcs, dim=1)
    total_power = torch.sum(psd, dim=1)
    numer = total_power - (signal_band + harm_band)
    denom = total_power + EPSILON
    snr_harm_loss = torch.mean(numer / denom)
    return snr_harm_loss


def SNR_harmonic_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, freq_delta=BP_DELTA, normalized=False, bandpassed=False, device=None):
    ''' This sparsity loss incorporates the power in the second harmonic.
        We treat this as a dynamic IPR dependent on the maximum predicted frequency.
        Arguments:
            freq_delta (float): pad for maximum frequency window we integrate over in Hertz
    '''
    if speed is None:
        snr_loss = _SNR_harmonic_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, freq_delta=freq_delta, normalized=normalized, bandpassed=bandpassed, device=device)
    else:
        batch_size = psd.shape[0]
        snr_losses = torch.ones((batch_size,1)).to(device)
        for b in range(batch_size):
            low_hz_b = low_hz * speed[b]
            high_hz_b = high_hz * speed[b]
            snr_losses[b] = _SNR_harmonic_SSL(freqs, psd[b].view(1,-1), low_hz=low_hz_b, high_hz=high_hz_b, freq_delta=freq_delta, normalized=normalized, bandpassed=bandpassed, device=device)
        snr_loss = torch.mean(snr_losses)
    return snr_loss


def NP_SUPERVISED(x, y, spectral1=None, spectral2=None):
    ''' Same as negative pearson loss, but the result is between 0 and 1.
    '''
    if len(x.shape) < 2:
        x = torch.reshape(x, (1,-1))
    mean_x = torch.mean(x, 1)
    mean_y = torch.mean(y, 1)
    xm = x.sub(mean_x[:, None])
    ym = y.sub(mean_y[:, None])
    r_num = torch.einsum('ij,ij->i', xm, ym)
    r_den = torch.norm(xm, 2, dim=1) * torch.norm(ym, 2, dim=1) + EPSILON
    r_vals = r_num / r_den
    r_val = torch.mean(r_vals)
    return (1 - r_val)/2


def ideal_bandpass(freqs, psd, low_hz, high_hz):
    freq_idcs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    freqs = freqs[freq_idcs]
    psd = psd[:,freq_idcs]
    return freqs, psd


def normalize_psd(psd):
    return psd / torch.sum(psd, keepdim=True, dim=1) ## treat as probabilities


def torch_power_spectral_density(x, nfft=5400, fps=90, low_hz=BP_LOW, high_hz=BP_HIGH, return_angle=False, radians=True, normalize=True, bandpass=True):
    centered = x - torch.mean(x, keepdim=True, dim=1)
    rfft_out = fft.rfft(centered, n=nfft, dim=1)
    psd = torch.abs(rfft_out)**2
    N = psd.shape[1]
    freqs = fft.rfftfreq(2*N-1, 1/fps)
    if return_angle:
        angle = torch.angle(rfft_out)
        if not radians:
            angle = torch.rad2deg(angle)
        if bandpass:
            freqs, psd, angle = ideal_bandpass(freqs, psd, low_hz, high_hz, angle=angle)
        if normalize:
            psd = normalize_psd(psd)
        return freqs, psd, angle
    else:
        if bandpass:
            freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz)
        if normalize:
            psd = normalize_psd(psd)
        return freqs, psd

