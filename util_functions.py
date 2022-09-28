import numpy as np
from scipy import signal
from scipy.fft import fft
import librosa as rosa


def construct_filters_iir(fs):
    hh_filt_order = 2
    hh_filt_freq = 220

    b_hh, a_hh = signal.butter(hh_filt_order, hh_filt_freq / (fs / 2), btype='highpass', output='ba')
    ic_hh = np.zeros((hh_filt_order,), dtype=float)
    filt_params = {'a_hh': a_hh,
                   'b_hh': b_hh,
                   'ic_hh': ic_hh}

    return filt_params


def initialize_ftr_mtx(fft_length):
    m = int(fft_length / 8)
    bin_idxs = np.arange(1, m)
    ftr_size = len(bin_idxs)
    ftr_mtx = np.zeros((0, ftr_size))

    return ftr_mtx, bin_idxs


def detect_onsets(s, detect_params, filt_params):
    frame_size = detect_params['frame_size']
    num_samples = np.shape(s)[0]
    sample_idxs = np.arange(3 * frame_size, num_samples, frame_size, dtype=int)
    if sample_idxs[-1] + frame_size >= num_samples:
        sample_idxs = np.delete(sample_idxs, len(sample_idxs) - 1)
    num_idxs = len(sample_idxs)
    onset = {'sequence': np.zeros((num_idxs,)),
             'idxs': []
             }

    frame_enrg = np.zeros((num_idxs + 1,))
    frame_enrg_smooth = np.zeros((num_idxs + 1,))
    active_idxs = np.zeros((num_idxs,), dtype=int)
    activity = np.zeros((num_idxs,))

    num_idx = 1
    for ni in sample_idxs:
        frame_in = s[ni:ni + frame_size]  # *hanwin1
        s_hh, ic = signal.lfilter(filt_params['b_hh'], filt_params['a_hh'], frame_in, zi=filt_params['ic_hh'])
        filt_params['ic_hh'] = ic
        frame_enrg[num_idx] = np.linalg.norm(s_hh) / np.sqrt(frame_size)
        frame_enrg_smooth[num_idx] = 0.8 * frame_enrg_smooth[num_idx - 1] + 0.2 * frame_enrg[num_idx]
        if frame_enrg[num_idx] / frame_enrg_smooth[num_idx] == 0:
            # print("aaaaaaaaaaa")
            frame_enrg_ratio = 20 * np.log10(10**-10)
        else:
            frame_enrg_ratio = 20 * np.log10(frame_enrg[num_idx] / frame_enrg_smooth[num_idx])
        if frame_enrg[num_idx] > detect_params['power_thresh']:  # and frame_enrg_ratio>detectParams['dBrat']: #activationEnergy>frame_enrg_smooth:
            activity[num_idx - 1] = frame_enrg_ratio + 0 * np.log10(frame_enrg[num_idx] / detect_params['power_thresh'])
            active_idxs[num_idx - 1] = 1
        else:
            activity[num_idx - 1] = 0
            active_idxs[num_idx - 1] = 0
        num_idx += 1

    return onset, activity, frame_enrg, frame_enrg_smooth, sample_idxs, active_idxs


def extract_features(s, detect_params, filt_params, fft_length, stamps):


    onset, activity, frame_enrg, frame_enrg_smooth, sample_idxs, active_idxs = detect_onsets(s, detect_params,
                                                                                             filt_params)

    #  for spectral features
    # ftr_mtx = np.zeros((0, 425))

    # for energy feaures
    ftr_mtx, bin_idxs = initialize_ftr_mtx(fft_length)
    ftr_size = len(bin_idxs)

    a_idxs = np.argwhere(activity >= 6)
    for a in a_idxs:
        n = a[0]
        # if stamps[0] < n < stamps[3] and np.sum(active_idxs[n - detectParams['lookBackFrames']:n]) == 0:  # 3
        if np.sum(active_idxs[n - detect_params['look_back_frames']:n]) == 0:  # 3
            onset['sequence'][n] = 1
            onset['idxs'].append(n)

            s_in = s[sample_idxs[n]:sample_idxs[n] + fft_length] * np.hanning(fft_length)

            # # for spectral features
            # # spectrogram
            # spectr = np.abs(rosa.stft(s_in, n_fft=fft_length, hop_length=fft_length//4))
            #
            # # frequencies corresponding to every frequency bin
            # freq_bins = np.arange(0, 1 + fft_length / 2) * 48000 / fft_length
            #
            # # bins with desired frequencies
            # desired_freq_bins = np.argwhere(freq_bins <= 4000)
            #
            # # keep only that part of spectrogram
            # # spectr = spectr[0:desired_freq_bins[-1][0], :]
            # spectr = spectr[0:desired_freq_bins[-1][0], :]
            # ftr_mtx = np.append(ftr_mtx, spectr.flatten(order="F").reshape(1, -1), axis=0)


            # for original energy features

            s_in_fft = fft(s_in)

            s_in_enrg = np.abs(s_in_fft[bin_idxs]) ** 2
            s_in_enrg = np.sqrt(s_in_enrg / np.sum(s_in_enrg))
            ftr_mtx = np.append(ftr_mtx, np.reshape(s_in_enrg, (1, ftr_size)), axis=0)
    # ftr_mtx =

    return ftr_mtx, onset, activity, frame_enrg, frame_enrg_smooth
