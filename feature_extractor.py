# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:58:31 2019
extracts the features required for training
assumes file is in the same folder as modelTrain
@author: nstefana
"""
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


# %%
def construct_filters_IIR(Fs, detectParams):
    frs = detectParams['frs']

    hhfiltOrder = 2
    hhfiltFreq = 220

    Bhh, Ahh = signal.butter(hhfiltOrder, hhfiltFreq / (Fs / 2), btype='highpass')
    IChh = np.zeros((hhfiltOrder,), dtype=float)
    filtParams = {'Ahh': Ahh,
                  'Bhh': Bhh,
                  'IChh': IChh}

    return filtParams


# %%
def initialize_ftrMtx():
    fftLength = 1024
    M = int(fftLength / 8)
    binIdxs = np.arange(1, M)
    ftrSize = len(binIdxs)
    ftrMtx = np.zeros((0, ftrSize))
    return ftrMtx, fftLength, binIdxs


# %%
def extract_features(s, Fs, detectParams, filtParams, stamps):
    frs = detectParams['frs']
    Ns = np.shape(s)[0]
    sampleIdxs = np.arange(3 * frs, Ns, frs, dtype=int)
    if sampleIdxs[-1] + frs >= Ns:
        sampleIdxs = np.delete(sampleIdxs, len(sampleIdxs) - 1)
    N = len(sampleIdxs)
    onset = {'sequence': np.zeros((N,)),
             'idxs': [],
             'true_label': []
             }

    ftrMtx, fftLength, binIdxs = initialize_ftrMtx()
    ftrSize = len(binIdxs)
    frameEn = np.zeros((N + 1,))
    frameEnSmooth = np.zeros((N + 1,))
    activeIdxs = np.zeros((N,), dtype=int)
    activity = np.zeros((N,))
    true_labels = np.empty_like(activity, dtype=np.str)

    nIdx = 1
    for ni in sampleIdxs:
        frameIN = s[ni:ni + frs]  # *hanwin1
        sHH, ic = signal.lfilter(filtParams['Bhh'], filtParams['Ahh'], frameIN, zi=filtParams['IChh'])
        filtParams['IChh'] = ic
        frameEn[nIdx] = np.linalg.norm(sHH) / np.sqrt(frs)
        frameEnSmooth[nIdx] = 0.8 * frameEnSmooth[nIdx - 1] + 0.2 * frameEn[nIdx]
        frameEnRatio = 20 * np.log10(frameEn[nIdx] / frameEnSmooth[nIdx])
        if frameEn[nIdx] > detectParams['powerThresh']:  # and frameEnRatio>detectParams['dBrat']: #activationEnergy>frameEnSmooth:
            activity[nIdx - 1] = frameEnRatio + 0 * np.log10(frameEn[nIdx] / detectParams['powerThresh'])
            activeIdxs[nIdx - 1] = 1

        else:
            activity[nIdx - 1] = 0
            activeIdxs[nIdx - 1] = 0
        nIdx += 1

        # da = activeIdxs[1:]-activeIdxs[:-1]
    aIdxs = np.argwhere(activity >= 6)
    for a in aIdxs:
        n = a[0]
        # if stamps[0] < n < stamps[3] and np.sum(activeIdxs[n - detectParams['lookBackFrames']:n]) == 0:  # 3
        if np.sum(activeIdxs[n - detectParams['lookBackFrames']:n]) == 0:  # 3
            # if  activity[n:n+detectParams['lookAheadFrames']]>=8:
            onset['sequence'][n] = 1
            onset['idxs'].append(n)
            onset['true_label'].append(true_labels[n])

            sIN = s[sampleIdxs[n]:sampleIdxs[n] + fftLength] * np.hanning(fftLength)
            Sin = fft(sIN)
            # if withoutNormalization
            # ftrIN = np.abs(Sin[binIdxs])
            # if withNormalization
            ESin = np.abs(Sin[binIdxs]) ** 2
            ftrIN = np.sqrt(ESin / np.sum(ESin))
            ftrMtx = np.append(ftrMtx, np.reshape(ftrIN, (1, ftrSize)), axis=0)

    return ftrMtx, onset, activity, frameEn, frameEnSmooth


def check_ftrMtx(ftrMtx, labels):
    similarityThreshold = 0.8
    c0 = np.argwhere(labels == 0)
    c1 = np.argwhere(labels == 1)
    c2 = np.argwhere(labels == 2)
    M0 = np.dot(ftrMtx[c0.flatten(), :], np.transpose(ftrMtx[c0.flatten(), :]))
    M1 = np.dot(ftrMtx[c1.flatten(), :], np.transpose(ftrMtx[c1.flatten(), :]))
    M2 = np.dot(ftrMtx[c2.flatten(), :], np.transpose(ftrMtx[c2.flatten(), :]))

    # fig, axs = plt.subplots(3,1)    
    # axs[0].pcolor(M0)
    # axs[1].pcolor(M1)
    # axs[2].pcolor(M2)    

    match0 = (np.sum(M0, axis=0) - 1) / (len(c0))
    match1 = (np.sum(M1, axis=0) - 1) / (len(c1))
    match2 = (np.sum(M2, axis=0) - 1) / (len(c2))

    # fig, axs = plt.subplots()    
    # axs.plot(match0)
    # axs.plot(match1)
    # axs.plot(match2)

    delIdxs0 = np.argwhere(match0 < similarityThreshold)
    delIdxs1 = np.argwhere(match1 < similarityThreshold)
    delIdxs2 = np.argwhere(match2 < similarityThreshold)

    deleteIdxs = np.concatenate((delIdxs0, delIdxs1, delIdxs2))
    Nsamples = len(labels)
    ftrSize = np.shape(ftrMtx)[1]
    finalFtrMtx = np.zeros((0, ftrSize))
    finalLabels = np.zeros((0,))
    for i in range(Nsamples):
        if i not in deleteIdxs:
            finalFtrMtx = np.append(finalFtrMtx, np.reshape(ftrMtx[i, :], (1, ftrSize)), axis=0)
            finalLabels = np.append(finalLabels, labels[i])
    print('Deleting ' + str(len(deleteIdxs)) + ' outliers')
    return finalFtrMtx, finalLabels
