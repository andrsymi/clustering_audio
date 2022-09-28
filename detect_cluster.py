# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:38:50 2019
offline classifier that may correct mistakes
made by real time classifier

"""
import glob
import os
import numpy as np
from scipy.io import wavfile
import time
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt
# from modelTrain import modelTrain
from feature_extractor import extract_features
from feature_extractor import construct_filters_IIR
from feature_extractor import initialize_ftrMtx
from feature_extractor import check_ftrMtx
from sklearn.cluster import KMeans


def main_function(filename, detectParams, Fs, frs):
    labels = np.zeros((0,))
    ftrMtx, fftLength, binIdxs = initialize_ftrMtx()
    ftrSize = np.shape(ftrMtx)[1]
    filtParams = construct_filters_IIR(Fs, detectParams)
    # model = tensorflow.keras.models.load_model('Model')
    # %%
    # os.chdir(os.getcwd() + '/data')
    # print(os.getcwd())
    # for filename in glob.glob('*.wav'):
    split1 = filename.split('.wav')
    wavName = split1[0]
    # wavName=temp.split(os.sep)[-1]

    sr, sig0UT = wavfile.read(wavName + '.wav')
    if np.max(sig0UT) > 10:
        sigOUT = sig0UT / 32768.0
    ftrsOUT, onset, activeIdxs, frameEn, frameEnSmooth = extract_features(sigOUT, Fs, detectParams, filtParams, [7.5, 18.4, 39.7, 56.5])
    labels2plotKmeans = np.zeros((len(activeIdxs),))
    labels2plotSpectral = np.zeros((len(activeIdxs),))###

    # axs[cIdx].plot(frameEn[1:])
    # axs[cIdx].plot(frameEnSmooth[1:])
    # axs[cIdx].plot(0.001*activeIdxs)
    # axs[cIdx].stem(0.03*onsetIdxs)

    km = KMeans(
        n_clusters=3, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    labels = km.fit_predict(ftrsOUT)



    Nsamples = np.shape(np.log(ftrsOUT))[0]
    for s in range(Nsamples):
        nIdx = onset['idxs'][s]
        labels2plotKmeans[nIdx] = labels[s] + 1

        # true_labels[s] = onset['true_label'][s]
    #     edw tab

    # %%
    from sklearn.cluster import SpectralClustering
    spc = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0).fit(ftrsOUT)
    labels = spc.labels_

    for s in range(Nsamples):
        nIdx = onset['idxs'][s]
        labels2plotSpectral[nIdx] = labels[s] + 1

    return labels2plotKmeans, labels2plotSpectral, activeIdxs, frameEn


# %%
if __name__ == "__main__":
    FsRun = 48000
    frs = 64

    detectParams = {'frs': frs,
                    'powerThresh': 0.003,  # changed to refer to RMS
                    'dBrat': 3,
                    'lookBackFrames': 25,
                    'lookAheadFrames': 1}
    # 0.1 is good for bo
    # input.wav is in same folder

    os.chdir(os.getcwd() + '/data')
    print(os.getcwd())
    for file in glob.glob('*.wav'):
        print(file)
        labelsPlotKmeans, labelsPlotSpectral, activation, frameEn = main_function(file, detectParams, FsRun, frs)
        fig, axs = plt.subplots()
        axs.stem(labelsPlotKmeans)
        axs.plot(0.1 * activation, 'red')
        axs.plot(20 * frameEn, 'green')

        fig, axs = plt.subplots()
        axs.stem(labelsPlotSpectral)
        axs.plot(0.1 * activation, 'red')
        axs.plot(20 * frameEn, 'green')

        plt.show()

print("finished recording")
