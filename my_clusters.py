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
from sklearn.metrics import confusion_matrix, accuracy_score


def main(filename, detect_params, fs):
    sr, wave = wavfile.read("{}.wav".format(filename))

    time_stamps = pd.read_csv("{}.csv".format(filename), header=None)
    sample_stamps = (time_stamps * sr).values[:, 0]

    feature_matrix, fft_length, bin_idx = initialize_ftrMtx()
    filter_params = construct_filters_IIR(fs, detect_params)

    # why???
    # if np.max(wave) > 10:
    #     wave = wave / 32768.0

    ftrsOUT, onset, activeIdxs, frameEn, frameEnSmooth = extract_features(wave, fs, detect_params, filter_params, sample_stamps)

    labels2plot = np.zeros((len(activeIdxs),))

    km = KMeans(
        n_clusters=3, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    labels = km.fit_predict(ftrsOUT)

    Nsamples = np.shape(np.log(ftrsOUT))[0]
    for s in range(Nsamples):
        nIdx = onset['idxs'][s]
        labels2plot[nIdx] = labels[s] + 1

        # true_labels[s] = onset['true_label'][s]

    plt.plot(wave)
    for stamp in sample_stamps:
        plt.axvline(x=stamp, c='red')

    fig, axs = plt.subplots()
    axs.stem(labelsPlotKmeans)
    axs.plot(0.1 * activation, 'red')
    axs.plot(20 * frameEn, 'green')

    fig, axs = plt.subplots()
    axs.stem(labelsPlotSpectral)
    axs.plot(0.1 * activation, 'red')
    axs.plot(20 * frameEn, 'green')

    plt.show()

    plt.show()


if __name__ == "__main__":
    fs_run = 48000
    frs = 64

    detect_parameters = {'frs': frs,
                         'powerThresh': 0.003,  # changed to refer to RMS
                         'dBrat': 3,
                         'lookBackFrames': 25,
                         'lookAheadFrames': 1}

    os.chdir(os.getcwd() + '/data')
    for file in glob.glob('*.wav'):
        print(file)
        if file == "input1.wav":
            continue
        main(filename=file.split('.wav')[0], detect_params=detect_parameters, fs=fs_run)

    a = 1
