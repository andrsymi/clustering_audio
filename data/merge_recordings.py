import glob
import os
import numpy as np
from scipy.io import wavfile
import pandas as pd


def main():
    print(os.getcwd())
    sr = 48000
    silence = np.zeros(sr)

    classes = [silence, silence, silence]

    for filename in glob.glob('*.wav'):
        file = filename.split('.wav')[0]
        if file == "merged":
            continue
        print(file)

        sr, wave = wavfile.read("{}.wav".format(file))
        if np.max(wave) > 10:
            wave = wave / 32768.0

        time_stamps = pd.read_csv("{}.csv".format(file), header=None)
        sample_stamps = (time_stamps * sr).values[:, 0].astype(int)

        for i in range(len(sample_stamps) - 1):
            classes[i] = np.concatenate([classes[i], wave[sample_stamps[i]:sample_stamps[i + 1]]])
            classes[i] = np.concatenate([classes[i], silence])

    result_file = np.array([])
    for i in range(len(classes)):
        classes[i] = np.concatenate([classes[i], silence])
        result_file = np.concatenate([result_file, classes[i]])

    wavfile.write("merged.wav", sr, result_file)

    a = 0


if __name__ == "__main__":
    main()
