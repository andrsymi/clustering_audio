import util_functions as uf
import glob
import os
import numpy as np
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, silhouette_score, calinski_harabasz_score
from sklearn.metrics.cluster import adjusted_rand_score, fowlkes_mallows_score
from scipy.spatial import KDTree
from tabulate import tabulate
from prettytable import PrettyTable


def main_function(filename, detectParams, Fs, frs, fft_length):
    filt_params = uf.construct_filters_iir(Fs)

    file = filename.split('.wav')[0]

    sr, wave = wavfile.read("{}.wav".format(file))
    if np.max(wave) > 10:
        wave = wave / 32768.0

    time_stamps = pd.read_csv("{}.csv".format(file), header=None)
    sample_stamps = (time_stamps * sr).values[:, 0].astype(int)

    # class_parts = []
    # labels_string = "ABCDEFG"
    # class_labels = []
    result = []
    for i in range(len(sample_stamps) - 1):
        # class_parts.append(wave[sample_stamps[i]:sample_stamps[i+1]])
        # class_labels.append(labels_string[i])
        result.append(
            uf.extract_features(wave[sample_stamps[i]:sample_stamps[i + 1]], detectParams, filt_params, fft_length,
                                [7.5, 18.4, 39.7, 56.5]))

    # ftrsOUT, onset, activeIdxs, frameEn, frameEnSmooth = uf.extract_features(wave, detectParams, filt_params, fft_length, [7.5, 18.4, 39.7, 56.5])

    # result = [uf.extract_features(wave[sample_stamps[i]:sample_stamps[i+1]], detectParams, filt_params, fft_length, [7.5, 18.4, 39.7, 56.5]) for i in range(len(sample_stamps) - 1)]
    ftrsOUT, onset, activeIdxs, frameEn, frameEnSmooth = zip(*result)

    data = np.concatenate((ftrsOUT[0], ftrsOUT[1], ftrsOUT[2]))

    # true_labes = np.empty([data.shape[0], 1], dtype=str)
    # true_labes[:ftrsOUT[0].shape[0]] = "A"
    # true_labes[ftrsOUT[0].shape[0]:ftrsOUT[0].shape[0] + ftrsOUT[1].shape[0]] = "B"
    # true_labes[ftrsOUT[0].shape[0] + ftrsOUT[1].shape[0]:] = "C"

    true_labes = np.empty([data.shape[0], 1])
    true_labes[:ftrsOUT[0].shape[0]] = 0
    true_labes[ftrsOUT[0].shape[0]:ftrsOUT[0].shape[0] + ftrsOUT[1].shape[0]] = 1
    true_labes[ftrsOUT[0].shape[0] + ftrsOUT[1].shape[0]:] = 2

    models = []
    model_names = []
    labels = []
    confusion_matrices = []
    # ari = []
    # fms = []
    # s_score = []
    # ccs = []
    metrics = [[], [], [], []]
    metric_names = ["Adjusted Rand Index", "Fowlkes Mallows Score", "Silhouette Score", "Calinski Harabasz Score"]

    model_names.append("KMeans")
    models.append(KMeans(n_clusters=3,
                         init='random',
                         n_init=10,
                         max_iter=500,
                         tol=1e-04,
                         random_state=0))

    model_names.append("Spectral Clustering")
    models.append(SpectralClustering(n_clusters=3,
                                     assign_labels='discretize',
                                     random_state=0))

    model_names.append("HAC (Ward Linkage)")
    models.append(AgglomerativeClustering(n_clusters=3,
                                          linkage="ward"))

    model_names.append("HAC (Complete Linkage)")
    models.append(AgglomerativeClustering(n_clusters=3,
                                          linkage="complete"))

    model_names.append("HAC (Average Linkage)")
    models.append(AgglomerativeClustering(n_clusters=3,
                                          linkage="average"))

    model_names.append("HAC (Single Linkage)")
    models.append(AgglomerativeClustering(n_clusters=3,
                                          linkage="single"))

    model_names.append("Gaussian Mixture Model")
    models.append(GaussianMixture(n_components=3,
                                  tol=1e-04,
                                  max_iter=500,
                                  random_state=0))

    model_names.append("Bayesian Gaussian Mixture")
    models.append(BayesianGaussianMixture(n_components=3,
                                          tol=1e-04,
                                          max_iter=500,
                                          random_state=0))

    # model_names.append("Mean Shift")
    # models.append(MeanShift(max_iter=500))

    fig, axs = plt.subplots(3, int(np.ceil(len(models) / 3)), sharex="all", sharey="all")  # , figsize=(12, 6))
    ax = axs.flat
    fig.suptitle("{}.wav".format(file), fontsize=16)
    fig.tight_layout(h_pad=2)

    for i, model in enumerate(models):
        labels.append(model.fit_predict(data))
        confusion_matrices.append(confusion_matrix(true_labes, labels[i]))

        # ari.append(adjusted_rand_score(true_labes, labels[i]))
        metrics[0].append(adjusted_rand_score(true_labes.flatten(), labels[i].flatten()))
        # fms.append(fowlkes_mallows_score(true_labes, labels[i]))
        metrics[1].append(fowlkes_mallows_score(true_labes.flatten(), labels[i].flatten()))
        # s_score.append(silhouette_score(data, true_labes))
        # metrics[2].append(silhouette_score(data, true_labes.flatten()))
        # ccs.append(calinski_harabasz_score(data, true_labes))
        # metrics[3].append(calinski_harabasz_score(data, true_labes.flatten()))

        ax[i].set_title(model_names[i])
        sns.heatmap(confusion_matrices[i], annot=True, fmt="d", cmap="Blues", ax=ax[i])

    fig1, axs1 = plt.subplots(2, 1)  # , sharex="all", sharey="all")
    ax1 = axs1.flat
    fig1.suptitle("Metrics", fontsize=16)
    fig1.tight_layout(h_pad=2)
    fig1.subplots_adjust(left=0.321, right=0.96, top=0.884, bottom=0.08)

    for i, ax in enumerate(ax1):
        ax.set_title(metric_names[i])
        #     # ax.barh(y=model_names[i], height=metrics[i], width=0.8)
        #     # sns.barplot(x=metrics[i], ax=ax)
        #     ax.bar(x=model_names, y=metrics[i], height=0.8)

        sns.barplot(x=metrics[i], y=model_names, ax=ax)

    best_model_idx = np.argmax(metrics[0])
    print("Best Model:", model_names[best_model_idx])
    true_labels = true_labes.flatten().astype(np.int64)
    tree = KDTree(data=data)

    #  For cluster labels in the best model
    for i in np.unique(labels[best_model_idx]):
        idxs = np.argwhere(labels[best_model_idx] == i).flatten()
        counts = np.bincount(true_labels[idxs])
        corresponding_true_label = np.argmax(counts)  # the true label that corresponds to i

        # misclassified = np.argwhere(true_labels == corresponding_true_label and labels[best_model_idx] != i)

        samples_in_class = np.argwhere(true_labels == corresponding_true_label).flatten()  # samples that belong to the corresponding true class
        samples_not_classified_in_class = np.argwhere(labels[best_model_idx] != i).flatten()  # samples that where not clustered in ith cluster
        misclassified_samples = np.intersect1d(samples_in_class, samples_not_classified_in_class)

        # do nearest neighbour thing
        _, ii = tree.query(x=data[misclassified_samples], k=6)

        print("\nCluster label:", i, "Corresponding true label:", corresponding_true_label)
        print("Number of misclassified samples:", misclassified_samples.shape[0])
        # table = []
        # head = ["Sample idx", "True Label of Neighbours"]
        pretty_table = PrettyTable(["Sample idx", "Cluster label", "True Label of Neighbours", "Cluster labels of Neighbours"])
        for sample in range(misclassified_samples.shape[0]):
            # table.append([misclassified_samples[sample], true_labels[ii[sample, 1:]]])
            pretty_table.add_row([misclassified_samples[sample], labels[best_model_idx][sample], true_labels[ii[sample, 1:]], labels[best_model_idx][ii[sample, 1:]]])
            # print("Misclassified sample:", misclassified_samples[sample])
            # print(true_labels[ii[sample, 1:]])
        print(pretty_table)
        # print(tabulate(table, headers=head))#, tablefmt="grid"))


    z = 1




    # fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)  # , figsize=(12, 6))
    # ax = axs.flat
    # fig.suptitle("{}.wav".format(file), fontsize=16)
    # fig.tight_layout(h_pad=2)
    #
    # ### KMeans
    # km = KMeans(n_clusters=3,
    #             init='random',
    #             n_init=10,
    #             max_iter=300,
    #             tol=1e-04,
    #             random_state=0)
    # labels_kmeans = km.fit_predict(data)
    #
    # cm_kmeans = confusion_matrix(true_labes, labels_kmeans)
    #
    # axs[0, 0].set_title('KMeans')
    # axs[0, 0].set_xlabel('Cluster labels')
    # axs[0, 0].set_ylabel('True labels')
    # sns.heatmap(cm_kmeans, annot=True, fmt="d", cmap="Blues", ax=axs[0, 0])
    #
    # ### Spectral
    # spectral = SpectralClustering(n_clusters=3,
    #                               assign_labels='discretize',
    #                               random_state=0)
    # labels_spectral = spectral.fit_predict(data)
    #
    # cm_spectral = confusion_matrix(true_labes, labels_spectral)
    #
    # axs[0, 1].set_title('Spectral Clustering')
    # axs[0, 1].set_xlabel('Cluster labels')
    # axs[0, 1].set_ylabel('True labels')
    # sns.heatmap(cm_spectral, annot=True, fmt="d", cmap="Blues", ax=axs[0, 1])
    #
    # ### Agglomerative
    # # Ward
    # agg_ward_clust = AgglomerativeClustering(n_clusters=3,
    #                                          linkage="ward")
    # labels_agg_ward = agg_ward_clust.fit_predict(data)
    #
    # cm_agg_ward = confusion_matrix(true_labes, labels_agg_ward)
    #
    # axs[1, 0].set_title("HAC (Ward Linkage)")
    # axs[1, 0].set_xlabel('Cluster labels')
    # axs[1, 0].set_ylabel('True labels')
    # sns.heatmap(cm_agg_ward, annot=True, fmt="d", cmap="Blues", ax=axs[1, 0])
    #
    # # Complete
    # agg_complete_clust = AgglomerativeClustering(n_clusters=3,
    #                                              linkage="complete")
    # labels_complete_ward = agg_complete_clust.fit_predict(data)
    # cm_complete_ward = confusion_matrix(true_labes, labels_complete_ward)
    #
    # axs[1, 1].set_title("HAC (Complete Linkage)")
    # axs[1, 1].set_xlabel('Cluster labels')
    # axs[1, 1].set_ylabel('True labels')
    # sns.heatmap(cm_complete_ward, annot=True, fmt="d", cmap="Blues", ax=axs[1, 1])
    #
    # # Average
    # agg_average_clust = AgglomerativeClustering(n_clusters=3,
    #                                             linkage="average")
    # labels_average_ward = agg_average_clust.fit_predict(data)
    # cm_average_ward = confusion_matrix(true_labes, labels_average_ward)
    #
    # axs[2, 0].set_title("HAC (Complete Linkage)")
    # axs[2, 0].set_xlabel('Cluster labels')
    # axs[2, 0].set_ylabel('True labels')
    # sns.heatmap(cm_average_ward, annot=True, fmt="d", cmap="Blues", ax=axs[2, 0])
    #
    # # Single
    # agg_single_clust = AgglomerativeClustering(n_clusters=3,
    #                                            linkage="single")
    # labels_single_ward = agg_single_clust.fit_predict(data)
    # cm_single_ward = confusion_matrix(true_labes, labels_single_ward)
    #
    # axs[2, 1].set_title("HAC (Single Linkage)")
    # axs[2, 1].set_xlabel('Cluster labels')
    # axs[2, 1].set_ylabel('True labels')
    # sns.heatmap(cm_single_ward, annot=True, fmt="d", cmap="Blues", ax=axs[2, 1])

    # ax.xaxis.set_ticklabels(['business', 'health'])
    # ax.yaxis.set_ticklabels(["A", "B", "C"])

    plt.show()

    labels2plotKmeans = np.zeros((len(activeIdxs),))
    labels2plotSpectral = np.zeros((len(activeIdxs),))  ###
    #
    # # axs[cIdx].plot(frameEn[1:])
    # # axs[cIdx].plot(frameEnSmooth[1:])
    # # axs[cIdx].plot(0.001*activeIdxs)
    # # axs[cIdx].stem(0.03*onsetIdxs)
    #
    # km = KMeans(
    #     n_clusters=3, init='random',
    #     n_init=10, max_iter=300,
    #     tol=1e-04, random_state=0
    # )
    # labels = km.fit_predict(ftrsOUT)
    #
    #
    #
    # Nsamples = np.shape(np.log(ftrsOUT))[0]
    # for s in range(Nsamples):
    #     nIdx = onset['idxs'][s]
    #     labels2plotKmeans[nIdx] = labels[s] + 1
    #
    #     # true_labels[s] = onset['true_label'][s]
    # #     edw tab
    #
    # # %%
    # from sklearn.cluster import SpectralClustering
    # spc = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0).fit(ftrsOUT)
    # labels = spc.labels_
    #
    # for s in range(Nsamples):
    #     nIdx = onset['idxs'][s]
    #     labels2plotSpectral[nIdx] = labels[s] + 1

    return labels2plotKmeans, labels2plotSpectral, activeIdxs, frameEn


if __name__ == "__main__":
    ###
    fft_length = 1024
    ###
    FsRun = 48000
    frs = 64

    detectParams = {'frame_size': frs,
                    'power_thresh': 0.003,  # changed to refer to RMS
                    'dBrat': 3,
                    'look_back_frames': 25,
                    'look_ahead_frames': 1}
    # 0.1 is good for bo
    # input.wav is in same folder

    os.chdir(os.getcwd() + '/data')
    print(os.getcwd())

    for file in glob.glob('*.wav'):
        print("\n===================")
        print('\033[1m' + file + '\033[0m')
        labelsPlotKmeans, labelsPlotSpectral, activation, frameEn = main_function(file, detectParams, FsRun, frs,
                                                                                  fft_length)
        # fig, axs = plt.subplots()
        # axs.stem(labelsPlotKmeans)
        # axs.plot(0.1 * activation, 'red')
        # axs.plot(20 * frameEn, 'green')
        #
        # fig, axs = plt.subplots()
        # axs.stem(labelsPlotSpectral)
        # axs.plot(0.1 * activation, 'red')
        # axs.plot(20 * frameEn, 'green')
        #
        # plt.show()
