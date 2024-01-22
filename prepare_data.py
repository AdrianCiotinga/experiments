import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from KDEpy import NaiveKDE


def prepare_noisy_dataset(dataset, filePath, protected_attr, normalized=True):
    dataAX = pd.read_csv(
        filePath + dataset + "_binerized.csv"
    )

    dataX = dataAX.drop(protected_attr, axis=1)
    dataA = dataAX[protected_attr]
    dataY = None

    # Need labels to code this out

    #dataY = pd.read_csv(
    #    filePath + "IBM_compas_Y.csv", sep="\t", index_col=0, header=None
    #)

    if normalized:
        dataX = normalize(dataX)
    return dataA, dataY, dataX

def prepare_compas(normalized=True):

    filePath = "datasets/IBM_compas/"
    dataA = pd.read_csv(
        filePath + "IBM_compas_A.csv", sep="\t", index_col=0, header=None
    )
    dataY = pd.read_csv(
        filePath + "IBM_compas_Y.csv", sep="\t", index_col=0, header=None
    )
    dataX = pd.read_csv(filePath + "IBM_compas_X.csv", sep="\t", index_col=0)
    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:, 0], dataY.iloc[:, 0], dataX


def prepare_german(normalized=True):
    filePath = "datasets/A,Y,X/IBM_german/"

    dataA = pd.read_csv(
        filePath + "IBM_german_A.csv", sep="\t", index_col=0, header=None
    )
    dataY = pd.read_csv(
        filePath + "IBM_german_Y.csv", sep="\t", index_col=0, header=None
    )
    dataX = pd.read_csv(filePath + "IBM_german_X.csv", sep="\t", index_col=0)

    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:, 0], dataY.iloc[:, 0], dataX


def prepare_drug(normalized=True):
    filePath = "datasets/drug/"

    dataA = pd.read_csv(filePath + "drug_A.csv", sep="\t", index_col=0, header=None)
    dataY = pd.read_csv(filePath + "drug_Y.csv", sep="\t", index_col=0, header=None)
    dataX = pd.read_csv(filePath + "drug_X.csv", sep="\t", index_col=0)

    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:, 0], dataY.iloc[:, 0], dataX


def prepare_arrhythmia(normalized=True):
    filePath = "datasets/arrhythmia/"

    dataA = pd.read_csv(
        filePath + "arrhythmia_A.csv", sep="\t", index_col=0, header=None
    )
    dataY = pd.read_csv(
        filePath + "arrhythmia_Y.csv", sep="\t", index_col=0, header=None
    )
    dataX = pd.read_csv(filePath + "arrhythmia_X.csv", sep="\t", index_col=0)

    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:, 0], dataY.iloc[:, 0], dataX


def normalize(X):
    for c in list(X.columns):
        if X[c].min() < 0 or X[c].max() > 1:
            mu = X[c].mean()
            s = X[c].std(ddof=0)
            X.loc[:, c] = (X[c] - mu) / s
    return X



def prepare_dataset(dataA, dataY, dataX, alpha, beta, kdebw, epsilon, sample_size_ratio):
    data = pd.concat([dataA, dataX], axis=1).values  # include A in features
    tr_idx, ts_idx, ratios = create_shift(
        data,
        src_split=sample_size_ratio,
        alpha=alpha,
        beta=beta,
        kdebw=kdebw,
        eps=epsilon,
    )
    tr_X, tr_ratio = dataX.iloc[tr_idx, :], ratios[tr_idx]
    ts_X, ts_ratio = dataX.iloc[ts_idx, :], ratios[ts_idx]
    tr_A, tr_Y = dataA.iloc[tr_idx].squeeze(), dataY.iloc[tr_idx].squeeze()
    ts_A, ts_Y = dataA.iloc[ts_idx].squeeze(), dataY.iloc[ts_idx].squeeze()

    dataset = dict(
        X_src=tr_X.values,
        A_src=tr_A.values,
        Y_src=tr_Y.values,
        ratio_src=tr_ratio,
        X_trg=ts_X.values,
        A_trg=ts_A.values,
        Y_trg=ts_Y.values,
        ratio_trg=ts_ratio,
    )

    return dataset

def create_shift(
    data,
    src_split=0.4,
    alpha=1,
    beta=2,
    kdebw=0.3,
    eps=0.001,
):
    """
    Creates covariate shift sampling of data into disjoint source and target set.

    Let \mu and \sigma be the mean and the standard deviation of the first principal component retrieved by PCA on the whole data.
    The target is randomly sampled based on a Gaussian with mean = \mu and standard deviation = \sigma.
    The source is randomly sampled based on a Gaussian with mean = \mu + alpha and standard devaition = \sigma / beta

    data: [m, n]
    alpha, beta: the parameter that distorts the gaussian used in sampling
                   according to the first principle component
    output: source indices, target indices, ratios based on kernel density estimation with bandwidth = kdebw and smoothed by eps
    """
    m = np.shape(data)[0]
    source_size = int(m * src_split)
    target_size = source_size

    pca = PCA(n_components=2)
    pc2 = pca.fit_transform(data)
    pc = pc2[:, 0]
    pc = pc.reshape(-1, 1)

    pc_mean = np.mean(pc)
    pc_std = np.std(pc)

    sample_mean = pc_mean + alpha
    sample_std = pc_std / beta

    # sample according to the probs
    prob_s = norm.pdf(pc, loc=sample_mean, scale=sample_std)
    sum_s = np.sum(prob_s)
    prob_s = prob_s / sum_s
    prob_t = norm.pdf(pc, loc=pc_mean, scale=pc_std)
    sum_t = np.sum(prob_t)
    prob_t = prob_t / sum_t

    source_ind = np.random.choice(
        range(m), size=source_size, replace=False, p=np.reshape(prob_s, (m))
    )

    pt_proxy = np.copy(prob_t)
    pt_proxy[source_ind] = 0
    pt_proxy = pt_proxy / np.sum(pt_proxy)
    target_ind = np.random.choice(
        range(m), size=target_size, replace=False, p=np.reshape(pt_proxy, (m))
    )

    assert np.all(np.sort(source_ind) != np.sort(target_ind))
    src_kde = KDEAdapter(kde=NaiveKDE(kernel="gaussian", bw=kdebw)).fit(
        pc2[source_ind, :]
    )
    trg_kde = KDEAdapter(kde=NaiveKDE(kernel="gaussian", bw=kdebw)).fit(
        pc2[target_ind, :]
    )

    ratios = src_kde.p(pc2, eps) / trg_kde.p(pc2, eps)
    print("min ratio= {:.5f}, max ratio= {:.5f}".format(np.min(ratios), np.max(ratios)))

    return source_ind, target_ind, ratios


class KDEAdapter:
    def __init__(self, kde=NaiveKDE(kernel="gaussian", bw=0.3)):
        self._kde = kde

    def fit(self, sample):
        self._kde.fit(sample)
        return self

    def pdf(self, sample):
        density = self._kde.evaluate(sample)
        return density

    def p(self, sample, eps=0):
        density = self._kde.evaluate(sample)
        return (density + eps) / np.sum(density + eps)