import prepare_data
import fair_covariate_shift.run_experiment
import argparse

default_prepare = prepare_data.prepare_noisy_dataset
data2prepare = {
    "compas": prepare_data.prepare_compas,
    "german": prepare_data.prepare_german,
    "drug": prepare_data.prepare_drug,
    "arrhythmia": prepare_data.prepare_arrhythmia,
}

default_reg = 0.001
dataset2reg = {
    "compas": 0.001,
    "german": 0.01,
    "drug": 0.001,
    "arrhythmia": 0.01,
}

default_eps = 0.001
dataset2eps = {
    "compas": 0.001,
    "german": 0.001,
    "drug": 0.001,
    "arrhythmia": 0.001,
}

exp2exp = {
    "fair_covariate_shift": fair_covariate_shift.run_experiment.run
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help='Experiment to run: [fair_covariate_shift].',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help='Dataset name : ["adult","compas","income","synthetic_20"].',
    )
    parser.add_argument(
        "--corruption_type",
        type=str,
        required=True,
        help="The corruption type for the dataset. Can be \"balanced\", \"bias\", or \"flip\".",
    )
    parser.add_argument(
        "--subcategory",
        type=str,
        required=True,
        help="The subcategory for the dataset. Can be \"0.1\" or \"0.3\".",
    )
    parser.add_argument(
        "--protected_attr",
        type=str,
        required=True,
        help="The protected attribute for the dataset..",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=False,
        default='noisy_datasets',
        help='Dataset directory.',
    )
    parser.add_argument(
        "--repeat",
        type=int,
        required=False,
        default=1,
        help="number of random shuffle runs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=1,
        help="Shift the Gaussian mean -> mean + alpha in sampling of covariates.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=False,
        default=2,
        help="Scale the Gaussian std -> std / beta in sampling of covariates.",
    )
    parser.add_argument(
        "--mu_range",
        type=float,
        required=False,
        nargs="+",
        default=[-1.5, 1.5],
        help="The search range for \mu - the fairness penalty weight for the \"Robust Fairness under Covariate Shift\" experiment.",
    )

    args = parser.parse_args()

    # Experiment arguments
    dataset = args.dataset
    sample_size_ratio = 0.4
    alpha = args.alpha
    beta = args.beta
    kdebw = 0.3
    mu_range = args.mu_range

    dataset_dir = args.dataset_dir + "/" + args.dataset + "/" + args.corruption_type + "/" + args.subcategory + "/"
    protected_attr = args.protected_attr

    # If it is one of the datasets in the fair_covariate_shift repo (used for testing right now)
    if dataset in dataset2eps.keys():
        eps = dataset2eps[dataset]
        C = dataset2reg[dataset]
        dataA, dataY, dataX = data2prepare[dataset]()
    else: # If it is a noisy dataset provided by Saurav
        eps = default_eps
        C = default_reg
        dataA, dataY, dataX = default_prepare(dataset, dataset_dir, protected_attr)


    print(dataX)
    print(type(dataX))
    print(dataY)
    print(type(dataY))
    print(dataA)
    print(type(dataA))

    sample = prepare_data.prepare_dataset(dataA, dataY, dataX, alpha, beta, kdebw, eps, sample_size_ratio)

    exp2exp[args.experiment](sample, 1, mu_range, C, eps, kdebw)
    