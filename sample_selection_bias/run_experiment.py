import numpy as np
import pandas as pd
import argparse


from sample_selection_bias.fair_covariate_shift import eopp_fair_covariate_shift_logloss

def run(dataset, repeat_count=1, mu_range=[-1.5, 1.5], C=0.001, eps=0.001, kdebw=0.3):
    n = repeat_count
    sample = dataset
    
    errs, violations = [], []
    for i in range(n):
        print(
            "------------------------------- sample {:d} / {:d}---------------------------------".format(
                i + 1, n
            )
        )
        h = eopp_fair_covariate_shift_logloss(
            verbose=1, tol=1e-7, random_initialization=True
        )
        h.trg_grp_marginal_matching = True
        h.C = C
        h.max_epoch = 3
        h.max_iter = 3000
        h.tol = 1e-7
        h.random_start = True
        h.verbose = 1
        h.fit(
            sample["X_src"],
            sample["Y_src"],
            sample["A_src"],
            sample["ratio_src"],
            sample["X_trg"],
            sample["A_trg"],
            sample["ratio_trg"],
            mu_range=mu_range,
        )
        err = 1 - h.score(
            sample["X_trg"], sample["Y_trg"], sample["A_trg"], sample["ratio_trg"]
        )
        violation = abs(
            h.fairness_violation(
                sample["X_trg"], sample["Y_trg"], sample["A_trg"], sample["ratio_trg"]
            )
        )
        errs.append(err)
        violations.append(violation)
        print(
            "Test  - prediction_err : {:.3f}\t fairness_violation : {:.3f} ".format(
                err, violation
            )
        )
        print("Mu = {:.4f}".format(h.mu))
        print("")

    print(
        "------------------------------- Summary: {:d} samples---------------------------------".format(
            n
        )
    )
    errs = np.array(errs, dtype=float)
    violations = np.array(violations, dtype=float)
    print(
        "Test  - prediction_err : {:.3f} \u00B1 {:.3f} \t fairness_violation : {:.3f} \u00B1 {:.3f} ".format(
            errs.mean(),
            1.96 / np.sqrt(n) * errs.std(),
            violations.mean(),
            1.96 / np.sqrt(n) * violations.std(),
        )
    )

    return errs.mean(), errs.std(), violations.mean(), violations.std()
