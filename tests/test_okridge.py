from okridge import okridge

import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6

import numpy as np

# print(np.__config__.show())
import time
import sys

from okridge.tree import BNBTree

import cProfile, pstats


def dataset(n, p, k):
    X = np.random.uniform(-1.0, 1.0, (n, p))
    w = np.zeros(p)
    nonzero_indices = np.random.choice(p, k, replace=False)
    print(sorted(nonzero_indices))
    w[nonzero_indices] = 1.0
    w[nonzero_indices[0]] = 28.0
    w[nonzero_indices[-1]] = -20.0
    y = X.dot(w) + np.random.normal(0.0, 0.2, n)
    return X, y


def main():

    k = 10
    loaded_data = np.load(
        "/usr/xtmp/jl888/BeamSearchRegression/tests/data011723/dataMethod=genSynthetic4_n=10000_p=3000_k=10_rho=0.9_snr=5.0_seed=0.npy",
        allow_pickle=True,
    )
    X, y = loaded_data.item().get("X"), loaded_data.item().get("y")

    # loaded_data = np.load("/usr/xtmp/jl888/BeamSearchRegression/tests/data011723/dataMethod=genSynthetic4_n=4000_p=3000_k=10_rho=0.1_snr=5.0_seed=0.npy", allow_pickle=True)
    # X, y = loaded_data.item().get('X'), loaded_data.item().get('y')

    # data = {'X': X, 'y': y, 'beta': b}
    # np.save("/usr/xtmp/jl888/BeamSearchRegression/tests/dataPathological/dataMethod=genSynthetic4_n=3000_p=3000_k=10_rho=0.01_snr=10.0_seed=0.npy", data)
    # X, y = loaded_data.item().get('X'), loaded_data.item().get('y')

    # lambda2 = 1e-3
    lambda2 = 1e1

    # # ours method
    start_t = time.time()
    BnB_manager = BNBTree(X, y, lambda2=lambda2)
    (
        upper_bound,
        betas,
        optimality_gap,
        max_lower_bound,
        running_time,
    ) = BnB_manager.solve(k=k, gap_tol=1e-4, verbose=True, time_limit=60)
    print("time used including set up is", time.time() - start_t)
    print(upper_bound, betas, optimality_gap, max_lower_bound, running_time)
    print("ours method time used is {}".format(running_time))
    print(
        "upper bound: {}, lower bound: {}, gap: {}, calculated gap: {}".format(
            upper_bound,
            max_lower_bound,
            optimality_gap,
            (upper_bound - max_lower_bound) / abs(upper_bound),
        )
    )
    print(np.nonzero(betas))
    print("ours method is finished!")
    print("time used including set up and extra printing is", time.time() - start_t)

if __name__ == "__main__":
    main()