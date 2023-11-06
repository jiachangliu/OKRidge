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
from okridge.utils import download_file_from_google_drive
from pathlib import Path

def test_BNBTree():

    k = 10
    data_file_path = "./tests/Synthetic_n=6000_p=3000_k=10_rho=0.5_snr=5.0_seed=0.npy"

    if not os.path.isfile(data_file_path):
        download_file_from_google_drive('1lizlnufRBmEzMNpr0OlgE-P7otC8opkX', data_file_path)

    loaded_data = np.load(data_file_path, allow_pickle=True)
    X, y = loaded_data.item().get("X"), loaded_data.item().get("y")

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

    indices_support = np.nonzero(betas)[0]
    true_indices_support = np.asarray([0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700], dtype=np.int64)

    assert np.all(indices_support == true_indices_support)

if __name__ == "__main__":
    test_BNBTree()