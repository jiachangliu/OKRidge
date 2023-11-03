import time
import queue
import sys
from collections import namedtuple

import numpy as np

import matplotlib.pyplot as plt

from .node import Node, branch  # , presolve

# from .utilities import branch, is_integral

from beamsearchregression.sparseBeamSearch import sparseLogRegModel_big_n
from .solvers import (
    sparseLogRegModel_big_n as sparseLogRegModel_big_n_with_cache,
    total_size,
)

import psutil
import gc
import scipy

from .utils import (
    get_RAM_available_in_bytes,
    get_RAM_available_in_GB,
    get_RAM_used_in_bytes,
    get_RAM_used_in_GB,
)


class DataClass:
    def __init__(self, X, y, lambda2):
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.X = X
        self.y = y
        self.X_norm_2 = np.linalg.norm(X, axis=0) ** 2
        self.XTX = X.T @ X
        # self.X_norm_2 = self.XTX.diagonal() ** 2
        self.XTX_lambda2 = self.XTX + lambda2 * np.eye(N=self.p)
        self.XTy = y.dot(self.X)
        self.lambda2 = lambda2
        self.half_Lipschitz = self.XTX_lambda2.diagonal()

        eigenvalues = np.linalg.eigvalsh(self.XTX)
        # print("smallest, second smallest, third smallest, and largest eigenvalues are", eigenvalues[0], eigenvalues[1], eigenvalues[2], eigenvalues[-1])
        self.smallest_eigenval = np.linalg.eigvalsh(self.XTX)[0] * 0.999
        print("we take lambda to be", self.smallest_eigenval)

        # print("optimal rho should be", 2/np.sqrt((eigenvalues[0]-self.smallest_eigenval)*(eigenvalues[-1] - self.smallest_eigenval)))
        # print("second optimal rho should be", 2/np.sqrt((eigenvalues[1]-self.smallest_eigenval)*(eigenvalues[-1] - self.smallest_eigenval)))
        # print("third optimal rho should be", 2/np.sqrt((eigenvalues[2]-self.smallest_eigenval)*(eigenvalues[-1] - self.smallest_eigenval)))

        self.rho_ADMM = 2 / np.sqrt(
            (eigenvalues[1] - self.smallest_eigenval)
            * (eigenvalues[-1] - self.smallest_eigenval)
        )

        self.rho_ADMM_is_finetuned = True
        if eigenvalues[1] < self.smallest_eigenval + 1e-6:
            self.rho_ADMM_is_finetuned = False
        # eigenvalues_pos = eigenvalues[eigenvalues > 1e-8]
        # self.rho_ADMM = 2/np.sqrt((eigenvalues_pos[0]-self.smallest_eigenval)*(eigenvalues[-1] - self.smallest_eigenval))

        # self.rho_ADMM = 50/np.sqrt((eigenvalues[0]-self.smallest_eigenval)*(eigenvalues[-1] - self.smallest_eigenval))

        if self.smallest_eigenval < 1e-3:
            self.smallest_eigenval = 0.0
        self.XTX_minus_smallest_eigenval = (
            self.XTX - np.eye(self.p) * self.smallest_eigenval
        )
        self.lambda2_plus_smallest_eigenval = self.lambda2 + self.smallest_eigenval


class BNBTree:
    def __init__(
        self,
        X,
        y,
        int_tol=1e-4,
        gap_tol=1e-4,
        lambda2=1e-5,
        max_memory_GB=300,
        useBruteForce=False,
        tighten_bound=True,
    ):
        """
        Initiate a BnB Tree to solve the least squares regression problem with
        l0l2 regularization

        Parameters
        ----------
        X: np.array
            n x p numpy array
        y: np.array
            1 dimensional numpy array of size n
        int_tol: float, optional
            The integral tolerance of a variable. Default 1e-4
        rel_tol: float, optional
            primal-dual relative tolerance. Default 1e-4
        """
        data = DataClass(X, y, lambda2)
        self.data = data

        self.int_tol = int_tol
        self.gap_tol = gap_tol

        self.useBruteForce = useBruteForce
        self.bruteForceThreshold = 1000

        self.tighten_bound = tighten_bound

        # The number of features

        self.bfs_queue = None
        self.dfs_queue = None

        self.levels = {}
        # self.leaves = []
        self.number_of_nodes = 0

        self.root = None

        # self.upper_solver = sparseLogRegModel_big_n(X=self.data.X, y=self.data.y, lambda2=self.data.lambda2, intercept=False)
        available_memory_GB = get_RAM_available_in_GB()
        if max_memory_GB is None:
            print(
                "No max_memory_GB is given. Using all available memory ({} GB) in the machine".format(
                    available_memory_GB
                )
            )
            self.max_memory_GB = available_memory_GB
        elif max_memory_GB > available_memory_GB:
            print(
                "max_memory_GB is larger than available memory. Using all available memory ({} GB) in the machine".format(
                    available_memory_GB
                )
            )
            self.max_memory_GB = available_memory_GB
        else:
            print("Using max memory ({} GB)".format(max_memory_GB))
            self.max_memory_GB = max_memory_GB
        self.safe_max_memory_GB = 0.95 * self.max_memory_GB
        self.RAM_used_GB_start = get_RAM_used_in_GB()

        self.upper_solver_with_cache = sparseLogRegModel_big_n_with_cache(
            data=self.data,
            intercept=False,
            parent_size=50,
            child_size=50,
            max_memory_GB=50,
        )

    def get_RAM_used_since_start(self):
        RAM_used_GB = get_RAM_used_in_GB()
        return RAM_used_GB - self.RAM_used_GB_start

    def solve(
        self,
        k,
        gap_tol=1e-2,
        number_of_dfs_levels=0,
        verbose=False,
        time_limit=3600,
        warm_start=None,
    ):
        """
        Solve the least squares problem with l2 regularization and sparsity constraint

        Parameters
        ----------
        k: int
            Number of nonzero coefficients
        l2: float
            The second norm coefficient
        gap_tol: float, optional
            the relative gap between the upper and lower bound after which the
            algorithm will be terminated. Default 1e-2
        warm_start: np.array, optional
            (p x 1) array representing a warm start
        number_of_dfs_levels: int, optional
            number of levels to solve as dfs. Default is 0
        verbose: int, optional
            print progress. Default False
        time_limit: float, optional
            The time (in seconds) after which the solver terminates.
            Default is 3600
        Returns
        -------
        tuple
            cost, beta, sol_time, lower_bound, gap
        """
        st = time.time()
        upper_beta = np.zeros((self.data.p,))
        upper_bound = self.data.XTX_lambda2.dot(upper_beta).dot(upper_beta)

        # upper_bound, upper_beta, support = self. \
        # _warm_start(warm_start, verbose, k)
        if verbose:
            print(f"initializing took {time.time() - st} seconds")

        st = time.time()
        # root node
        self.root = Node(None, None, None, data=self.data)

        # if warm_start is not None:
        #     self.root = presolve(self.root, k)

        self.bfs_queue = queue.Queue()
        self.dfs_queue = queue.LifoQueue()
        self.bfs_queue.put(self.root)

        # lower and upper bounds initialization
        lower_bound, dual_bound = {}, {}
        self.levels = {0: 1}
        min_open_level = 0

        max_lower_bound_value = -sys.maxsize
        best_gap = gap_tol + 1

        if verbose:
            print(f"{number_of_dfs_levels} levels of depth used")

        # upper_bound = self.root.upper_solve(k, self.upper_solver)
        upper_bound = self.root.upper_solve_with_cache(k, self.upper_solver_with_cache)
        upper_beta = self.root.upper_beta.copy()
        if (k == 1) or (time_limit < 0):
            max_lower_bound_value = upper_bound
            best_gap = 0
            return (
                upper_bound,
                upper_beta,
                best_gap,
                max_lower_bound_value,
                time.time() - st,
            )
        
        print("initial upper_bound is", upper_bound)
        print("initial nonzero indices are", np.nonzero(upper_beta)[0])

        if self.root.data.rho_ADMM_is_finetuned is False:
            self.root.finetune_ADMM_rho(k, upper_bound, factor=10)

        RAM_used_GB_since_start = self.get_RAM_used_since_start()
        # keep searching through the queue if the queue is not empty AND time limit is not reached
        while (
            (self.bfs_queue.qsize() > 0 or self.dfs_queue.qsize() > 0)
            and (time.time() - st < time_limit)
            and (RAM_used_GB_since_start < self.safe_max_memory_GB)
        ):

            gc.collect()

            RAM_used_GB_since_start = self.get_RAM_used_since_start()
            print(
                "!!! number of saved solutions is",
                len(self.upper_solver_with_cache.saved_solution),
            )

            # get current node
            if self.dfs_queue.qsize() > 0:
                curr_node = self.dfs_queue.get()
            else:
                curr_node = self.bfs_queue.get()

            # prune?
            if (
                curr_node.parent_lower_bound
                and upper_bound <= curr_node.parent_lower_bound
            ):
                self.levels[curr_node.level] -= 1
                # self.leaves.append(current_node)
                # curr_node.delete_storedData_on_allowed_support()
                # del curr_node

                continue

            # calculate dual values
            curr_dual = self._solve_node(
                curr_node, k, dual_bound, upper_bound, tighten_bound=self.tighten_bound
            )

            # prune?with newly calculated lower bound
            if curr_dual >= upper_bound:
                curr_node.delete_storedData_on_allowed_support()
                # del curr_node
                continue

            # curr_upper_bound = curr_node.upper_solve(k, self.upper_solver)
            curr_upper_bound = curr_node.upper_solve_with_cache(
                k, self.upper_solver_with_cache
            )

            # if verbose:
            #     print("curr_upper_bound: {}, curr_dual: {}".format(curr_upper_bound, curr_dual)) # debugging

            if curr_upper_bound < upper_bound:
                upper_bound = curr_upper_bound
                upper_beta = curr_node.upper_beta.copy()
                best_gap = (upper_bound - max_lower_bound_value) / abs(upper_bound)
                print("********************************\nfind better solution with upper_bound", upper_bound)
                print("the nonzero indices are", np.nonzero(upper_beta)[0], "\n********************************")

            # update gap?
            if self.levels[min_open_level] == 0:
                print(
                    "there are {} nodes left in the bfs queue".format(
                        self.bfs_queue.qsize()
                    )
                )
                del self.levels[min_open_level]
                max_lower_bound_value = max(
                    [j for i, j in dual_bound.items() if i <= min_open_level]
                )
                best_gap = (upper_bound - max_lower_bound_value) / abs(upper_bound)
                if verbose:
                    print(
                        f"l: {min_open_level}, (d: {max_lower_bound_value}, "
                        f"u: {upper_bound}, g: {best_gap}, "
                        f"t: {time.time() - st} s"
                    )
                min_open_level += 1

            # arrived at a solution?
            if best_gap <= gap_tol:
                print("reaching gap_tol mid way!")
                print(
                    "there are {} nodes left".format(
                        self.bfs_queue.qsize() + self.dfs_queue.qsize()
                    )
                )
                return (
                    upper_bound,
                    upper_beta,
                    best_gap,
                    max_lower_bound_value,
                    time.time() - st,
                )

            # branch?
            curr_gap = (curr_upper_bound - curr_dual) / abs(curr_upper_bound)
            if curr_gap <= gap_tol:
                print("curr_gap is smaller than gap_tol; skipping!")
                pass
            elif (curr_dual < upper_bound) and (
                len(curr_node.fixed_support_on_allowed_support) < k
            ):
                left_node, right_node = branch(curr_node, k)
                self.levels[curr_node.level + 1] = (
                    self.levels.get(curr_node.level + 1, 0) + 2
                )
                if curr_node.level < min_open_level + number_of_dfs_levels:
                    self.dfs_queue.put(right_node)
                    self.dfs_queue.put(left_node)
                else:
                    self.bfs_queue.put(right_node)
                    self.bfs_queue.put(left_node)
            else:
                print(
                    "fixed support size is {}".format(
                        len(curr_node.fixed_support_on_allowed_support)
                    )
                )
                print("no branching!!!")
                pass

            curr_node.delete_storedData_on_allowed_support()
            # del curr_node

            # print("total size of saved_solution is", total_size(self.upper_solver_with_cache.saved_solution))
        
        print("counting number of heuristic solutions")
        print("number of heuristic solutions is", len(self.upper_solver_with_cache.saved_solution))
        losses = []
        sub_betas = []
        indices_strs = []
        yTy = self.data.y.dot(self.data.y)

        for indices_str in self.upper_solver_with_cache.saved_solution.keys():
            sub_beta, _, loss = self.upper_solver_with_cache.saved_solution[indices_str]
            losses.append(loss + yTy)
            sub_betas.append(sub_beta)
            indices_strs.append(indices_str)
        
        loss_indices = np.argsort(losses)
        print("smallest 10 losses are", np.sort(losses)[:30])

        beta_collections = np.zeros((1000, self.data.p))
        loss_collections = []
        for i in range(1000):
            loss_collections.append(losses[loss_indices[i]])
            nonzero_indices = np.fromstring(indices_strs[loss_indices[i]], dtype=bool).nonzero()[0].astype(int)
            print("nonzero_indices are", nonzero_indices)
            print("sub_beta is", sub_betas[loss_indices[i]])
            beta_collections[i, nonzero_indices] = sub_betas[loss_indices[i]]


        plt.figure()
        plt.hist(loss_collections, bins=100)
        plt.xlabel("loss")
        plt.ylabel("frequency")
        plt.title("top 1000 Rashomon set solutions")
        plt.savefig("tmp_losses.png")

        plt.figure()
        plt.step(np.arange(1000), np.sort(loss_collections))
        plt.xlabel("loss ranking index")
        plt.ylabel("loss")
        plt.title("loss step plot of top 1000 Rashomon set solutions")
        plt.savefig("tmp_losses_step.png")

        plt.figure()
        beta_collections_mean = np.mean(beta_collections, axis=0)
        beta_collections_std = np.std(beta_collections, axis=0)
        plt.errorbar(np.arange(self.data.p), beta_collections_mean, yerr=beta_collections_std, fmt='o')
        plt.xlabel("feature index")
        plt.ylabel("beta")
        plt.title("coefficients of top 1000 Rashomon set solutions")
        plt.savefig("tmp_losses_beta.png")

        sys.exit()

        if not (self.bfs_queue.qsize() > 0 or self.dfs_queue.qsize() > 0):
            print("There are no nodes left in the queue")
            # update lower bound and gap
            max_lower_bound_value = upper_bound
            best_gap = 0.0
        elif RAM_used_GB_since_start >= self.safe_max_memory_GB:
            print(
                "RAM used since start is greater than 0.95*max_memory(0.95*{} GB = {} GB)!".format(
                    self.max_memory_GB, self.safe_max_memory_GB
                )
            )
        else:
            print("Time limit is reached!")
        return (
            upper_bound,
            upper_beta,
            best_gap,
            max_lower_bound_value,
            time.time() - st,
        )

    def _solve_node(self, curr_node, k, dual_, upper_bound, tighten_bound=True):
        self.number_of_nodes += 1
        total_num_bruteForce = scipy.special.comb(
            len(curr_node.unfixed_support_on_allowed_support),
            k - len(curr_node.fixed_support_on_allowed_support),
        )
        if self.useBruteForce and (total_num_bruteForce < self.bruteForceThreshold):
            curr_dual = curr_node.lower_solve_brute_force(
                k, self.upper_solver_with_cache
            )
            print("lower bound by brute force method is", curr_dual)
        else:
            curr_dual = curr_node.lower_solve(k, upper_bound)
            print("lower bound by fast method is", curr_dual)

            if tighten_bound and (curr_dual < upper_bound):
                # curr_dual = curr_node.lower_solve_improve(k, upper_bound)
                curr_dual = curr_node.lower_solve_admm(k, upper_bound)
                print("lower bound by admm method is", curr_dual)

        dual_[curr_node.level] = min(curr_dual, dual_.get(curr_node.level, sys.maxsize))
        self.levels[curr_node.level] -= 1
        return curr_dual

    def _warm_start(self, warm_start, verbose, k, l2):
        if warm_start is None:
            return sys.maxsize, None, None
        else:
            if verbose:
                print("used a warm start")
            support = np.nonzero(warm_start)[0]
            upper_bound, upper_beta = upper_bound_solve(
                self.data.X, self.data.y, k, l2, support
            )
            return upper_bound, upper_beta, support

    def solve_root(
        self,
        k,
        gap_tol=1e-2,
        number_of_dfs_levels=0,
        verbose=False,
        time_limit=3600,
        warm_start=None,
    ):

        st = time.time()
        upper_beta = np.zeros((self.data.p,))
        upper_bound = self.data.XTX_lambda2.dot(upper_beta).dot(upper_beta)

        if verbose:
            print(f"initializing took {time.time() - st} seconds")

        st = time.time()
        # root node
        self.root = Node(None, None, None, data=self.data)

        lower_bound, dual_bound = {}, {}
        self.levels = {0: 1}
        max_lower_bound_value = -sys.maxsize
        best_gap = gap_tol + 1

        # calculate upper bound
        upper_bound = self.root.upper_solve_with_cache(k, self.upper_solver_with_cache)
        upper_beta = self.root.upper_beta.copy()

        # calculate lower bound
        curr_dual = self._solve_node(
            self.root, k, dual_bound, upper_bound, tighten_bound=self.tighten_bound
        )
        max_lower_bound_value = curr_dual
        best_gap = (upper_bound - max_lower_bound_value) / abs(upper_bound)

        if k == 1:
            max_lower_bound_value = upper_bound
            best_gap = 0

        return (
            upper_bound,
            upper_beta,
            best_gap,
            max_lower_bound_value,
            time.time() - st,
        )