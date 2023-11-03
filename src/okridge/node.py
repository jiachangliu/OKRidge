from copy import deepcopy
import numpy as np
import math

import time
import sys

from sklearn.isotonic import IsotonicRegression
import scipy
import gc
import psutil

gc.enable()


def get_RAM_used_in_GB():
    # print(psutil.virtual_memory())
    # print(psutil.virtual_memory()[0])
    return psutil.virtual_memory()[3] / 1000000000


class Node:
    def __init__(self, parent, zlb, zub, **kwargs):
        """
        Initialize a Node
        Parameters
        ----------
        parent: Node or None
            the parent Node
        zlb: np.array
            p x 1 array representing the lower bound of the integer variables z
        zub: np.array
            p x 1 array representing the upper bound of the integer variables z
        Other Parameters
        ----------------
        x: np.array
            The data matrix (n x p). If not specified the data will be
            inherited from the parent node
        y: np.array
            The data vector (n x 1). If not specified the data will be
            inherited from the parent node
        xi_xi: np.array
            The norm of each column in x (p x 1). If not specified the data
            will be inherited from the parent node
        l0: float
            The zeroth norm coefficient. If not specified the data will
            be inherited from the parent node
        l2: float
            The second norm coefficient. If not specified the data will
            be inherited from the parent node
        m: float
            The bound for the features (\beta). If not specified the data will
            be inherited from the parent node
        """
        self.data = kwargs.get("data", parent.data if parent else None)
        self.inherit_parent_upper_sol = kwargs.get("inherit_parent_upper_sol", False)
        self.inherit_parent_lower_sol = kwargs.get("inherit_parent_lower_sol", False)
        if parent is not None:
            self.parent_lower_bound = parent.lower_bound
        else:
            self.parent_lower_bound = 0
        # self.parent_primal = parent.primal_value if parent else None

        # self.r = deepcopy(parent.r) if parent else None
        # if parent:
        #     self.warm_start = \
        #         {i: j for i, j in zip(parent.support, parent.primal_beta)}
        # else:
        #     self.warm_start = None

        self.level = parent.level + 1 if parent else 0

        if zlb is not None:
            self.zlb = zlb
        else:
            self.zlb = np.zeros((self.data.p,), dtype=bool)
        if zub is not None:
            self.zub = zub
        else:
            self.zub = np.ones((self.data.p,), dtype=bool)

        if self.inherit_parent_upper_sol:
            self.upper_bound = parent.upper_bound
            self.upper_beta = parent.upper_beta
            self.upper_r = parent.upper_r
        else:
            self.upper_bound = None
            self.upper_beta = None
            self.upper_r = None

        if self.inherit_parent_lower_sol:
            # self.lower_bound = parent.lower_bound
            self.lower_beta_on_allowed_support = (
                parent.lower_beta_on_allowed_support.copy()
            )
        else:
            self.lower_bound = None
            self.lower_beta_on_allowed_support = None

        self.allowed_support = np.where(self.zub == True)[0]
        self.fixed_support_on_allowed_support = np.where(
            self.zlb[self.allowed_support] == True
        )[0]
        self.unfixed_support_on_allowed_support = np.where(
            self.zlb[self.allowed_support] == False
        )[0]

        self.data_isStored_on_allowed_support = False
        self.lower_solve_used_brute_force = False

    def delete_storedData_on_allowed_support(self):
        RAM_used = get_RAM_used_in_GB()
        # print("before deleting, RAM_used is", RAM_used)
        del self.allowed_support_2d_indices
        del self.XTX_minus_smallest_eigenval_on_allowed_support
        del self.XTX_lambda2_on_allowed_support
        del self.XTy_on_allowed_support
        del self.allowed_support
        del self.fixed_support_on_allowed_support
        del self.unfixed_support_on_allowed_support
        gc.collect()
        # print("after deleting, RAM_used is", RAM_used)

    def store_data_on_allowed_support(self):
        # self.XTX_minus_smallest_eigenval_on_allowed_support = self.data.XTX_minus_smallest_eigenval[self.allowed_support][:, self.allowed_support]
        self.allowed_support_2d_indices = np.ix_(
            self.allowed_support, self.allowed_support
        )
        self.XTX_minus_smallest_eigenval_on_allowed_support = (
            self.data.XTX_minus_smallest_eigenval[self.allowed_support_2d_indices]
        )

        self.XTX_lambda2_on_allowed_support = self.data.XTX_lambda2[
            self.allowed_support_2d_indices
        ]
        self.XTX_minus_smallest_eigenval_on_allowed_support_twice = (
            2 * self.XTX_lambda2_on_allowed_support
        )
        self.XTy_on_allowed_support = self.data.XTy[self.allowed_support]

        self.data_isStored_on_allowed_support = True

    def upper_solve(self, k, upper_solver, parent_size=50, child_size=50):
        if self.data_isStored_on_allowed_support is False:
            self.store_data_on_allowed_support()

        # print("inherit parent upper sol is {}".format(self.inherit_parent_upper_sol))
        if self.inherit_parent_upper_sol:
            return self.upper_bound
        betas_warmstart = np.zeros((len(self.allowed_support),))
        if len(self.fixed_support_on_allowed_support) >= 1:
            sub_betas_warmstart = np.linalg.solve(
                self.data.XTX_lambda2[self.fixed_support_on_allowed_support][
                    :, self.fixed_support_on_allowed_support
                ],
                self.data.XTy[self.fixed_support_on_allowed_support],
            )
            betas_warmstart[self.fixed_support_on_allowed_support] = sub_betas_warmstart

        upper_solver.reset(
            self.data.X[:, self.allowed_support],
            self.data.X_norm_2[self.allowed_support],
            self.data.XTX_lambda2[self.allowed_support][:, self.allowed_support],
            self.data.XTy[self.allowed_support],
        )
        upper_solver.warm_start_from_beta0_betas(0, betas_warmstart)

        upper_solver.get_sparse_sol_via_OMP(
            k=k, parent_size=parent_size, child_size=child_size
        )

        (
            _,
            betas_on_allowed_support,
            r_on_allowed_support,
        ) = upper_solver.get_beta0_betas_r()

        self.upper_beta = np.zeros((self.data.p,))
        self.upper_beta[self.allowed_support] = betas_on_allowed_support

        self.upper_r = np.zeros((self.data.p,))
        self.upper_r[self.allowed_support] = r_on_allowed_support

        self.upper_bound = min(upper_solver.loss_arr_child)

        return self.upper_bound

    def upper_solve_with_cache(self, k, upper_solver):
        if self.lower_solve_used_brute_force:
            return self.upper_bound

        if self.data_isStored_on_allowed_support is False:
            self.store_data_on_allowed_support()

        # print("inherit parent upper sol is {}".format(self.inherit_parent_upper_sol))
        if self.inherit_parent_upper_sol:
            return self.upper_bound

        # allowed_supp_mask = self.allowed_support
        # fixed_supp_mask = np.zeros((self.data.p, ), dtype=bool)
        allowed_supp_mask = self.zub == 1
        fixed_supp_mask = self.zlb == 1

        upper_solver.reset_fixed_supp_and_allowed_supp(
            fixed_supp_mask, allowed_supp_mask
        )

        upper_solver.get_sparse_sol_via_OMP(k=k)

        # _, betas_on_allowed_support, r_on_allowed_support = upper_solver.get_beta0_betas_r()

        # self.upper_beta = np.zeros((self.data.p, ))
        # self.upper_beta[self.allowed_support] = betas_on_allowed_support

        # self.upper_r = np.zeros((self.data.p, ))
        # self.upper_r[self.allowed_support] = r_on_allowed_support

        # self.upper_bound = min(upper_solver.loss_arr_child)

        (
            self.upper_beta,
            self.upper_r,
            self.upper_bound,
        ) = upper_solver.get_betas_r_loss()

        return self.upper_bound

    def lower_solve_brute_force(self, k, upper_solver):
        if self.data_isStored_on_allowed_support is False:
            self.store_data_on_allowed_support()

        allowed_supp_mask = self.zub == 1
        fixed_supp_mask = self.zlb == 1

        upper_solver.reset_fixed_supp_and_allowed_supp(
            fixed_supp_mask, allowed_supp_mask
        )
        upper_solver.get_sparse_sol_via_brute_force(k=k)

        (
            self.upper_beta,
            self.upper_r,
            self.upper_bound,
        ) = upper_solver.get_betas_r_loss()
        self.lower_solve_used_brute_force = True
        self.lower_bound = self.upper_bound

        return self.lower_bound

    def lower_solve(self, k, upper_bound):
        if self.data_isStored_on_allowed_support is False:
            self.store_data_on_allowed_support()

        if len(self.fixed_support_on_allowed_support) >= k:
            self.lower_bound = upper_bound
            return self.lower_bound
        if self.inherit_parent_lower_sol is False:
            # self.lower_beta_on_allowed_support = np.linalg.solve(self.data.XTX_lambda2[self.allowed_support][:, self.allowed_support], self.data.XTy[self.allowed_support])
            # self.lower_beta_on_allowed_support = np.linalg.solve(self.XTX_lambda2_on_allowed_support, self.XTy_on_allowed_support)
            try:
                self.lower_beta_on_allowed_support = scipy.linalg.solve(
                    self.XTX_lambda2_on_allowed_support,
                    self.XTy_on_allowed_support,
                    assume_a="pos",
                )
            except:
                try:
                    self.lower_beta_on_allowed_support = np.linalg.solve(
                        self.XTX_lambda2_on_allowed_support, self.XTy_on_allowed_support
                    )
                except:
                    self.lower_beta_on_allowed_support = np.linalg.pinv(
                        self.XTX_lambda2_on_allowed_support
                    ).dot(self.XTy_on_allowed_support)

        self.lower_bound = self.calculate_lower_bound(
            self.lower_beta_on_allowed_support, k
        )

        return self.lower_bound

    def calculate_lower_bound(self, lower_beta_on_allowed_support, k):

        lower_p_on_allowed_support = (
            2
            / self.data.lambda2_plus_smallest_eigenval
            * (
                self.XTy_on_allowed_support
                - self.XTX_minus_smallest_eigenval_on_allowed_support.dot(
                    lower_beta_on_allowed_support
                )
            )
        )

        lower_z_on_allowed_support = np.zeros((len(lower_beta_on_allowed_support),))

        lower_z_on_allowed_support[self.fixed_support_on_allowed_support] = 1
        lower_p_sq_on_allowed_support = lower_p_on_allowed_support ** 2
        largest_lower_p_sq_indices = np.argsort(
            -lower_p_sq_on_allowed_support[self.unfixed_support_on_allowed_support]
        )[: (k - len(self.fixed_support_on_allowed_support))]
        lower_z_on_allowed_support[
            self.unfixed_support_on_allowed_support[largest_lower_p_sq_indices]
        ] = 1

        term1 = self.XTX_minus_smallest_eigenval_on_allowed_support.dot(
            lower_beta_on_allowed_support
        ).dot(lower_beta_on_allowed_support)
        term2 = -2.0 * self.XTy_on_allowed_support.dot(lower_beta_on_allowed_support)
        term3 = self.data.lambda2_plus_smallest_eigenval * (
            lower_p_on_allowed_support.dot(lower_beta_on_allowed_support)
            - 1 / 4.0 * lower_z_on_allowed_support.dot(lower_p_sq_on_allowed_support)
        )

        lower_bound = term1 + term2 + term3

        return lower_bound

    def finetune_ADMM_rho(self, k, upper_bound, factor=10):
        print("start finetuning ADMM rho")
        print("at the beginning, rho is", self.data.rho_ADMM)
        original_rho = self.data.rho_ADMM

        _ = self.lower_solve(k, upper_bound)
        current_lower_bound = self.lower_solve_admm(k, upper_bound)

        total_factor = 1

        self.data.rho_ADMM = self.data.rho_ADMM / factor
        _ = self.lower_solve(k, upper_bound)
        tmp_lower_bound = self.lower_solve_admm(k, upper_bound)

        if tmp_lower_bound > current_lower_bound:
            factor = 1 / factor
            total_factor = total_factor * factor
            current_lower_bound = tmp_lower_bound

        while True:
            self.data.rho_ADMM = self.data.rho_ADMM * factor
            _ = self.lower_solve(k, upper_bound)
            tmp_lower_bound = self.lower_solve_admm(k, upper_bound)

            if tmp_lower_bound > current_lower_bound:
                total_factor = total_factor * factor
                current_lower_bound = tmp_lower_bound
            else:
                break
        self.data.rho_ADMM = original_rho * total_factor
        print("at the end, rho is", self.data.rho_ADMM)

    def lower_solve_admm(self, k, upper_bound):
        if self.data_isStored_on_allowed_support is False:
            self.store_data_on_allowed_support()

        print("upper_bound is", upper_bound)
        print("initial lower bound is", self.lower_bound)

        lower_beta_on_allowed_support_k = self.lower_beta_on_allowed_support.copy()
        lower_p_on_allowed_support_k = (
            self.XTy_on_allowed_support
            - self.XTX_minus_smallest_eigenval_on_allowed_support.dot(
                lower_beta_on_allowed_support_k
            )
        )
        lower_q_on_allowed_support_k = np.zeros((len(lower_beta_on_allowed_support_k),))

        rho = self.data.rho_ADMM
        # rho = 0.001
        # rho = 0.032267
        # rho = 0.009844818945193998
        print("rho inside ADMM is", rho)
        # rho = 0.0042578934357783045

        # step_1_Q = 2 / rho * np.eye(len(lower_beta_on_allowed_support_k)) + self.XTX_minus_smallest_eigenval_on_allowed_support
        step_1_Q = self.XTX_minus_smallest_eigenval_on_allowed_support.copy()
        np.fill_diagonal(step_1_Q, step_1_Q.diagonal() + 2 / rho)

        step_1_L, step_1_low = scipy.linalg.cho_factor(
            step_1_Q, overwrite_a=True, check_finite=False
        )
        del step_1_Q
        step_2_factor = 1 + 2 / (self.data.lambda2_plus_smallest_eigenval * rho)
        isotonic_clf = IsotonicRegression()

        k_minus_fixed = k - len(self.fixed_support_on_allowed_support)
        print("k_minus_fixed is", k_minus_fixed)

        lower_beta_Q_on_allowed_support_k = np.zeros(
            (len(lower_beta_on_allowed_support_k),)
        )

        for _ in range(20):
            # step 1
            # lower_beta_on_allowed_support_k = np.linalg.solve(step_1_Q, self.XTy_on_allowed_support - lower_p_on_allowed_support_k - lower_q_on_allowed_support_k)
            lower_beta_on_allowed_support_k = scipy.linalg.cho_solve(
                (step_1_L, step_1_low),
                self.XTy_on_allowed_support
                - lower_p_on_allowed_support_k
                - lower_q_on_allowed_support_k,
                overwrite_b=True,
                check_finite=False,
            )

            lower_beta_Q_on_allowed_support_k = (
                2
                * self.XTX_minus_smallest_eigenval_on_allowed_support.dot(
                    lower_beta_on_allowed_support_k
                )
                + lower_p_on_allowed_support_k
                - self.XTy_on_allowed_support
            )

            # lower_beta_Q_on_allowed_support_k =  2 * lower_beta_on_allowed_support_k.dot(self.XTX_minus_smallest_eigenval_on_allowed_support)
            # lower_beta_Q_on_allowed_support_k += (lower_p_on_allowed_support_k - self.XTy_on_allowed_support)

            # np.dot(self.XTX_minus_smallest_eigenval_on_allowed_support_twice, lower_beta_on_allowed_support_k, out=lower_beta_Q_on_allowed_support_k)
            # lower_beta_Q_on_allowed_support_k += (lower_p_on_allowed_support_k - self.XTy_on_allowed_support)

            # step 2
            # lower_p_on_allowed_support_k = self.XTy_on_allowed_support - self.XTX_minus_smallest_eigenval_on_allowed_support.dot(lower_beta_on_allowed_support_k) - lower_q_on_allowed_support_k
            lower_p_on_allowed_support_k = (
                self.XTy_on_allowed_support
                - lower_beta_Q_on_allowed_support_k
                - lower_q_on_allowed_support_k
            )
            lower_p_on_allowed_support_k[
                self.fixed_support_on_allowed_support
            ] /= step_2_factor

            tmp_vec_x = np.abs(
                lower_p_on_allowed_support_k[self.unfixed_support_on_allowed_support]
            )
            tmp_vec_weight = np.ones((len(tmp_vec_x),))
            tmp_vec_y = tmp_vec_x.copy()
            top_k_minus_fixed_indicies = np.argpartition(tmp_vec_x, -k_minus_fixed)[
                -k_minus_fixed:
            ][:k_minus_fixed]
            tmp_vec_weight[top_k_minus_fixed_indicies] = step_2_factor
            tmp_vec_y[top_k_minus_fixed_indicies] /= step_2_factor

            lower_p_on_allowed_support_k[
                self.unfixed_support_on_allowed_support
            ] = np.sign(
                lower_p_on_allowed_support_k[self.unfixed_support_on_allowed_support]
            ) * isotonic_clf.fit_transform(
                X=tmp_vec_x, y=tmp_vec_y, sample_weight=tmp_vec_weight
            )

            # step 3
            # lower_q_on_allowed_support_k += (self.XTX_minus_smallest_eigenval_on_allowed_support.dot(lower_beta_on_allowed_support_k) + lower_p_on_allowed_support_k - self.XTy_on_allowed_support)
            lower_q_on_allowed_support_k += (
                lower_beta_Q_on_allowed_support_k
                + lower_p_on_allowed_support_k
                - self.XTy_on_allowed_support
            )

            # tmp_loss, _, _ = self.calculate_lower_bound(lower_beta_on_allowed_support_k, k)
            # print("loss becomes", tmp_loss)

            tmp_loss = self.calculate_lower_bound(lower_beta_on_allowed_support_k, k)
            print("loss becomes", tmp_loss)

        tmp_lower_bound = self.calculate_lower_bound(lower_beta_on_allowed_support_k, k)
        print("after ADMM, lower bound is {}".format(tmp_lower_bound))

        if tmp_lower_bound > self.lower_bound:
            self.lower_bound = tmp_lower_bound
            self.lower_beta_on_allowed_support = lower_beta_on_allowed_support_k.copy()
        print("we are using {} as the final lower bound".format(self.lower_bound))
        print()

        return self.lower_bound

    def lower_solve_improve(self, k, upper_bound):
        if self.data_isStored_on_allowed_support is False:
            self.store_data_on_allowed_support()

        # print("before improving, lower bound is ", self.lower_bound, "upper bound is ", upper_bound)
        # st = time.time()

        min_loss = -np.Inf

        # subgradient descent with Polyak step size. We get the upper bound from our heuristic solution
        tmp_lower_beta_on_allowed_support = self.lower_beta_on_allowed_support.copy()
        best_lower_beta_on_allowed_support = self.lower_beta_on_allowed_support.copy()
        tmp_lower_bound = self.lower_bound

        Ar = self.XTX_minus_smallest_eigenval_on_allowed_support.dot(
            tmp_lower_beta_on_allowed_support
        )
        c = self.XTy_on_allowed_support - Ar
        c_fixed = c[self.fixed_support_on_allowed_support].copy()
        c[self.fixed_support_on_allowed_support] = np.Inf
        z_chosen_on_allowed_support = np.argpartition(np.abs(c), -k)[(-k):]
        c[self.fixed_support_on_allowed_support] = c_fixed
        sub_grad_prev = -2 * Ar + 2 / self.data.lambda2_plus_smallest_eigenval * c[
            z_chosen_on_allowed_support
        ].dot(
            self.XTX_minus_smallest_eigenval_on_allowed_support[
                z_chosen_on_allowed_support
            ]
        )
        sub_grad_norm2 = np.linalg.norm(sub_grad_prev) ** 2

        print("before subgradient descent, loss is {}".format(self.lower_bound))
        for step in range(1, 200):

            # print("hello", self.calculate_lower_bound(tmp_lower_beta_on_allowed_support, k)[0])
            # get the z indicies
            Ar = self.XTX_minus_smallest_eigenval_on_allowed_support.dot(
                tmp_lower_beta_on_allowed_support
            )
            c = self.XTy_on_allowed_support - Ar
            c_fixed = c[self.fixed_support_on_allowed_support].copy()
            c[self.fixed_support_on_allowed_support] = np.Inf

            z_chosen_on_allowed_support = np.argpartition(np.abs(c), -k)[(-k):]
            # z_indicies = self.allowed_support[z_chosen_on_allowed_support]
            # print(np.sort(z_indicies))
            c[self.fixed_support_on_allowed_support] = c_fixed

            # calculate the subgradient:
            # sub_grad = -2 * Ar + 2/self.data.lambda2_plus_smallest_eigenval * c[z_chosen_on_allowed_support].dot(self.data.XTX_minus_smallest_eigenval[z_indicies][:, self.allowed_support])
            sub_grad = -2 * Ar + 2 / self.data.lambda2_plus_smallest_eigenval * c[
                z_chosen_on_allowed_support
            ].dot(
                self.XTX_minus_smallest_eigenval_on_allowed_support[
                    z_chosen_on_allowed_support
                ]
            )

            # # subgradient method 1
            # tmp_lower_beta_on_allowed_support += 7e-6/math.sqrt(step) * sub_grad # diminishing step size

            # # subgradient method 2
            # tmp_lower_beta_on_allowed_support += sub_grad / np.linalg.norm(sub_grad) * 0.1# constant step size

            # # subgradient method 3
            # step_size = (upper_bound - tmp_lower_bound + 1. / step) / (np.linalg.norm(sub_grad) ** 2) # calculate the step size using Polyak's step size choice
            # tmp_lower_beta_on_allowed_support += step_size * sub_grad # Polyak's method

            # subgradient method 4 CFM
            beta_k = max(0, -1.5 * sub_grad_prev.dot(sub_grad) / sub_grad_norm2)
            sub_grad += beta_k * sub_grad_prev
            sub_grad_norm2 = np.linalg.norm(sub_grad) ** 2
            step_size = (upper_bound - tmp_lower_bound + 1 / step) / sub_grad_norm2
            tmp_lower_beta_on_allowed_support += step_size * sub_grad
            sub_grad_prev = sub_grad.copy()

            # tmp_lower_bound, tmp_lower_z_on_allowed_support, tmp_lower_p_on_allowed_support = self.calculate_lower_bound(tmp_lower_beta_on_allowed_support, k)
            tmp_lower_bound = self.calculate_lower_bound(
                tmp_lower_beta_on_allowed_support, k
            )

            # print(tmp_lower_bound)

            if tmp_lower_bound > min_loss:
                min_loss = tmp_lower_bound
                best_lower_beta_on_allowed_support = (
                    tmp_lower_beta_on_allowed_support.copy()
                )

            # print(tmp_lower_bound)
            # if step % 20 == 0:
            #     print("during subgradient descent, lower bound is", min_loss)
        if min_loss > self.lower_bound:
            self.lower_beta_on_allowed_support = (
                best_lower_beta_on_allowed_support.copy()
            )
            print("subgradient ascent has succeeded")
            print(
                "loss by fast method: {}\n loss by slow method: {}".format(
                    self.lower_bound, min_loss
                )
            )
        else:
            print("subgradient ascent has failed")
            print(
                "loss by fast method: {}\n loss by slow method: {}".format(
                    self.lower_bound, min_loss
                )
            )
        print()

        self.lower_bound = self.calculate_lower_bound(
            self.lower_beta_on_allowed_support, k
        )

        # print(" after improving, lower bound is ", self.lower_bound)
        # print("the best lower bound we get is", min_loss)
        # print("time elapsed is", time.time() - st)
        # sys.exit()

        return self.lower_bound

    def lower_solve_simpler(self, k):
        if self.data_isStored_on_allowed_support is False:
            self.store_data_on_allowed_support()

        self.lower_beta_on_allowed_support = np.linalg.solve(
            self.data.XTX_lambda2[self.allowed_support][:, self.allowed_support],
            self.data.XTy[self.allowed_support],
        )

        tmp_lower_p_on_allowed_support = (
            2
            / self.data.lambda2_plus_smallest_eigenval
            * (
                self.data.XTy[self.allowed_support]
                - self.data.XTX_minus_smallest_eigenval[self.allowed_support][
                    :, self.allowed_support
                ].dot(self.lower_beta_on_allowed_support)
            )
        )

        tmp_lower_z_on_allowed_support = np.zeros(
            (len(self.lower_beta_on_allowed_support),)
        )

        tmp_lower_z_on_allowed_support[self.fixed_support_on_allowed_support] = 1
        lower_p_sq_on_allowed_support = tmp_lower_p_on_allowed_support ** 2
        largest_lower_p_sq_indices = np.argsort(
            -lower_p_sq_on_allowed_support[self.unfixed_support_on_allowed_support]
        )[: (k - len(self.fixed_support_on_allowed_support))]
        tmp_lower_z_on_allowed_support[
            self.unfixed_support_on_allowed_support[largest_lower_p_sq_indices]
        ] = 1

        term1 = (
            -self.data.XTX_minus_smallest_eigenval[self.allowed_support][
                :, self.allowed_support
            ]
            .dot(self.lower_beta_on_allowed_support)
            .dot(self.lower_beta_on_allowed_support)
        )
        term3 = (
            -self.data.lambda2_plus_smallest_eigenval
            / 4.0
            * tmp_lower_z_on_allowed_support.dot(lower_p_sq_on_allowed_support)
        )

        self.lower_bound = term1 + term3

        return self.lower_bound

    def __str__(self):
        return (
            f"level: {self.level}, lower cost: {self.primal_value}, "
            f"upper cost: {self.upper_bound}"
        )

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        if self.level == other.level:
            if self.primal_value is None and other.primal_value:
                return True
            if other.primal_value is None and self.primal_value:
                return False
            elif not self.primal_value and not other.primal_value:
                return self.parent_primal > other.parent_cost
            return self.primal_value > other.lower_bound_value
        return self.level < other.level

    def __lt__(self, other):
        if self.parent_lower_bound - other.parent_lower_bound < -1e-4:
            return True
        elif self.parent_lower_bound - other.parent_lower_bound > 1e-4:
            return False
        else:
            return self.level < other.level


def new_z(node, index):
    new_zlb = node.zlb.copy()
    new_zub = node.zub.copy()
    new_zlb[index] = 1
    new_zub[index] = 0
    return new_zlb, new_zub


def branch(current_node, k):
    unfixed_and_allowed_support = current_node.allowed_support[
        current_node.unfixed_support_on_allowed_support
    ]
    nonzero_unfixed_and_allowed_support = unfixed_and_allowed_support[
        np.nonzero((current_node.upper_beta)[unfixed_and_allowed_support])
    ]

    delta_beta = -current_node.upper_beta[nonzero_unfixed_and_allowed_support]
    increase_in_loss = (
        current_node.data.half_Lipschitz[nonzero_unfixed_and_allowed_support]
        * delta_beta ** 2
        + 2 * current_node.upper_r[nonzero_unfixed_and_allowed_support] * delta_beta
        - 2 * current_node.data.XTy[nonzero_unfixed_and_allowed_support] * delta_beta
    )
    branching_variable = nonzero_unfixed_and_allowed_support[
        np.argmax(increase_in_loss)
    ]

    new_zlb, new_zub = new_z(current_node, branching_variable)
    # right_node = Node(current_node, new_zlb, current_node.zub.copy(), inherit_parent_upper_sol=(len(current_node.fixed_support_on_allowed_support) == k-1))
    right_node = Node(
        current_node,
        new_zlb,
        current_node.zub.copy(),
        inherit_parent_upper_sol=False,
        inherit_parent_lower_sol=True,
    )
    left_node = Node(current_node, current_node.zlb.copy(), new_zub)
    return left_node, right_node
