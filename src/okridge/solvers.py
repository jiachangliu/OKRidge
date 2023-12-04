import numpy as np
import sys
import functools

import gc
import psutil

from .utils import convert_GB_to_bytes

from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

from itertools import combinations
import scipy

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def calculate_size(tmp_dict):
    size = 0
    for tmp_key in tmp_dict:
        size += (sys.getsizeof(tmp_dict[tmp_key][0]) + sys.getsizeof(tmp_dict[tmp_key][1]))
    return size



class linRegModel_unnormalized_big_n:
    def __init__(self, data):
        """Initialize the linear regression model

        Args:
            data (CustomClass): data containing all the necessary information
        """
        self.X = data.X
        self.n, self.p = data.n, data.p

        self.y = data.y.reshape(-1).astype(float)
        self.XTX_lambda2 = data.XTX_lambda2
        self.XTy = data.XTy
        self.betas = np.zeros((self.p, ))

        self.r = self.XTX_lambda2.dot(self.betas)
        self.loss = self.r.dot(self.betas) - 2 * self.XTy.dot(self.betas)

        self.lambda2 = data.lambda2

        self.half_Lipschitz = data.half_Lipschitz
        self.total_child_added = 0
    
    def warm_start_from_betas(self, betas):
        """Warm start the model from a given betas

        Args:
            betas (np.array): 1D array of coefficients
        """
        self.betas = betas
        self.r = self.XTX_lambda2.dot(self.betas)
        self.total_child_added = 0

    def get_betas(self):
        """Get the current betas

        Returns:
            np.array: 1D array of coefficients
        """
        return self.betas

    def get_betas_r(self):
        """Get the current betas, and r

        Returns:
            np.array: 1D array of coefficients
            np.array: 1D array of intermediate values - XTX_lambda2.dot(betas)
        """
        return self.betas, self.r
    
    def get_betas_r_loss(self):
        """Get the current betas, r, and loss

        Returns:
            np.array: 1D array of coefficients
            np.array: 1D array of intermediate values - XTX_lambda2.dot(betas)
            float: loss
        """
        return self.betas, self.r, self.loss
        
    # @functools.cache
    def finetune_on_current_support(self, supp_mask):
        """Finetune the current solution on a given support

        Args:
            supp_mask (np.array): 1D array of boolean values indicating the support

        Returns:
            np.array: 1D array of coefficients on the support
            np.array: 1D array of intermediate values - XTX_lambda2.dot(betas)
            float: loss
        """

        XTX_lambda2_supp_mask = self.XTX_lambda2[supp_mask]
        XTy_supp_mask = self.XTy[supp_mask]
        betas_on_supp = np.linalg.solve(XTX_lambda2_supp_mask[:, supp_mask], XTy_supp_mask)
        r = betas_on_supp.dot(XTX_lambda2_supp_mask)
        loss = r[supp_mask].dot(betas_on_supp) - 2 * XTy_supp_mask.dot(betas_on_supp)

        return betas_on_supp, r, loss


class sparseLogRegModel_big_n(linRegModel_unnormalized_big_n):
    def __init__(self, data, parent_size=10, child_size=10, allowed_supp_mask=None, max_memory_GB=50):
        """Initialize the sparse logistic regression model

        Args:
            data (CustomClass): data containing all the necessary information
            parent_size (int, optional): number of solutions to keep after doing beam search so that these can be used as parent solutions for the next stage of support expansion. Defaults to 10.
            child_size (int, optional): number of solutions to explore for each parent solution during beam search. Defaults to 10.
            allowed_supp_mask (np.array, optional): 1D of boolean values indicating which features are allowed to use. Defaults to None.
            max_memory_GB (int, optional): maximum memory (in GB) to use. Defaults to 50.
        """
        super().__init__(data=data)
        
        self.parent_size = parent_size
        self.child_size = child_size

        self.supp_mask_arr_parent = np.zeros((parent_size, self.p), dtype=bool)
        self.num_parent = 1

        self.total_child_size = self.parent_size * self.child_size
        self.supp_mask_arr_child = np.zeros((self.total_child_size, self.p), dtype=bool)
        self.loss_arr_child = 1e12 * np.ones((self.total_child_size, ))

        if allowed_supp_mask is None:
            self.allowed_supp_mask = np.ones((self.p, ), dtype=bool)
        else:
            self.allowed_supp_mask = allowed_supp_mask
        
        self.saved_solution = {}
        supp_mask_all_False = np.zeros((self.p, ), dtype=bool)
        tmp_support_str = supp_mask_all_False.tobytes()
        self.saved_solution[tmp_support_str] = (None, self.r, 0.0)

        self.max_memory_GB = max_memory_GB
        tmp_vec_bytes = sys.getsizeof(np.zeros((self.p, )))
        self.max_saved_solutions = int(convert_GB_to_bytes(self.max_memory_GB) / tmp_vec_bytes)

        # print("max number of saved solutions is", self.max_saved_solutions)

    def reset_fixed_supp_and_allowed_supp(self, fixed_supp_mask, allowed_supp_mask):
        """Reset the fixed support and allowed support

        Args:
            fixed_supp_mask (np.array): 1D array of boolean values indicating the fixed support
            allowed_supp_mask (np.array): 1D array of boolean values indicating the allowed support
        """
        self.fixed_supp_mask = fixed_supp_mask
        self.allowed_supp_mask = allowed_supp_mask
        self.betas.fill(0)
        tmp_support_str = fixed_supp_mask.tobytes()
        if tmp_support_str in self.saved_solution:
            betas_on_supp_tmp, self.r, self.loss = self.saved_solution[tmp_support_str]
        else:
            betas_on_supp_tmp, self.r, self.loss = self.finetune_on_current_support(fixed_supp_mask)
        self.betas[fixed_supp_mask] = betas_on_supp_tmp
    
    def get_sparse_sol_via_brute_force(self, k):
        """Get sparse solution through brute force

        Args:
            k (int): cardinality of the final sparse solution
        """

        # enumerate all possible support sets
        best_loss = self.loss
        best_supp_str = self.fixed_supp_mask.tobytes()

        total_num_combinations = scipy.special.comb(sum(self.allowed_supp_mask) - sum(self.fixed_supp_mask), k - sum(self.fixed_supp_mask), exact=True)
        total_supp_mask = np.zeros((total_num_combinations, self.p), dtype=bool)
        total_supp_mask[:, self.fixed_supp_mask] = True

        additional_fixed_supp_masks = combinations(list(np.where(np.logical_xor(self.allowed_supp_mask, self.fixed_supp_mask))[0]), k - sum(self.fixed_supp_mask))

        for i, additional_fixed_supp_mask in enumerate(additional_fixed_supp_masks):
            total_supp_mask[i, list(additional_fixed_supp_mask)] = True
            tmp_support_str = total_supp_mask[i].tobytes()
            if tmp_support_str in self.saved_solution:
                betas_on_supp_tmp, r_on_supp_tmp, loss_tmp = self.saved_solution[tmp_support_str]
            else:
                betas_on_supp_tmp, r_on_supp_tmp, loss_tmp = self.finetune_on_current_support(total_supp_mask[i])
                if len(self.saved_solution) < self.max_saved_solutions:
                    self.saved_solution[tmp_support_str] = (betas_on_supp_tmp, r_on_supp_tmp, loss_tmp)

            if loss_tmp < best_loss:
                best_loss = loss_tmp
                best_supp_str = tmp_support_str
        
        self.betas.fill(0)
        best_betas_on_supp, self.r, self.loss = self.finetune_on_current_support(np.frombuffer(best_supp_str, dtype=bool))
        self.betas[np.frombuffer(best_supp_str, dtype=bool)] = best_betas_on_supp

    def get_sparse_sol_via_OMP(self, k):
        """Get sparse solution through beam search and orthogonal matching pursuit (OMP), for level i, each parent solution generates [child_size] child solutions, so there will be [parent_size] * [child_size] number of total child solutions. However, only the top [parent_size] child solutions are retained as parent solutions for the next level i+1

        Args:
            k (int): cardinality of the final sparse solution
        """

        nonzero_indices_set = set(np.where(np.abs(self.betas) > 1e-9)[0])
        num_nonzero = len(nonzero_indices_set)
        zero_indices_set = set(np.where(self.allowed_supp_mask)[0]) - nonzero_indices_set

        if len(zero_indices_set) == 0:
            return
    
        self.supp_mask_arr_parent[0] = np.abs(self.betas) > 1e-9

        self.num_parent = 1
        self.forbidden_support = set()

        while num_nonzero < min(k, self.p):
            num_nonzero += 1
            self.beamSearch_multipleSupports_via_OMP_by_1()
        
        del self.forbidden_support
        gc.collect()

        best_sol_supp_mask = self.supp_mask_arr_parent[0]
        betas_on_supp, self.r, self.loss = self.finetune_on_current_support(best_sol_supp_mask)
        self.betas.fill(0.0)
        self.betas[best_sol_supp_mask] = betas_on_supp
   
    def beamSearch_multipleSupports_via_OMP_by_1(self):
        """Each parent solution generates [child_size] child solutions, so there will be [parent_size] * [child_size] number of total child solutions. However, only the top [parent_size] child solutions are retained as parent solutions for the next level i+1.
        """
        self.loss_arr_child.fill(1e32)
        self.total_child_added = 0

        for i in range(self.num_parent):
            self.expand_parent_i_support_via_OMP_by_1(i)
        
        child_indices = np.argsort(self.loss_arr_child)[:min(self.parent_size, self.total_child_added)] # get indices of children which have the smallest losses
        num_child_indices = len(child_indices)

        # print("size of saved solution dict is {}".format(total_size(self.saved_solution)))


        self.supp_mask_arr_parent[:num_child_indices] = self.supp_mask_arr_child[child_indices]

        self.num_parent = num_child_indices

    def expand_parent_i_support_via_OMP_by_1(self, i):
        """For parent solution i, generate [child_size] child solutions

        Args:
            i (int): index of the parent solution
        """
        fixed_supp_mask = self.supp_mask_arr_parent[i]
        unfixed_and_allowed_mask = np.logical_xor(self.allowed_supp_mask, fixed_supp_mask)
        unfixed_and_allowed_indicies = np.where(unfixed_and_allowed_mask)[0]


        tmp_support_str = self.supp_mask_arr_parent[i].tobytes()
        if tmp_support_str in self.saved_solution:
            _, r_parent_i, _ = self.saved_solution[tmp_support_str]
        else:
            betas_on_supp_tmp, r_parent_i, loss_tmp = self.finetune_on_current_support(self.supp_mask_arr_parent[i])
            if len(self.saved_solution) < self.max_saved_solutions:
                self.saved_solution[tmp_support_str] = (betas_on_supp_tmp, r_parent_i, loss_tmp)
        half_grad_on_unfixed_and_allowed_supp = r_parent_i[unfixed_and_allowed_indicies] - self.XTy[unfixed_and_allowed_indicies]
        abs_half_grad_on_unfixed_and_allowed_supp =  half_grad_on_unfixed_and_allowed_supp ** 2 / self.half_Lipschitz[unfixed_and_allowed_indicies]

        num_new_js = min(self.child_size, len(unfixed_and_allowed_indicies))
        new_js = unfixed_and_allowed_indicies[np.argsort(-abs_half_grad_on_unfixed_and_allowed_supp)][:num_new_js]
        child_start, child_end = i * self.child_size, i*self.child_size + num_new_js

        self.supp_mask_arr_child[child_start:child_end] = self.supp_mask_arr_parent[i]

        for l in range(num_new_js):
            child_id = child_start + l
            self.supp_mask_arr_child[child_id, new_js[l]] = True
            tmp_support_str = self.supp_mask_arr_child[child_id].tobytes()
            if tmp_support_str not in self.forbidden_support:
                self.total_child_added += 1
                self.forbidden_support.add(tmp_support_str)

                if tmp_support_str in self.saved_solution:
                    _, _, self.loss_arr_child[child_id] = self.saved_solution[tmp_support_str]
                else:
                    betas_on_supp_tmp, r_tmp, self.loss_arr_child[child_id] = self.finetune_on_current_support(self.supp_mask_arr_child[child_id])
                    if len(self.saved_solution) <= self.max_saved_solutions:
                        self.saved_solution[tmp_support_str] = (betas_on_supp_tmp, r_tmp, self.loss_arr_child[child_id])
            
            # print("forbidden_support has {} elements".format(len(self.forbidden_support)))
            # print("saved_solution has {} elements".format(len(self.saved_solution)))
