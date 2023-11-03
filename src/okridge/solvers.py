import numpy as np
import sys
import functools

import gc
import psutil

from .utils import convert_GB_to_bytes

# from __future__ import print_function
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
    def __init__(self, data, intercept=True):
        self.X = data.X
        # self.X_norm_2 = data.X_norm_2
        self.n, self.p = data.n, data.p

        self.y = data.y.reshape(-1).astype(float)
        # self.XT = np.transpose(self.X)
        self.XTX_lambda2 = data.XTX_lambda2
        self.XTy = data.XTy
        self.beta0 = 0.0
        self.betas = np.zeros((self.p, ))

        self.r = self.XTX_lambda2.dot(self.betas)
        self.loss = self.r.dot(self.betas) - 2 * self.XTy.dot(self.betas)

        self.intercept = intercept
        self.lambda2 = data.lambda2

        # self.half_Lipschitz = self.X_norm_2 + self.lambda2
        # self.half_Lipschitz = self.XTX_lambda2.diagonal()
        self.half_Lipschitz = data.half_Lipschitz
        self.total_child_added = 0
    
    def warm_start_from_beta0_betas(self, beta0, betas):
        self.beta0, self.betas = beta0, betas
        self.r = self.XTX_lambda2.dot(self.betas)
        self.total_child_added = 0

    def get_beta0_betas(self):
        return self.beta0, self.betas

    def get_beta0_betas_r(self):
        return self.beta0, self.betas, self.r
    
    def get_betas_r_loss(self):
        return self.betas, self.r, self.loss
        
    # @functools.cache
    def finetune_on_current_support(self, supp_mask):
        # betas_on_supp = np.linalg.solve(self.XTX_lambda2[supp_mask][:, supp_mask], self.XTy[supp_mask])
        # r = betas_on_supp.dot(self.XTX_lambda2[supp_mask])
        # loss = r[supp_mask].dot(betas_on_supp) - 2 * self.XTy[supp_mask].dot(betas_on_supp)

        XTX_lambda2_supp_mask = self.XTX_lambda2[supp_mask]
        XTy_supp_mask = self.XTy[supp_mask]
        betas_on_supp = np.linalg.solve(XTX_lambda2_supp_mask[:, supp_mask], XTy_supp_mask)
        r = betas_on_supp.dot(XTX_lambda2_supp_mask)
        loss = r[supp_mask].dot(betas_on_supp) - 2 * XTy_supp_mask.dot(betas_on_supp)

        return betas_on_supp, r, loss

# https://algrt.hm/2022-04-02-caching-a-function-with-unhashable-arguments/


# from functools import lru_cache

# def hash_list(l: list) -> int:
#     __hash = 0
#     for i, e in enumerate(l):
#         __hash = hash((__hash, i, hash_item(e)))
#     return __hash

# def hash_dict(d: dict) -> int:
#     __hash = 0
#     for k, v in d.items():
#         __hash = hash((__hash, k, hash_item(v)))
#     return __hash

# def hash_item(e) -> int:
#     if hasattr(e, '__hash__') and callable(e.__hash__):
#         try:
#             return hash(e)
#         except TypeError:
#             pass
#     if isinstance(e, (list, set, tuple)):
#         return hash_list(list(e))
#     elif isinstance(e, (dict)):
#         return hash_dict(e)
#     else:
#         raise TypeError(f'unhashable type: {e.__class__}')

# def my_lru_cache(*opts, **kwopts):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             __hash = hash_item([id(func)] + list(args) + list(kwargs.items()))

#             @lru_cache(*opts, **kwopts)
#             def cached_func(args_hash):
#                 return func(*args, **kwargs)
            
#             return cached_func(__hash)
#         return wrapper
#     return decorator


class sparseLogRegModel_big_n(linRegModel_unnormalized_big_n):
    def __init__(self, data, intercept=True, parent_size=10, child_size=10, allowed_supp_mask=None, max_memory_GB=50):
        super().__init__(data=data, intercept=intercept)
        
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
        tmp_support_str = supp_mask_all_False.tostring()
        self.saved_solution[tmp_support_str] = (None, self.r, 0.0)

        self.max_memory_GB = max_memory_GB
        tmp_vec_bytes = sys.getsizeof(np.zeros((self.p, )))
        self.max_saved_solutions = int(convert_GB_to_bytes(self.max_memory_GB) / tmp_vec_bytes)
        # self.max_saved_solutions = 10

        print("max number of saved solutions is", self.max_saved_solutions)

    def reset_fixed_supp_and_allowed_supp(self, fixed_supp_mask, allowed_supp_mask):
        self.fixed_supp_mask = fixed_supp_mask
        self.allowed_supp_mask = allowed_supp_mask
        self.betas.fill(0)
        tmp_support_str = fixed_supp_mask.tostring()
        if tmp_support_str in self.saved_solution:
            betas_on_supp_tmp, self.r, self.loss = self.saved_solution[tmp_support_str]
        else:
            betas_on_supp_tmp, self.r, self.loss = self.finetune_on_current_support(fixed_supp_mask)
        self.betas[fixed_supp_mask] = betas_on_supp_tmp
    
    def get_sparse_sol_via_brute_force(self, k):
        """Get optimal sparse solution through brute force; only works for small p choose k

        Parameters
        ----------
        k : int
            number of nonzero coefficients for the final sparse solution
        """

        # enumerate all possible support sets
        best_loss = self.loss
        best_supp_str = self.fixed_supp_mask.tostring()

        total_num_combinations = scipy.special.comb(sum(self.allowed_supp_mask) - sum(self.fixed_supp_mask), k - sum(self.fixed_supp_mask), exact=True)
        total_supp_mask = np.zeros((total_num_combinations, self.p), dtype=bool)
        total_supp_mask[:, self.fixed_supp_mask] = True

        additional_fixed_supp_masks = combinations(list(np.where(np.logical_xor(self.allowed_supp_mask, self.fixed_supp_mask))[0]), k - sum(self.fixed_supp_mask))

        for i, additional_fixed_supp_mask in enumerate(additional_fixed_supp_masks):
            total_supp_mask[i, list(additional_fixed_supp_mask)] = True
            tmp_support_str = total_supp_mask[i].tostring()
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
        """Get sparse solution through beam search and orthogonal matching pursuit (OMP), for level i, each parent solution generates [child_size] child solutions, so there will be [parent_size] * [child_size] number of total child solutions. However, only the top [parent_size] child solutions are retained as parent solutions for the next level i+1.

        Parameters
        ----------
        k : int
            number of nonzero coefficients for the final sparse solution
        parent_size : int, optional
            how many top solutions to retain at each level, by default 10
        child_size : int, optional
            how many child solutions to generate based on each parent solution, by default 10
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

        Parameters
        ----------
        parent_size : int, optional
            how many top solutions to retain at each level, by default 10
        child_size : int, optional
            how many child solutions to generate based on each parent solution, by default 10
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

        Parameters
        ----------
        i : int
            index of the parent solution
        child_size : int, optional
            how many child solutions to generate based on parent solution i, by default 10
        """
        fixed_supp_mask = self.supp_mask_arr_parent[i]
        unfixed_and_allowed_mask = np.logical_xor(self.allowed_supp_mask, fixed_supp_mask)
        unfixed_and_allowed_indicies = np.where(unfixed_and_allowed_mask)[0]


        tmp_support_str = self.supp_mask_arr_parent[i].tostring()
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
            tmp_support_str = self.supp_mask_arr_child[child_id].tostring()
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
