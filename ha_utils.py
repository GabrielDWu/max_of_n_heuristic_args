# Utils specific to a heuristic argument. CURRENTLY UNUSED

# A class for keeping track of a binned distribution
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np

class BinnedDist:

    def __init__(self, lb, ub, num_bins, bins, lst = None):
        self.lb = lb
        self.ub = ub
        self.num_bins = num_bins
        self.bins = bins
        self.bin_sz = (ub - lb) / self.num_bins
        self.lst = lst

    @staticmethod
    def from_list(lst, num_bins, lb=None, ub=None):
        lb = min(lst) if lb is None else lb
        ub = max(lst) if ub is None else ub
        bd = BinnedDist(lb, ub, num_bins, [0]*num_bins, lst)
        for x in lst:
            bd.bins[bd.get_bin(x)] += 1 / len(lst)
        return bd

    @staticmethod 
    def from_cdf(cdf, num_bins, lb, ub):
        bd = BinnedDist(lb, ub, num_bins, [0]*num_bins)
        for i in range(num_bins):
            bd.bins[i] += cdf(bd.bin_sz * (i+1)+lb) - cdf(bd.bin_sz * i + lb)
        return bd
    
    def get_bin(self, x):
        bin = int((x - self.lb) / self.bin_sz)
        if bin == self.num_bins: # this is ugly
            bin -= 1
        assert 0 <= bin < self.num_bins
        return bin
    
    def get_rep(self, i):
        """Returns representative (midpoint) of bin i"""
        assert 0 <= i < self.num_bins
        return self.lb + (i + .5) * self.bin_sz
    
    def plot(self, include_lst=False, **kwargs):
        plt.bar(np.linspace(self.lb, self.ub, self.num_bins), self.bins, width=self.bin_sz, **kwargs)
        if include_lst:
            assert self.lst is not None
            plt.hist(self.lst, bins=100, alpha=.5)

    def apply_func(self, func, preserve_gt=False):
        weighted_lst = defaultdict(float)
        for i, cnt in enumerate(self.bins):
            weighted_lst[func(self.get_rep(i))] += cnt
        
        lb = min(weighted_lst.keys())
        ub = max(weighted_lst.keys())
        bd = BinnedDist(lb, ub, self.num_bins, [0]*self.num_bins)
        for x, cnt in weighted_lst.items():
            bd.bins[bd.get_bin(x)] += cnt
        
        if preserve_gt:
            assert self.lst is not None
            bd.lst = list(map(func, self.lst))
        
        return bd
    
    def apply_func2(self, func, other_bd, preserve_gt=False):
        """Approximates the binned distribution of func(self.lst, other_bd.lst) in quadratic time"""
        weighted_lst = defaultdict(float)
        for i, cnt_i in enumerate(self.bins):
            for j, cnt_j in enumerate(other_bd.bins):
                if cnt_i * cnt_j > 0:
                    weighted_lst[func(self.get_rep(i), other_bd.get_rep(j))] += cnt_i * cnt_j

        lb = min(weighted_lst.keys())
        ub = max(weighted_lst.keys())
        bd = BinnedDist(lb, ub, self.num_bins, [0]*self.num_bins)
        for x, cnt in weighted_lst.items():
            bd.bins[bd.get_bin(x)] += cnt
        
        if preserve_gt:
            assert self.lst is not None
            bd.lst = list([func(x, y) for x in self.lst for y in other_bd.lst])
        
        return bd
    
    def convolve_with(self, other_bd, preserve_gt=False):
        return self.apply_func2(lambda x, y: x+y, other_bd, preserve_gt)

    def mean(self):
        return sum(self.get_rep(i) * cnt for i, cnt in enumerate(self.bins))



# Functions to estimate E[f(X)] using a Taylor approximation of f around X
import torch
from math import factorial
from scipy.special import comb
ell = 2 # The number of terms of the taylor expansion we're using

torch.set_grad_enabled(True)


def get_derivatives(func, num_derivatives, center):
    if type(center) != torch.Tensor:
        center = torch.tensor(center)
    assert center.shape == (), "center must be a scalar"
    center = center.clone().detach().requires_grad_(True)

    derivatives = [func(center)]
    for _ in range(1, num_derivatives + 1):
        # Compute the i-th derivative
        grad = torch.autograd.grad(derivatives[-1], center, create_graph=True)[0]
        derivatives.append(grad)

    return [d.item() for d in derivatives]

def expected_taylor(func, num_derivatives, moments, central=False, mean=None):
    """
    Moments is a list of the moments of X, where moments[i] = E[(X-E[X])^i] if central==True, else moments[i] = E[X^i]. If central==True, you should also specify the mean.
    Approximates func(x) as C_0 + C_1 (x-E[x]) + C_2 (x-E[x])^2 + ..., going up to C_{num_derivatives}. Then takes the expectation of this polynomial.
    """
    assert central == (mean is not None)
    assert num_derivatives > 0
    assert num_derivatives == len(moments)-1
    assert moments[0] == 1

    if mean is None:
        mean = moments[1]

    derivatives = get_derivatives(func, num_derivatives, mean)
    C = [derivatives[i] / factorial(i) for i in range(num_derivatives + 1)]

    if central:
        central_moments = moments
    else:
        # Compute central moments from moments
        # central_moment[i] = E[(X-E[X])^i]
        central_moments = [1]
        for i in range(1, num_derivatives+1):
            central_moments.append(0)
            for j in range(i + 1):
                central_moments[i] += comb(i, j) * (-moments[0])**j * moments[i+1-j]
    return sum(C[i] * central_moments[i] for i in range(num_derivatives+1))

