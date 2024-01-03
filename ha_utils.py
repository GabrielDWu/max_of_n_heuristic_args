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