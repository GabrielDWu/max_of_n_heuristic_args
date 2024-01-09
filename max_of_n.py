# %%
from collections import defaultdict
from utils_cleaned import *
import torch as th
!wandb login --anonymously
model = get_model('10-15000')
# model = get_model('5-9950')
# model = get_model('2-32')

torch.set_grad_enabled(False)
# %%
# Perturb all parameters by a small amount

def perturb(model, eps=1e-3):
    for name, param in model.named_parameters():
        if "b_" in name:
            continue
        param += eps * th.randn_like(param)

def masked_softmax(logits):
    n_ctx = logits.shape[-1]
    mask = th.tril(th.ones(n_ctx, n_ctx), 0)
    mask = mask[(None,) * (logits.ndim - mask.ndim) + (Ellipsis,)]
    mask = mask.to(logits.device)
    logits = logits - logits.max(dim=-1, keepdim=True)[0]
    probs = th.exp(logits) * mask
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs

def forward(model, input, *, hook_fn=None):
    def hook(key, x):
        if hook_fn is None:
            return x
        new_x = hook_fn(key, x)
        return x if new_x is None else new_x

    if th.is_floating_point(input):
        x = input[(None,) * (3 - input.ndim)]
    else:
        x = input[(None,) * (2 - input.ndim)]
        x = th.nn.functional.one_hot(x, num_classes=model.cfg.d_vocab).float()

    x = x.to(model.W_E.device)
    x = x @ model.W_E
    x = hook("embed", x)
    x = x + model.W_pos[None, : x.shape[-2]]
    x = hook("attn_block_pre", x)

    Q = x[:, None] @ model.W_Q[None, 0] + model.b_Q[None, 0].unsqueeze(-2)
    K = x[:, None] @ model.W_K[None, 0] + model.b_K[None, 0].unsqueeze(-2)
    V = x[:, None] @ model.W_V[None, 0] + model.b_V[None, 0].unsqueeze(-2)
    Q = hook("Q", Q)
    K = hook("K", K)
    V = hook("V", V)

    d_k = model.W_K.shape[-1]
    attn_prob = masked_softmax(Q @ K.transpose(-2, -1) / d_k**0.5)
    attn_prob = hook("attn_prob", attn_prob)
    attn = attn_prob @ V
    attn = hook("attn_z", attn)
    attn = (attn @ model.W_O[None, 0]).sum(-3) + model.b_O[None, 0].unsqueeze(-2)
    attn = hook("attn_result", attn)
    x = x + attn

    x = hook("attn_block_post", x)
    x = x @ model.W_U + model.b_U[None, None]

    return x
# %%

sequences = generate_some_sequences(model.cfg.d_vocab, model.cfg.n_ctx, unique=True, cnt=100000)

# Check `forward` correctly matches `model.forward`
assert th.allclose(model.forward(sequences), forward(model, sequences), atol=0.001)

# Model's accuracy
probs = th.nn.functional.softmax(forward(model, sequences), dim=-1)[:, -1]
print("Accuracy:", float((probs.argmax(-1) == sequences.max(-1).values).float().mean()))
print("Soft Accuracy:", float(probs[th.arange(len(sequences)), sequences.max(-1).values].mean()))

# %% [markdown]
# Get a heuristic estimate for the (soft) accuracy of the model
SHOW = False
BINS = 10


# %%
# Quantiled BinnedDist
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.special import comb

class BinnedDist:

    def __init__(self, num_bins, bins=None):
        """
        If num_bins = 10, then bins is an array of 10 numbers. The first number is the mean of the bottom 10% of the distribution, etc.
        """
        self.num_bins = num_bins
        self.bins = np.array(bins if bins is not None else [0]*num_bins)

    @staticmethod
    def from_list(lst):
        assert len(lst) > 0
        num_bins = len(lst)
        lst = np.sort(lst)
        bins = np.array(lst)
        return BinnedDist(num_bins, bins)

    @staticmethod
    def from_weighted_list(lst, num_bins, equal_weight=False):
        """lst is a bunch of pairs (x, w) where x is a number and w is its weight.
        when equal_weight=True, lst is just the x values"""
        assert len(lst) > 0
        if equal_weight:
            lst = [(x, 1/len(lst)) for x in lst]
        else:
            # normalize weights
            total_weight = sum(w for _, w in lst)
            for i in range(len(lst)):
                lst[i] = (lst[i][0], lst[i][1] / total_weight)

        lst = sorted(lst)
        bins = np.zeros(num_bins)
        ind = 0
        curr_moment = 0
        curr_weight = 0
        for i in range(num_bins):
            while curr_weight < 1/num_bins:
                if ind >= len(lst):
                    assert abs(1/num_bins - curr_weight) < 1e-6, f"curr_weight = {curr_weight} != {1/num_bins} = 1/num_bins"
                    break
                if lst[ind][1] + curr_weight < 1/num_bins:
                    curr_moment += lst[ind][0] * lst[ind][1]
                    curr_weight += lst[ind][1]
                    ind += 1
                else:
                    curr_moment += lst[ind][0] * (1/num_bins - curr_weight)
                    lst[ind] = (lst[ind][0], lst[ind][1] - (1/num_bins - curr_weight))
                    curr_weight = 1/num_bins
                    break
            bins[i] = curr_moment * num_bins
            curr_moment = 0
            curr_weight = 0
        return BinnedDist(num_bins, bins)


    @staticmethod 
    def from_truncated_expectation(conditional_expectation, num_bins):
        bins = np.array([conditional_expectation(i / num_bins, (i+1) / num_bins) for i in range(num_bins)])
        return BinnedDist(num_bins, bins)
    
    @staticmethod
    def from_normal(mu, sigma, num_bins):
        norm_cond = lambda a, b: mu + sigma * (norm.pdf(norm.ppf(a)) - norm.pdf(norm.ppf(b))) / (b - a)
        return BinnedDist.from_truncated_expectation(norm_cond, num_bins)
    
    @staticmethod
    def constant(c):
        return BinnedDist(1, np.array([c]))
    
    # @staticmethod
    # def mixture(bd1: 'BinnedDist', bd2: 'BinnedDist', p, num_bins=None):
    #     """Mixture of bd1 and bd2, with probability p of bd1 and 1-p of bd2"""
    #     weighted_lst = [(x, p / bd1.num_bins) for x in bd1.bins] + [(x, (1-p) / bd2.num_bins) for x in bd2.bins]
    #     return BinnedDist.from_weighted_list(weighted_lst, num_bins=num_bins if num_bins is not None else max(bd1.num_bins, bd2.num_bins))
    
    @staticmethod
    def mixture(dists: list[tuple['BinnedDist', float]], num_bins=None):
        """Mixture of dists[i][0] with probability dists[i][1]"""
        weighted_lst = []
        for dist, p in dists:
            weighted_lst += [(x, p / dist.num_bins) for x in dist.bins]
        return BinnedDist.from_weighted_list(weighted_lst, num_bins=num_bins if num_bins is not None else max(dist.num_bins for dist, _ in dists))
    
    def plot(self, **kwargs):
        xs = self.bins
        ys = np.zeros(self.num_bins)
        ys[1:-1] = 2 / (self.bins[2:] - self.bins[:-2]) / self.num_bins
        plt.plot(xs, ys, **kwargs)

    def apply_func(self, func):
        new_bins = np.sort(func(self.bins))
        return BinnedDist(self.num_bins, new_bins)
    
    def apply_func2(self, func, other_bd: 'BinnedDist', new_bin_cnt=None):
        x = self.bins[:, np.newaxis]
        y = other_bd.bins
        all_pairs = func(x, y)
        all_pairs_flat = all_pairs.flatten()
        bin_count = new_bin_cnt if new_bin_cnt is not None else min(len(all_pairs_flat), BINS)
        return BinnedDist.from_weighted_list(all_pairs_flat, bin_count, equal_weight=True)

    def __add__(self, other):
        return self.apply_func2(np.add, other)

    def __mul__(self, other):
        return self.apply_func2(np.multiply, other)
    
    def __sub__(self, other):
        return self.apply_func2(np.subtract, other)
    
    def __neg__(self):
        return self.apply_func(np.negative)
    
    def convolve_n_times(self, n):
        """Convolve with itself n times, O(log n) using 'square and multiply'"""
        if n == 0: return BinnedDist.constant(0)
        if n == 1: return self
        to_convolve = []
        curr_dist = self
        curr_power = 1
        while curr_power <= n:
            if curr_power & n:
                to_convolve.append(curr_dist)
            curr_dist += curr_dist
            curr_power <<= 1
        return sum(to_convolve[1:], start=to_convolve[0])

    def mean(self):
        return np.mean(self.bins)


def prob_x_max(x_max):
    """Returns probability of max = x_max"""
    return comb(x_max, n_ctx-1) / comb(d_vocab, n_ctx)

# %% [markdown]
# Õ(n^3) estimate

def estimate_slow(model):

    W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
    W_Q, W_K= model.W_Q, model.W_K
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx

    accuracies = []

    EQKE = (W_E + W_pos[-1]) @ W_Q[0, 0, :, :] @ W_K[0, 0, :, :].T @ (W_E).T
    EQKP = (W_E + W_pos[-1]) @ W_Q[0, 0, :, :] @ W_K[0, 0, :, :].T @ (W_pos).T
    EVOU = W_E @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U
    PVOU = W_pos @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U
    EU = W_E @ W_U
    PU = W_pos @ W_U

    pos_contrib_through_vou = [BinnedDist.from_list(PVOU[:, ell]) for ell in range(d_vocab)]
    lower_contrib = [[BinnedDist.from_list(EVOU[:x_max, ell]) if x_max > 0 else None for ell in range(d_vocab)] for x_max in range(d_vocab)]

    for x_last in range(d_vocab):

        pos_noise_in_attn = BinnedDist.from_list(EQKP[x_last, :])

        for x_max in range(n_ctx-1, d_vocab):

            presoftmax_good_attn = BinnedDist.constant(EQKE[x_last, x_max]) + pos_noise_in_attn
            presoftmax_bad_attn = BinnedDist.from_list(EQKE[x_last, :x_max]) + pos_noise_in_attn
            exp_presoftmax_good_attn = presoftmax_good_attn.apply_func(np.exp)
            exp_presoftmax_bad_attn = presoftmax_bad_attn.apply_func(np.exp)
            good_attn = exp_presoftmax_good_attn.apply_func2(lambda a, b: a/(a+b), exp_presoftmax_bad_attn.convolve_n_times(n_ctx-1))
            bad_attn = exp_presoftmax_bad_attn.apply_func2(lambda a, b: a/(a+b), exp_presoftmax_good_attn+exp_presoftmax_bad_attn.convolve_n_times(n_ctx-2))

            logits = [None] * d_vocab
            exp_logits = [None] * d_vocab
            for ell in range(d_vocab):
                logits[ell] = good_attn * (BinnedDist.constant(EVOU[x_max, ell]) + pos_contrib_through_vou[ell]) + (bad_attn * (lower_contrib[x_max][ell] + pos_contrib_through_vou[ell])).convolve_n_times(n_ctx-1) + BinnedDist.constant(EU[x_last, ell] + PU[n_ctx-1, ell])
                exp_logits[ell] = logits[ell].apply_func(np.exp)
            
            acc = exp_logits[x_max].apply_func2(lambda a, b: a/(a+b), sum(exp_logits[:x_max] + exp_logits[x_max+1:], start=BinnedDist.constant(0)))

            accuracies.append((acc, prob_x_max(x_max)))

    overall_acc = BinnedDist.mixture(accuracies)
    return overall_acc.mean()
# %% [markdown]
# Õ(n^2) + O(n^3) estimate; joining on x_last

def estimate(model):

    W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
    W_Q, W_K= model.W_Q, model.W_K
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx

    accuracies = []

    EQKE = (W_E + W_pos[-1]) @ W_Q[0, 0, :, :] @ W_K[0, 0, :, :].T @ (W_E).T
    EQKP = (W_E + W_pos[-1]) @ W_Q[0, 0, :, :] @ W_K[0, 0, :, :].T @ (W_pos).T
    EVOU = W_E @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U
    PVOU = W_pos @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U
    EU = W_E @ W_U
    PU = W_pos @ W_U

    pos_contrib_through_vou = [BinnedDist.from_list(PVOU[:, ell]) for ell in range(d_vocab)]
    lower_contrib = [[BinnedDist.from_list(EVOU[:x_max, ell]) if x_max > 0 else None for ell in range(d_vocab)] for x_max in range(d_vocab)]
    resid_stream_contrib = [BinnedDist.from_list(EU[:, ell]) for ell in range(d_vocab)]

    good_attn = [[] for i in range(d_vocab)]
    bad_attn = [[] for i in range(d_vocab)]

    for x_last in range(d_vocab):

        pos_noise_in_attn = BinnedDist.from_list(EQKP[x_last, :])

        for x_max in range(n_ctx-1, d_vocab):

            presoftmax_good_attn = BinnedDist.constant(EQKE[x_last, x_max]) + pos_noise_in_attn
            presoftmax_bad_attn = BinnedDist.from_list(EQKE[x_last, :x_max]) + pos_noise_in_attn
            exp_presoftmax_good_attn = presoftmax_good_attn.apply_func(np.exp)
            exp_presoftmax_bad_attn = presoftmax_bad_attn.apply_func(np.exp)
            good_attn[x_max].append(exp_presoftmax_good_attn.apply_func2(lambda a, b: a/(a+b), exp_presoftmax_bad_attn.convolve_n_times(n_ctx-1)))
            bad_attn[x_max].append(exp_presoftmax_bad_attn.apply_func2(lambda a, b: a/(a+b), exp_presoftmax_good_attn+exp_presoftmax_bad_attn.convolve_n_times(n_ctx-2)))

    for x_max in range(n_ctx-1, d_vocab):

        logits = [None] * d_vocab
        exp_logits = [None] * d_vocab
        good_attn_marginalized = BinnedDist.mixture([(good_attn[x_max][x_last], 1 / d_vocab) for x_last in range(d_vocab)])
        bad_attn_marginalized = BinnedDist.mixture([(bad_attn[x_max][x_last], 1 / d_vocab) for x_last in range(d_vocab)])
        for ell in range(d_vocab):
            logits[ell] = good_attn_marginalized * (BinnedDist.constant(EVOU[x_max, ell]) + pos_contrib_through_vou[ell]) + (bad_attn_marginalized * (lower_contrib[x_max][ell] + pos_contrib_through_vou[ell])).convolve_n_times(n_ctx-1) + resid_stream_contrib[ell] + BinnedDist.constant(PU[n_ctx-1, ell])
            exp_logits[ell] = logits[ell].apply_func(np.exp)
        
        acc = exp_logits[x_max].apply_func2(lambda a, b: a/(a+b), sum(exp_logits[:x_max] + exp_logits[x_max+1:], start=BinnedDist.constant(0)))

        accuracies.append((acc, prob_x_max(x_max)))

    overall_acc = BinnedDist.mixture(accuracies)

    return overall_acc.mean()

# %%
TRIALS = 80
for i in range(TRIALS):
    # make a copy of model
    model = get_model('10-15000')
    torch.seed()
    perturb(model, .1)

    print(f"Trial {i+1}:")
    est = estimate(model)
    probs = th.nn.functional.softmax(forward(model, sequences), dim=-1)[:, -1]
    gt = float(probs[th.arange(len(sequences)), sequences.max(-1).values].mean())

    print("Estimate: ", est)
    print("Actual accuracy: ", gt)
    data.append((gt, est))

# %%
# plot data
plt.plot([x for x, _ in data], [y for _, y in data], 'o')
# plot line of best fit
m, b = np.polyfit([x for x, _ in data], [y for _, y in data], 1)
plt.plot([x for x, _ in data], [m*x + b for x, _ in data])
plt.xlabel("Actual accuracy")
plt.ylabel("Estimated accuracy")
plt.show()


# %%
sum((x-y) for x, y in data) / len(data)