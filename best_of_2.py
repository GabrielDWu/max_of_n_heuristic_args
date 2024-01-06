# %%
from collections import defaultdict
from utils_cleaned import *
import torch as th
!wandb login --anonymously
model = get_model('2-32')
# model = get_model('2-1500')

torch.set_grad_enabled(False)
# %%
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

all_sequences = generate_all_sequences(model.cfg.d_vocab, model.cfg.n_ctx)

# Check `forward` correctly matches `model.forward`
assert th.allclose(model.forward(all_sequences), forward(model, all_sequences), atol=0.00005)

# Model's accuracy
probs = th.nn.functional.softmax(forward(model, all_sequences), dim=-1)[:, -1]
print("Accuracy:", float((probs.argmax(-1) == all_sequences.max(-1).values).float().mean()))
print("Soft Accuracy:", float(probs[th.arange(len(all_sequences)), all_sequences.max(-1).values].mean()))

# %% [markdown]
# Get a heuristic estimate for the (soft) accuracy of the model
SHOW = False
W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
W_Q, W_K= model.W_Q, model.W_K
BINS = 100

d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx

# %%
# Quantiled BinnedDist
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

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
    
    @staticmethod
    def mixture(bd1: 'BinnedDist', bd2: 'BinnedDist', p, num_bins=None):
        """Mixture of bd1 and bd2, with probability p of bd1 and 1-p of bd2"""
        weighted_lst = [(x, p / bd1.num_bins) for x in bd1.bins] + [(x, (1-p) / bd2.num_bins) for x in bd2.bins]
        return BinnedDist.from_weighted_list(weighted_lst, num_bins=num_bins if num_bins is not None else max(bd1.num_bins, bd2.num_bins))
    
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
    
    @staticmethod
    def convolve_n_times(dist, n):
        """Convolve with itself n times, O(log n) using 'square and multiply'"""
        assert n > 1
        to_convolve = []
        curr_dist = dist
        curr_power = 1
        while curr_power <= n:
            if curr_power & n:
                to_convolve.append(curr_dist)
            curr_dist += curr_dist
            curr_power <<= 1
        return sum(to_convolve[1:], start=to_convolve[0])

    def mean(self):
        return np.mean(self.bins)


# %% [markdown]
# Notice: SVD of QK has big first element and small rest

QK = (W_E + W_pos[-1]) @ W_Q[0, 0, :, :] @ W_K[0, 0, :, :].T @ (W_E).T
assert QK.shape == (d_vocab, d_vocab), f"QK.shape = {QK.shape} != {(d_vocab, d_vocab)} = (d_vocab, d_vocab)"

# take SVD
U, S, V = torch.svd(QK)
sign = torch.sign(U[:,0].mean())
U *= sign
V *= sign
torch.testing.assert_close(QK, U @ S.diag() @ V.T, atol=0.001, rtol=0.001)
torch.testing.assert_close(torch.eye(d_vocab), U @ U.T, atol=0.0001, rtol=0.0001)
torch.testing.assert_close(torch.eye(d_vocab), V @ V.T, atol=0.0001, rtol=0.0001)

key_direction, query_direction = V[:, 0], U[:, 0]
other_V_dist = BinnedDist.from_weighted_list(V[:, 1:].flatten(), BINS, equal_weight=True)
other_U_dist = BinnedDist.from_weighted_list(U[:, 1:].flatten(), BINS, equal_weight=True)

first_singular_value = S[0]
other_singular_values_dist = BinnedDist.from_list(S[1:])

# Contribution of the positional encoding of the key to the attention. Sign is + or - with equal probability
positional_adjustment_list = (W_E + W_pos[-1]) @ W_Q[0,0,:,:] @ W_K[0,0,:,:].T @ (W_pos[1] - W_pos[0]).unsqueeze(0).T
pos_adj_dist = BinnedDist.from_weighted_list(torch.stack([positional_adjustment_list, -positional_adjustment_list]).flatten(), BINS, equal_weight=True)

if SHOW:
    print(f"first_singular_value: {first_singular_value}")
    other_singular_values_dist.plot()
    plt.title("other_singular_values_dist")
    plt.show()
    pos_adj_dist.plot()
    plt.title("pos_adj_dist")
    plt.show()
    other_V_dist.plot()
    plt.title("other_V_dist")
    plt.show()
    other_U_dist.plot()
    plt.title("other_U_dist")
    plt.show()


# %% [markdown]

# %% [markdown]
### Calculate mean and std of difference between pre-softmax attention on $x_{max}$ and $x_{min}$.
# Notation:
# - $p_1, p_2 \in \mathbb{R}^{d_{model}}$ are the positional embeddings.
# - $x_1, x_2 \in d_{vocab}$ are the first and second numbers being inputted. $x_{min} = \min(x_1, x_2)$ and $x_{max} = \max(x_1, x_2)$.
# - $t_1, t_2 \in \mathbb{R}^{d_{model}}$ are one-hot encodigs of $x_1, x_2$. Same for $t_{min}, t_{max}$
# - $s_i$ is the $i$-th largest singular value, and $u_i, v_i \in \mathbb{R}^{d_{model}}$ are the $i$-th row/column of $U$ and $V$.
#
#For any choice of $x_1, x_2 \sim [d_{vocab}]$ (condition on $x_1 \neq x_2$), we have the pre-softmax attention attended from the second position to the $i$-th position is:
# $$t_2 \cdot (E + p_2) \cdot Q \cdot K^T \cdot (E^T t_i^T + p_i^T) \\= t_2 \cdot U S V^T \cdot t_i^T + t_2 (E+p_2)\cdot Q \cdot K^T \cdot p_i^T$$
# This implies that the difference between the pre-softmax attention to the larger and smaller token streams is
# $$t_2 \cdot USV^T \cdot (t_{max} - t_{min})^T + (-1)^{\mathbb{I}_{x_1 > x_2}} t_2 \cdot \mathrm{pos\_adjustment}$$
# $$=\sum_{i = 1}^{d_{model}} s_i (t_2 \cdot u_i) ((t_{max} - t_{min}) \cdot v_i) + (-1)^{\mathbb{I}_{x_1 > x_2}} \mathrm{pos\_adjustment}_{x_2}$$

# first_component_lst = [[] for _ in range(d_vocab)]
first_component_lst = []
for x1 in range(d_vocab):
    for x2 in range(d_vocab):
        if x1 == x2:
            continue
        x_min, x_max = min(x1, x2), max(x1, x2)
        # first_component_lst[x_max].append(float(first_singular_value * query_direction[x2] * (key_direction[x_max] - key_direction[x_min])))
        first_component_lst.append(float(first_singular_value * query_direction[x2] * (key_direction[x_max] - key_direction[x_min])))
# first_component_lst[0].append(-22) # This shouldn't matter; just need it to be non-empty
# first_component_dist = [BinnedDist.from_weighted_list(x, BINS, equal_weight=True) for x in first_component_lst]
first_component_dist = BinnedDist.from_weighted_list(first_component_lst, BINS, equal_weight=True)

other_component_dist = other_singular_values_dist * other_U_dist * (other_V_dist - other_V_dist)
convolved_other_component_dist = BinnedDist.convolve_n_times(other_component_dist, d_vocab - 1)

# presoftmax_attn_diff_dist = [x + convolved_other_component_dist + pos_adj_dist for x in first_component_dist]
presoftmax_attn_diff_dist = first_component_dist + convolved_other_component_dist + pos_adj_dist
def sigmoid_neg_x(x):
    """sigmoid(-x)"""
    return 1 / (1 + np.exp(-x / W_K.shape[-1] ** .5))

# a_dist = [BinnedDist.mixture(BinnedDist.constant(1), x.apply_func(sigmoid_neg_x), 1/(2*i+1)) for i,x in enumerate(presoftmax_attn_diff_dist)]
a_dist = BinnedDist.mixture(BinnedDist.constant(1), presoftmax_attn_diff_dist.apply_func(sigmoid_neg_x), 1/d_model) # Mixture accounts for the fact that 1/d_model of the time, x_1 = x_2.


# %%
from functools import partial
def sequences_with_given_max(x_max):
    """Returns all sequences with max = x_max"""
    return torch.tensor([[x1, x2] for x1 in range(d_vocab) for x2 in range(d_vocab) if max(x1, x2) == x_max])

def get_average_attn_on_correct(s, attn_prob):
    if s == "attn_prob":
        sm = []
        for i in range(len(sequences)):
            if sequences[i][0] == sequences[i][1]:
                sm.append(1)
            else:
                sm.append(float(attn_prob[i, 0, 1, 0] if sequences[i, 0] > sequences[i, 1] else attn_prob[i, 0, 1, 1]))
        b = BinnedDist.from_weighted_list(sm, BINS, equal_weight=True)
        plt.plot(b.bins, label="true_dist")
        print(b.mean())
# forward(model, all_sequences, hook_fn=get_average_attn_on_correct)
# max_val = 50
# sequences = sequences_with_given_max(max_val)
# forward(model, sequences, hook_fn=get_average_attn_on_correct)
# plt.plot(a_dist[max_val].bins, label="estimated_dist")
sequences = all_sequences
forward(model, sequences, hook_fn=get_average_attn_on_correct)
plt.plot(a_dist.bins, label="estimated_dist")
plt.legend()
plt.title("Attention on correct token")

# %% [markdown]
# Notice: EVOU is basically identity

EVOU = W_E @ W_V @ W_O @ W_U
EVOU.squeeze_()


on_diagonal = EVOU.diag()
off_diagonal = th.tensor([EVOU[i, j] for j in range(EVOU.shape[1]) for i in range(EVOU.shape[0]) if i != j])

on_diag_mean = on_diagonal.mean()
on_diagonal_std = on_diagonal.std()
off_diag_mean = off_diagonal.mean()
off_diagonal_std = off_diagonal.std()
evou_mean = EVOU.mean()
evou_std = EVOU.std()

if SHOW:
    plt.imshow(EVOU.detach().cpu().numpy())
    plt.colorbar()
    plt.title("EVOU")
    plt.show()
    print(f"on_diag_mean: {on_diag_mean}\non_diagonal_std: {on_diagonal_std}\noff_diag_mean: {off_diag_mean}\noff_diagonal_std: {off_diagonal_std}")


# %% [markdown]
# Analytically-calculated probabilities of (i, j, k)
# i is max, j is min, k is something that is not i
def prob_ijk(i = None, j = None, k = None):
    """
    returns probability of (i, j, k) under the following distribution:
    x1, x2 ~ [d_vocab]
    i = max(x1, x2)
    j = min(x1, x2)
    k ~ [d_vocab] \ {i}
    """
    assert i is None or (i >= 0 and i < d_vocab)
    assert j is None or (j >= 0 and j < d_vocab)
    assert k is None or (k >= 0 and k < d_vocab)
    assert i is not None or j is not None or k is not None

    non_none = ("i" if i is not None else "") + ("j" if j is not None else "") + ("k" if k is not None else "")

    if non_none == "i":
        return (2*i + 1) / d_vocab**2
    elif non_none == "ij":
        if j > i: return 0
        if i == j: return 1 / d_vocab**2
        return 2 / d_vocab**2
    elif non_none == "j":
        return (2*(d_vocab - 1 - j) + 1) / d_vocab**2
    elif non_none == "k":
        return (d_vocab**2 - (2*k+1)) / d_vocab**2 / (d_vocab - 1)
    elif non_none == "jk":
        raise NotImplementedError # seems annoying; i'll do this later
    elif non_none in ["ik", "ijk"]:
        if i == k: return 0
        return prob_ijk(i, j, None) / (d_vocab - 1)


# %% [markdown]
# Make necessary binned distributions

weighted_lsts = {}

# pe2Ui: The direct contribution of positional encoding to the correct logit (doesn't go through attention)
weighted_lsts["P2Ui"] = [(float(x), prob_ijk(i=i)) for i, x in enumerate(W_pos[1] @ W_U)]

# pe2Uk: The direct contribution of positional encoding to an incorrect logit (doesn't go through attention)
weighted_lsts["P2Uk"] = [(float(x), prob_ijk(k=k)) for k, x in enumerate(W_pos[1] @ W_U)]

EU = W_E @ W_U
# EUx2i: The contribution of the embedding of x2 to the correct logit (doesn't go through attention)
weighted_lsts["EUx2i"] = []
for x1 in range(d_vocab):
    for x2 in range(d_vocab):
        weighted_lsts["EUx2i"].append((float(EU[x2, max(x1, x2)]), 1/d_vocab**2))

# EUx2k: The contribution of the embedding of x2 to an incorrect logit (doesn't go through attention)
weighted_lsts["EUx2k"] = []
for x2 in range(d_vocab):
    for k in range(d_vocab):
        weighted_lsts["EUx2k"].append((float(EU[x2, k]), 1/d_vocab**2))    # This isn't exact; actually x2 is correlated with k

# big_attention: The contribution of the attention to x_max to an incorrect logit - the correct logit. Before being multiplied by A
weighted_lsts["big_attention"] = []

for loc_max in [0, 1]:
    Ploc_maxVOU = (W_pos[loc_max] @ W_V @ W_O @ W_U).squeeze()
    for i in range(d_vocab):
        for k in range(d_vocab):
            weighted_lsts["big_attention"].append((float(EVOU[i,k] + Ploc_maxVOU[k] - (EVOU[i,i] + Ploc_maxVOU[i])), prob_ijk(i=i, k=k) / 2))

# small_attention: The contribution of the attention to x_min to an incorrect logit - the correct logit. Before being multiplied by (1-A)
weighted_lsts["small_attention"] = []

# This takes O(n^3)
for loc_min in [0, 1]:
    Ploc_maxVOU = (W_pos[loc_max] @ W_V @ W_O @ W_U).squeeze()
    for j in range(d_vocab):
        for i in range(d_vocab):
            for k in range(d_vocab):
                weighted_lsts["small_attention"].append(
                    (float(EVOU[j,k] + Ploc_maxVOU[k] - (EVOU[j,i] + Ploc_maxVOU[i])), prob_ijk(i=i, j=j, k=k) / 2))

# %%
dists = {}
for key, value in weighted_lsts.items():
    dists[key] = BinnedDist.from_weighted_list(value, BINS)

# %% [markdown]
# Estimate the final output distribution
import scipy.stats as stats

correct_logit_dist = dists["EUx2i"] + dists["P2Ui"]
incorrect_logit_dist = dists["big_attention"].apply_func2(lambda x, a: x*a, a_dist) + dists["small_attention"].apply_func2(lambda x, a: x*(1-a), a_dist) + dists["EUx2k"] + dists["P2Uk"]

exp_correct_logit_dist = correct_logit_dist.apply_func(np.exp)
exp_incorrect_logit_dist = incorrect_logit_dist.apply_func(np.exp)
convolved_exp_incorrect_logit_dist = BinnedDist.convolve_n_times(exp_incorrect_logit_dist, d_vocab-1)

final_output_dist = exp_correct_logit_dist.apply_func2(lambda a, b: a/(a+b), convolved_exp_incorrect_logit_dist)

if SHOW:
    final_output_dist.plot()
    plt.title("Estimated Final output distribution")
    plt.show()
    print(f"estimated accuracy: {final_output_dist.mean()}")

    # Model's accuracy
    probs = th.nn.functional.softmax(forward(model, all_sequences), dim=-1)[:, -1]
    problst = probs[th.arange(len(all_sequences)), all_sequences.max(-1).values].tolist()
    true_dist = BinnedDist.from_weighted_list(problst, BINS, equal_weight=True)
    true_dist.plot()

# %%
plt.plot(final_output_dist.bins, label="estimated")
plt.plot(true_dist.bins, label="true")
plt.legend()
plt.title("Estimated vs true final soft accuracy distribution (inverse CDF)")
# %%
# Below is scratch to debug error between true and estimate

def forward_attn_1(model, input, *, hook_fn=None):
    """Assume attention is always correctly placed 100% on correct token"""
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
    attn_1 = torch.zeros_like(Q @ K.transpose(-2,-1))
    # places 1 attn on correct answer
    attn_1[:, 0, 1, 0] = all_sequences[:, 0] > all_sequences[:, 1]
    attn_1[:, 0, 1, 1] = all_sequences[:, 0] <= all_sequences[:, 1]

    # attn = masked_softmax(Q @ K.transpose(-2, -1) / d_k**0.5) @ V
    attn = attn_1 @ V
    attn = hook("attn_z", attn)
    attn = (attn @ model.W_O[None, 0]).sum(-3) + model.b_O[None, 0].unsqueeze(-2)
    attn = hook("attn_result", attn)
    x = x + attn

    x = hook("attn_block_post", x)
    x = x @ model.W_U + model.b_U[None, None]

    return x

# Model's accuracy
probs = th.nn.functional.softmax(forward_attn_1(model, all_sequences), dim=-1)[:, -1]
print("Accuracy:", float((probs.argmax(-1) == all_sequences.max(-1).values).float().mean()))
print("Soft Accuracy:", float(probs[th.arange(len(all_sequences)), all_sequences.max(-1).values].mean()))
# %%
from random import randint
def exact_tree(x1, x2):
    """Should emulate exactly what the model does"""
    # x1, x2 = randint(0, d_vocab-1), randint(0, d_vocab-1)
    correct_loc = 1 if x2 >= x1 else 0
    i, j = max(x1, x2), min(x1, x2)
    E, V, O, U = W_E.squeeze(), W_V.squeeze(), W_O.squeeze(), W_U.squeeze()
    correct_logit = EVOU[i,i] + (W_pos[correct_loc] @ V @ O @ U)[i] + (E @ U)[x2, i] + (W_pos[1] @ U)[i]
    incorrect_logits = [EVOU[i,k] + (W_pos[correct_loc] @ V @ O @ U)[k] + (E @ U)[x2, k] + (W_pos[1] @ U)[k] for k in range(d_vocab) if k != i]
    return np.exp(correct_logit) / (np.exp(correct_logit) + sum(np.exp(incorrect_logits)))
sum([exact_tree(x1, x2) for x1, x2 in all_sequences]) / len(all_sequences) #Yup, this gets 94% as expected.
# %%

def no_pos(x1, x2):
    """no positional bias or direct cotribution"""
    # x1, x2 = randint(0, d_vocab-1), randint(0, d_vocab-1)
    correct_loc = 1 if x2 >= x1 else 0
    i, j = max(x1, x2), min(x1, x2)
    E, V, O, U = W_E.squeeze(), W_V.squeeze(), W_O.squeeze(), W_U.squeeze()
    correct_logit = EVOU[i,i] 
    incorrect_logits = [EVOU[i,k] for k in range(d_vocab) if k != i]
    return np.exp(correct_logit) / (np.exp(correct_logit) + sum(np.exp(incorrect_logits)))
sum([no_pos(x1, x2) for x1, x2 in all_sequences]) / len(all_sequences) #Yup, gets 91%, so not the main source of error

# %%
def no_pos_independent_k():
    x1, x2 = randint(0, d_vocab-1), randint(0, d_vocab-1)
    i, j = max(x1, x2), min(x1, x2)
    correct_logit = EVOU[i,i] 
    incorrect_logits = []
    for _ in range(d_vocab-1):
        k = randint(0, d_vocab-1)
        while k == i:
            k = randint(0, d_vocab-1)
        incorrect_logits.append(EVOU[i,k])
    return np.exp(correct_logit) / (np.exp(correct_logit) + sum(np.exp(incorrect_logits)))
trials = 10000
sum([no_pos_independent_k() for _ in range(trials)]) / trials # also 92%, so not a source of error.
# %%
def no_pos_independent_k_from_i_from_dist():
    x1, x2 = randint(0, d_vocab-1), randint(0, d_vocab-1)
    i, j = max(x1, x2), min(x1, x2)
    correct_logit = 0
    incorrect_logits = []
    for _ in range(d_vocab-1):
        ip = max(randint(0, d_vocab-1), randint(0, d_vocab-1))
        k = (randint(1, d_vocab-1) + ip)%d_vocab
        incorrect_logits.append(EVOU[ip,k] - EVOU[ip, ip])
    return np.exp(correct_logit) / (np.exp(correct_logit) + sum(np.exp(incorrect_logits)))

sum([no_pos_independent_k_from_i_from_dist() for _ in range(trials)]) / trials #85%
# %%
def no_pos_independent_k_from_i():
    x1, x2 = randint(0, d_vocab-1), randint(0, d_vocab-1)
    i, j = max(x1, x2), min(x1, x2)
    correct_logit = EVOU[i,i] 
    incorrect_logits = []
    for _ in range(d_vocab-1):
        k = randint(0, d_vocab-1)
        ip = (randint(1, d_vocab-1) + k)%d_vocab
        incorrect_logits.append(EVOU[ip,k])
    return np.exp(correct_logit) / (np.exp(correct_logit) + sum(np.exp(incorrect_logits)))

sum([no_pos_independent_k_from_i() for _ in range(trials)]) / trials # 84%
# %%
def no_pos_independent_k_subtract_i():
    x1, x2 = randint(0, d_vocab-1), randint(0, d_vocab-1)
    correct_logit = 0
    incorrect_logits = []
    for _ in range(d_vocab-1):
        k = randint(0, d_vocab-1)
        ip = (randint(1, d_vocab-1) + k)%d_vocab
        incorrect_logits.append(EVOU[ip,k] - EVOU[ip, ip])
    return np.exp(correct_logit) / (np.exp(correct_logit) + sum(np.exp(incorrect_logits)))

sum([no_pos_independent_k_subtract_i() for _ in range(trials)]) / trials # 48%
# %%
def no_pos_independent_k_subtract_i_weight_ip():
    x1, x2 = randint(0, d_vocab-1), randint(0, d_vocab-1)
    correct_logit = 0
    incorrect_logits = []
    for _ in range(d_vocab-1):
        ip = max(randint(0, d_vocab-1), randint(0, d_vocab-1))
        k = (randint(1, d_vocab-1) + ip)%d_vocab
        incorrect_logits.append(EVOU[ip,k] - EVOU[ip, ip])
    return np.exp(correct_logit) / (np.exp(correct_logit) + sum(np.exp(incorrect_logits)))

sum([no_pos_independent_k_subtract_i_weight_ip() for _ in range(trials)]) / trials # 85%
# %%
def no_pos_independent_k_uniform_i():
    i = randint(0, d_vocab-1)
    correct_logit = EVOU[i,i] 
    incorrect_logits = []
    for _ in range(d_vocab-1):
        k = randint(0, d_vocab-1)
        ip = (randint(1, d_vocab-1) + k)%d_vocab
        incorrect_logits.append(EVOU[ip,k])
    return np.exp(correct_logit) / (np.exp(correct_logit) + sum(np.exp(incorrect_logits)))
trials = 10000
sum([no_pos_independent_k_uniform_i() for _ in range(trials)]) / trials #74%
# %%