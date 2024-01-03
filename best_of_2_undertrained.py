# %%
from typing import Any
from best_of_2_undertrained_utils import *
from ha_utils import *

# %%
def gabe_get_model():
    simpler_cfg = HookedTransformerConfig(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        n_ctx=N_CTX,
        d_vocab=D_VOCAB,
        seed=SEED,
        device=DEVICE,
        attn_only=True,
        normalization_type=None,
    )

    model = HookedTransformer(simpler_cfg).to(DEVICE)

    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    model_name = f'neural-net-coq-interp-max-{model.cfg.n_ctx}-epochs-{N_EPOCHS}'
    wandb_model_path = f"{DEFAULT_WANDB_ENTITY}/{model_name}/{model_name}:latest"
    model_dir = None
    try:
        api = wandb.Api()
        model_at = api.artifact(wandb_model_path)
        model_dir = Path(model_at.download())
    except Exception as e:
        print(f'Could not load model {wandb_model_path} from wandb:\n', e)
    if model_dir is not None:
        for model_path in model_dir.glob('*.pth'):
            res = load_model(model, model_path)
            if res is not None:
               return model

# %%
m = gabe_get_model()
# %%
def train_or_load_model(
      model_name:str,
      model:HookedTransformer,
      loss_fn,
      acc_fn,
      train_data_gen_maybe_lambda,
      data_test,
      n_epochs=100,
      batches_per_epoch=10,
      device=None,
      wandb_project=None,
      save_model=True,
      model_pth_path=None,
      deterministic: bool = False,
      optimizer=torch.optim.Adam,
      optimizer_kwargs={'lr':1e-3, 'betas': (.9, .999)},
      train_data_gen_is_lambda: bool = False,
      loss_fn_kwargs={'return_per_token':True},
      print_every: Optional[int] = 10,
      log_acc: bool = False,
      force_train: bool = False,
      overwrite_data: bool = False,
      model_description: str = "trained model",
      wandb_entity:str = DEFAULT_WANDB_ENTITY,
      fail_if_cant_load: bool = False,
      save_in_google_drive: bool = False,
      **kwargs, # kwargs for **locals() below
  ):
#   if force_train and fail_if_cant_load: raise ValueError(f"force_train is {force_train} and fail_if_cant_load is {fail_if_cant_load}")
#   if device is None: device = default_device(deterministic=deterministic)

  pth_base_path = get_pth_base_path(save_in_google_drive=save_in_google_drive, create=True)
  if model_pth_path is None:
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_pth_path = pth_base_path / f'{model_name}-{model.cfg.n_ctx}-epochs-{n_epochs}-{datetime_str}.pth'

  if not force_train and os.path.exists(model_pth_path):
    res = load_model(model, model_pth_path)
    if res is not None: return res

  if wandb_project is not None:
    wandb_model_path = f"{wandb_entity}/{wandb_project}/{model_name}:latest"
    if not force_train:
      model_dir = None
      try:
        api = wandb.Api()
        model_at = api.artifact(wandb_model_path)
        model_dir = Path(model_at.download())
      except Exception as e:
        print(f'Could not load model {wandb_model_path} from wandb:\n', e)
      if model_dir is not None:
        for model_path in model_dir.glob('*.pth'):
          res = load_model(model, model_path)
          if res is not None: return res

  assert not fail_if_cant_load, f"Couldn't load model from {model_pth_path}{f' or wandb ({wandb_model_path})' if wandb_project is not None else ''}, and fail_if_cant_load is {fail_if_cant_load}"

  if wandb_project is not None:
    config_info = make_wandb_config(**locals())
    run = wandb.init(project=wandb_project, entity=wandb_entity, config=config_info, job_type="train")

  optimizer = optimizer(model.parameters(), **optimizer_kwargs)
  train_data_gen_lambda = (lambda: train_data_gen_maybe_lambda) if not train_data_gen_is_lambda else train_data_gen_maybe_lambda

  train_losses = []

  pbar = tqdm.tqdm(range(n_epochs))
  for epoch in pbar:
    train_data_gen = train_data_gen_lambda()
    epoch_losses = []
    for _ in range(batches_per_epoch):
      tokens = next(train_data_gen)
      logits = model(tokens)
      losses = loss_fn(logits, tokens, **loss_fn_kwargs)
      losses.mean().backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_losses.extend(losses.detach().cpu().numpy())

    train_losses.append(np.mean(epoch_losses))

    if print_every and epoch % print_every == 0:
      pbar.set_description(f'Epoch {epoch} train loss: {train_losses[-1]:.5e}')

    if wandb_project is not None:
      log_data = {'train_loss': train_losses[-1]}
      if log_acc: log_data['train_acc'] = acc_fn(model(tokens), tokens)
      wandb.log(log_data)

  model.eval()
  logits = model(data_test)
  acc = acc_fn(logits, data_test)

  print(f"Test accuracy after training: {acc}")

  if save_model:
    data = {
       "model":model.state_dict(),
       "config": model.cfg,
       "train_losses": train_losses,
       }
    if overwrite_data or not os.path.exists(model_pth_path):
      torch.save(data, model_pth_path)
      if wandb_project is not None:
        trained_model_artifact = wandb.Artifact(
            model_name, type="model", description=model_description, metadata=model.cfg.to_dict())
        trained_model_artifact.add_file(model_pth_path)
        run.log_artifact(trained_model_artifact)
    elif wandb_project is not None:
      print(f"Warning: {model_pth_path} already exists, saving model directly")
      run.log_artifact(data)

  if wandb_project is not None:
    run.finish()

  return train_losses, model_pth_path

import torch as th
!wandb login --anonymously
model = get_model(train_if_necessary=False)

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
    attn = masked_softmax(Q @ K.transpose(-2, -1) / d_k**0.5) @ V
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

# Undertrained model gets 98.7% accuracy, 91.6% soft accuracy
probs = th.nn.functional.softmax(forward(model, all_sequences), dim=-1)[:, -1]
print(float((probs.argmax(-1) == all_sequences.max(-1).values).float().mean()))
print(float(probs[th.arange(len(all_sequences)), all_sequences.max(-1).values].mean()))

# %% [markdown]
# Get a heuristic estimate for the (soft) accuracy of the model
SHOW = True
W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
W_Q, W_K= model.W_Q, model.W_K

d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx



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
# Statistics of EU

EU = W_E @ W_U
EU.squeeze_()

eu_mean = EU.mean()
eu_std = EU.std()

if SHOW:
    plt.imshow(EU.detach().cpu().numpy())
    plt.colorbar()
    plt.title("EU")
    plt.show()
    print(f"eu_mean: {eu_mean}\neu_std: {eu_std}")

plt.hist(EU.detach().cpu().numpy().flatten(), bins=100, alpha=.5)
# %% [markdown]
# Make necessary binned distributions
BINS = 100
evou_off_diag_dist = BinnedDist.from_list(list(off_diagonal), BINS)
resids = []
for i in range(EVOU.shape[0]):
    for j in range(EVOU.shape[1]):
        if i != j:
            resids.append((EVOU[i, j] - EVOU[i, i]).item())
evou_resid_dist = BinnedDist.from_list(resids, BINS)
eu_dist = BinnedDist.from_list(list(EU.flatten().detach().numpy()), BINS)

## This needs to be fixed to VOU * W_pos
pos_enc_dist = BinnedDist.from_list(list((W_pos @ W_V @ W_O @ W_U).flatten().detach().numpy()), BINS)
evou_dist = BinnedDist.from_list(list(EVOU.flatten().detach().numpy()), BINS)
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
other_V_mean = V[:, 1:].flatten().mean()
other_V_std = V[:, 1:].flatten().std()
other_U_mean = U[:, 1:].flatten().mean()
other_U_std = U[:, 1:].flatten().std()

first_singular_value = S[0]
other_singular_values_mean = S[1:].mean()
other_singular_values_std = S[1:].std()

# Contribution of the positional encoding of the key to the attention. Sign is + or - with equal probability
positional_adjustment = (W_E + W_pos[-1]) @ W_Q[0,0,:,:] @ W_K[0,0,:,:].T @ (W_pos[1] - W_pos[0]).unsqueeze(0).T
positional_adjustment_std = ((positional_adjustment**2).mean()**.5).item()

if SHOW:
    print(f"first_singular_value: {first_singular_value}\nother_singular_values_mean: {other_singular_values_mean}\nother_singular_values_std: {other_singular_values_std}")
    print(f"positional_adjustment_std: {positional_adjustment_std}")
    print(f"other_V_mean: {other_V_mean}\nother_V_std: {other_V_std}")
    print(f"other_U_mean: {other_U_mean}\nother_U_std: {other_U_std}")

# %% [markdown]
# Notice: Consecutive differences between projections of tokens onto key_direction are positive

consec_key_diffs = key_direction[1:] - key_direction[:-1]
consec_key_diffs_mean = consec_key_diffs.mean()
consec_key_diffs_std = consec_key_diffs.std()
query_proj_mean = query_direction.mean()
query_proj_std = query_direction.std()
if SHOW:
    print(f"consec_key_diffs_mean: {consec_key_diffs_mean}\nconsec_key_diffs_std: {consec_key_diffs_std}\nquery_proj_mean: {query_proj_mean}\nquery_proj_std: {query_proj_std}")
    plt.plot(consec_key_diffs.detach().cpu().numpy())
    plt.title("Consecutive differences in the key direction")
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
#
# We will approximate the mean and standard deviation of this value (over all choices of $x_1 \neq x_2$) by making the following assumptions:
# - For $i = 1$
#   - $t_2 \cdot u_1$ is an independent r.v. with mean and stdev given by the entries of the first column of $V$, i.e. `key_direction`
#   - $(t_{max} - t_{min}) \cdot v_i$ is an independent r.v. distributed by first choosing $x_1 \neq x_2$, then randomly summing $x_{max}-x_{min}$ independent normals (each with mean and std given by consecutive differences of entries in the first column of $U$, i.e. `query_direction`).
# - For $i > 1$:
#   - $s_i$ is an independent r.v. with the mean and stdev of the remaining singular values.
#   - $t_2 \cdot u_i$, $t_{max} \cdot v_i$, and $t_{min} \cdot v_i$ are independent r.v.s with means and stdevs of the non-first-column entries of $U$ and $V$, respectively.
# - $\mathrm{pos\_adjustment}_{x_2}$ is an independent r.v. with mean and stdev of entries of $\mathrm{pos\_adjustment}$ (with a random sign). Note this means the mean is $0$.
#
# The following facts about the distribution of $x_{max} - x_{min}$ come in handy, where both $x_{min}, x_{max}$ are drawn uniformly from $[n]$ and we condition on $x_{min} < x_{max}$:
# - Has mean $\frac{n+1}{3}$
# - Has variance $\frac{n^2+n}{6}$

x_diff_mean = (d_vocab + 1)/3
x_diff_var = (d_vocab**2 + d_vocab)/6 - x_diff_mean**2
X_mean = x_diff_mean * consec_key_diffs_mean
X_var = x_diff_mean * consec_key_diffs_std**2 + x_diff_var * consec_key_diffs_mean**2 # Law of total variance

presoftmax_attn_diff_mean = first_singular_value * query_proj_mean * consec_key_diffs_mean * x_diff_mean
presoftmax_attn_diff_std = (
    first_singular_value**2 * ((query_proj_mean**2 + query_proj_std**2) * (X_mean**2 + X_var) - query_proj_mean**2 * X_mean**2) +
    (d_vocab-1) * (other_singular_values_mean**2 + other_singular_values_std**2) * (other_U_mean**2 + other_U_std**2) * (2 * other_V_std**2) +
    positional_adjustment_std**2
)**.5

# TODO: Still have to adjust for the fact that we're conditioning on $x_1 \neq x_2$.

if SHOW:
    print(f"presoftmax_attn_diff_mean: {presoftmax_attn_diff_mean}\npresoftmax_attn_diff_std: {presoftmax_attn_diff_std}")

# %% [markdown]
# Visualize how close our approximation is to the actual distribution of pre-softmax attention difference
def get_actual_presoftmax_attn(model, input):

    if th.is_floating_point(input):
        x = input[(None,) * (3 - input.ndim)]
    else:
        x = input[(None,) * (2 - input.ndim)]
        x = th.nn.functional.one_hot(x, num_classes=model.cfg.d_vocab).float()

    x = x @ model.W_E
    x = x + model.W_pos[None, : x.shape[-2]]

    Q = x[:, None] @ model.W_Q[None, 0]
    K = x[:, None] @ model.W_K[None, 0]
    V = x[:, None] @ model.W_V[None, 0]

    return Q @ K.transpose(-2, -1)

all_attns = get_actual_presoftmax_attn(model, all_sequences).squeeze()
diffs = all_attns[:, 1, 0] - all_attns[:, 1, 1]
diffs[all_sequences[:, 0] < all_sequences[:, 1]] *= -1
diffs = diffs[all_sequences[:, 0] != all_sequences[:, 1]]

if SHOW:
    print(f"actual_diff_mean: {diffs.mean()}\nactual_diff_std: {diffs.std()}")

# This is very much not normally distributed...

# Simulate our approximation
from random import randint, normalvariate
def get_draw():
    xmin, xmax = 0, 0
    while(xmin == xmax):
        xmin = randint(0, d_vocab - 1)
        xmax = randint(0, d_vocab - 1)
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    ans = first_singular_value * normalvariate(query_proj_mean, query_proj_std) * normalvariate(consec_key_diffs_mean * (xmax - xmin), consec_key_diffs_std * (xmax - xmin)**.5) # most of the work going on here

    for _ in range(1, d_vocab):
        ans += normalvariate(other_singular_values_mean, other_singular_values_std) * normalvariate(other_U_mean, other_U_std) * normalvariate(0, other_V_std * 2**.5)
    ans += normalvariate(0, positional_adjustment_std)

    return ans

draws = [get_draw() for _ in range(diffs.shape[0])]
draws = torch.tensor(draws)
if SHOW:
    plt.hist(draws.detach().cpu().numpy(), bins=100, alpha=.5)
    plt.hist(diffs.detach().cpu().numpy(), bins=100, alpha=.5)
    plt.legend(["Simulated", "Actual"])

# %% [markdown]
# Functions to estimate E[f(X)] using a Taylor approximation of f around X
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


# %% [markdown]
# Estimate the mean and variance of post-softmax attention

def sigmoid_neg_x(x):
    """sigmoid(-x)"""
    return 1 / (1 + torch.exp(-x / W_K.shape[-1] ** .5))


postsoftmax_attn_mean_before_adjust = expected_taylor(sigmoid_neg_x, ell, [1, 0, presoftmax_attn_diff_std**2], True, presoftmax_attn_diff_mean)
postsoftmax_attn_2nd_moment_before_adjust = expected_taylor(lambda x: sigmoid_neg_x(x)**2, ell, [1, 0, presoftmax_attn_diff_std**2], True, presoftmax_attn_diff_mean)

# Adjust for fact that 1 / d_vocab of the time, x_1 = x_2 so the postsoftmax_attn should be 1.
postsoftmax_attn_mean = 1 / d_vocab + (d_vocab - 1) / d_vocab * postsoftmax_attn_mean_before_adjust
postsoftmax_attn_2nd_moment = 1 / d_vocab + (d_vocab - 1) / d_vocab * postsoftmax_attn_2nd_moment_before_adjust
postsoftmax_attn_var = postsoftmax_attn_2nd_moment - postsoftmax_attn_mean ** 2 # Huh this might be negative

if(postsoftmax_attn_var < 0):
    postsoftmax_attn_var = 0
    # We're definitely going to have to move away from taylor approximations now...


if SHOW:
    print(f"postsoftmax_attn_mean:{postsoftmax_attn_mean}\npostsoftmax_attn_var:{postsoftmax_attn_var}")
# %% [markdown]
# Estimate the final output distribution

import scipy.stats as stats
from math import exp

if postsoftmax_attn_var == 0: # Handle degenerate case
    a_dist = BinnedDist.from_cdf(lambda x: 1 if x>=postsoftmax_attn_mean else 0, BINS, float(postsoftmax_attn_mean-1e-6), float(postsoftmax_attn_mean+1e-6))
else:
    a_dist = BinnedDist.from_cdf(lambda x: stats.norm.cdf(x, loc=postsoftmax_attn_mean, scale=postsoftmax_attn_var**.5), BINS, float(postsoftmax_attn_mean - 10 * (postsoftmax_attn_var**.5)), float(postsoftmax_attn_mean + 10 * (postsoftmax_attn_var**.5)))

correct_logit_dist = evou_off_diag_dist.convolve_with(pos_enc_dist).apply_func2(lambda x, a: x*(1-a), a_dist).convolve_with(eu_dist)
incorrect_logit_dist = evou_resid_dist.convolve_with(pos_enc_dist).apply_func2(lambda x, a: x*a, a_dist).convolve_with(
    evou_dist.convolve_with(pos_enc_dist).apply_func2(lambda x, a: x*(1-a), a_dist)
).convolve_with(eu_dist)


exp_correct_logit_dist = correct_logit_dist.apply_func(lambda x: exp(x))
exp_incorrect_logit_dist = incorrect_logit_dist.apply_func(lambda x: exp(x))


# Now, convolve exp_incorrect_logit_dist with itself d_vocab-1 times, using binary lifting
to_convolve = []
curr_dist = exp_incorrect_logit_dist
curr_power = 1
while curr_power <= (d_vocab-1):
    if curr_power & (d_vocab-1):
        to_convolve.append(curr_dist)
    curr_dist = curr_dist.convolve_with(curr_dist)
    curr_power <<= 1
while len(to_convolve) > 1:
    to_convolve.append(to_convolve.pop().convolve_with(to_convolve.pop()))
convolved_exp_incorrect_logit_dist = to_convolve[0]

final_output_dist = exp_correct_logit_dist.apply_func2(lambda a, b: a/(a+b), convolved_exp_incorrect_logit_dist)

if SHOW:
    final_output_dist.plot()
    plt.title("Estimated Final output distribution")
    print(f"estimated accuracy: {final_output_dist.mean()}")
