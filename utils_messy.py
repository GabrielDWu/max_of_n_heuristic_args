# %%

from typing import Callable, Iterable, List, Optional, Tuple
import einops
from fancy_einsum import einsum
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import torch
import torch.nn.functional as F
import transformer_lens.utils as utils
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots
from inspect import signature
import itertools
import sys
from transformer_lens import HookedTransformer, HookedTransformerConfig
import tqdm.auto as tqdm

DETERMINISTIC = True # @param
DEVICE = "cuda" if torch.cuda.is_available() and not DETERMINISTIC else "cpu"
N_LAYERS = 1 # @param
N_HEADS = 1 # @param
D_MODEL = 32 # @param
D_HEAD = 32 # @param
D_MLP = None # @param
D_VOCAB = 64 # @param
SEED = 123 # @param
N_EPOCHS = 1500 # @param
N_CTX = 2 # @param
FORCE_ADJACENT = True # @param
BATCH_SIZE = 128 # @param
FAIL_IF_CANT_LOAD = '--fail-if-cant-load' in sys.argv[1:] # @param

ALWAYS_TRAIN_MODEL = False # @param
SAVE_IN_GOOGLE_DRIVE = False # @param
OVERWRITE_DATA = False # @param
TRAIN_MODEL_IF_CANT_LOAD = True # @param


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

model_is_trained = False


#from training_utils.py

# # Data Generation

# Helper functions for generating data and splitting it into batches for training.


import datetime
import os, os.path
from pathlib import Path
from typing import List, Any, Iterable, Optional
import numpy as np
import tqdm
from transformer_lens import HookedTransformer
import torch
import itertools

import wandb

def default_device(deterministic: bool = False) -> str:
   return "cuda" if torch.cuda.is_available() and not deterministic else "cpu"


DEFAULT_WANDB_ENTITY = 'tkwa-team' # 'team-jason'


def in_colab() -> bool:
    """
    Returns True if running in Google Colab, False otherwise.
    """
    try:
        import google.colab
        return True
    except:
        return False


def get_pth_base_path(save_in_google_drive: bool = False, create: bool = True) -> Path:
    """
    Returns the base path for saving models. If `save_in_google_drive` is True, returns the path to the Google Drive
    folder where models are saved. Otherwise, returns the path to the local folder where models are saved.
    """
    if in_colab():
        if save_in_google_drive:
            from google.colab import drive
            drive.mount('/content/drive/')
            pth_base_path = Path('/content/drive/MyDrive/Colab Notebooks/')
        else:
            pth_base_path = Path("/workspace/_scratch/")
    else:
        pth_base_path = Path(os.getcwd())

    pth_base_path = pth_base_path / 'trained-models'

    if create and not os.path.exists(pth_base_path):
        os.makedirs(pth_base_path)

    return pth_base_path


def generate_all_sequences(n_digits: int, sequence_length: int = 2):
  data = list(itertools.product(range(n_digits), repeat=sequence_length))
  data = torch.tensor(data)
  return data


def compute_all_tokens(model: HookedTransformer):
    return generate_all_sequences(n_digits=model.cfg.d_vocab, sequence_length=model.cfg.n_ctx)


def shuffle_data(data):
  indices = np.array(range(len(data)))
  np.random.shuffle(indices)
  data = data[indices]
  return data


def make_testset_trainset(
    model: HookedTransformer,
    training_ratio=0.7,
    force_adjacent=False):
  """
  Generate a train and test set of tuples containing `sequence_length` integers with values 0 <= n < n_digits.

  Args:
      sequence_length (int): The length of each tuple in the dataset.
      n_digits (int): The number of possible values for each element in the tuple.
      training_ratio (float): The ratio of the size of the training set to the full dataset.
      force_adjacent (bool): Whether to make training adversarial (force to include all (x, x +- 1))

  Returns:
      Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]: A tuple containing the training set and test set.
          The training set contains `training_ratio` percent of the full dataset, while the test set contains the
          remaining data. Each set is a list of tuples containing `sequence_length` integers with values 0 <= n < n_digits.
          The tuples have been shuffled before being split into the train and test sets.
  """
  data = compute_all_tokens(model)

  data = shuffle_data(data)

  if force_adjacent:
    idxs = (data[:,0] - data[:,1]).abs() == 1
    data, extra_data = data[~idxs], data[idxs]
    data = torch.cat([extra_data, data], dim=0)

  split_idx = int(len(data) * training_ratio)

  data_train = data[:split_idx]
  data_test = data[split_idx:]

  if force_adjacent:
    data_train = shuffle_data(data_train)
    data_test = shuffle_data(data_test)

  return data_train, data_test


def make_generator_from_data(data: List[Any], batch_size: int = 128) -> Iterable[List[Any]]:
  """
  Returns a generator that yields slices of length `batch_size` from a list.

  Args:
      data: The input list to be split into batches.
      batch_size: The size of each batch.

  Yields:
      A slice of the input list of length `batch_size`. The final slice may be shorter if the
      length of the list is not evenly divisible by `batch_size`.
  """
  data = shuffle_data(data)
  for i in range(0,len(data), batch_size):
    yield data[i:i+batch_size]


def make_wandb_config(
    model:HookedTransformer,
    optimizer_kwargs: dict,
    n_epochs=100,
    batch_size=128,
    batches_per_epoch=10,
    adjacent_fraction=0,
    use_complete_data=True,
    device=None,
    **kwargs):
  return {
      'model.cfg':model.cfg.to_dict(),
      'optimizer.cfg':optimizer_kwargs,
      'n_epochs':n_epochs,
      'batch_size':batch_size,
      'batches_per_epoch':batches_per_epoch,
      'adjacent_fraction':adjacent_fraction,
      'use_complete_data':use_complete_data,
      'device':device,
    }

def load_model(model: HookedTransformer, model_pth_path: str):
  try:
    cached_data = torch.load(model_pth_path)
    model.load_state_dict(cached_data['model'])
    #model_checkpoints = cached_data["checkpoints"]
    #checkpoint_epochs = cached_data["checkpoint_epochs"]
    #test_losses = cached_data['test_losses']
    train_losses = cached_data['train_losses']
    #train_indices = cached_data["train_indices"]
    #test_indices = cached_data["test_indices"]
    return train_losses, model_pth_path
  except Exception as e:
    print(f'Could not load model from {model_pth_path}:\n', e)

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
  if force_train and fail_if_cant_load: raise ValueError(f"force_train is {force_train} and fail_if_cant_load is {fail_if_cant_load}")
  if device is None: device = default_device(deterministic=deterministic)

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

# max_of_n.py

import numpy as np
import torch
from transformer_lens import HookedTransformer
import tqdm.auto as tqdm
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def loss_fn(
    logits, # [batch, pos, d_vocab]
    tokens, # [batch, pos]
    return_per_token=False,
    device=DEVICE,
  ):
  logits = logits[:, -1, :].to(device)
  true_maximum = torch.max(tokens.to(device), dim=1)[0]
  log_probs = logits.log_softmax(-1)
  correct_log_probs = log_probs.gather(-1, true_maximum.unsqueeze(-1))
  if return_per_token:
    return -correct_log_probs.squeeze()
  return -correct_log_probs.mean()


def acc_fn(
    logits, # [batch, pos, d_vocab]
    tokens, # [batch, pos]
    return_per_token=False,
    device=DEVICE,
  ):
  pred_logits = logits[:, -1, :].to(device)
  pred_tokens = torch.argmax(pred_logits, dim=1)
  true_maximum = torch.max(tokens.to(device), dim=1)[0]
  if return_per_token:
    return (pred_tokens == true_maximum).float()
  return (pred_tokens == true_maximum).float().mean().item()


def large_data_gen(n_digits, sequence_length=6, batch_size=128, context="train", device=DEVICE, adjacent_fraction=0):
  if context == "train":
    seed = 5
  else:
    seed = 6
  torch.manual_seed(seed)
  while True:
    result = torch.randint(0, n_digits, (batch_size, sequence_length)).to(device)
    if adjacent_fraction == 0: yield result
    else:
      adjacent = torch.randint(0, n_digits, (batch_size,))
      adjacent = adjacent.unsqueeze(1).repeat(1, sequence_length)
      # in half the rows, replace a random column with n+1
      rows_to_change = torch.randperm(batch_size)[:batch_size // 2]
      cols_to_change = torch.randint(0, sequence_length, (batch_size // 2,))
      adjacent[rows_to_change, cols_to_change] += 1
      adjacent %= n_digits
      adjacent = adjacent.to(device)
      mask = torch.rand(batch_size) < adjacent_fraction
      result[mask] = adjacent[mask]
      yield result

def make_wandb_config(
    model:HookedTransformer,
    n_epochs=100,
    batch_size=128,
    batches_per_epoch=10,
    adjacent_fraction=0,
    use_complete_data=True,
    device=DEVICE,
    lr=1e-3,
    betas=(.9, .999),
    **kwargs):
  return {
      'model.cfg':model.cfg.to_dict(),
      'optimizer.cfg':{
        'lr':lr,
        'betas':betas,
      },
      'n_epochs':n_epochs,
      'batch_size':batch_size,
      'batches_per_epoch':batches_per_epoch,
      'adjacent_fraction':adjacent_fraction,
      'use_complete_data':use_complete_data,
      'device':device,
    }

def train_model(
    model:HookedTransformer,
    n_epochs=100,
    batch_size=128,
    batches_per_epoch=10,
    adjacent_fraction=0,
    use_complete_data=True,
    device=DEVICE,
    use_wandb=False,
    wandb_project=None,
    save_model=True,
  ):
  lr = 1e-3
  betas = (.9, .999)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
  n_digits, sequence_length = model.cfg.d_vocab, model.cfg.n_ctx
  train_losses = []
  if wandb_project is not None:
    config_info = make_wandb_config(model, **locals())
    run = wandb.init(project=wandb_project, config=config_info, job_type="train")

  if use_complete_data:
    data_train, data_test = make_testset_trainset(model, force_adjacent=adjacent_fraction > 0)
    train_data_gen_gen = lambda: make_generator_from_data(data_train, batch_size=batch_size)
  else:
    train_data_gen = large_data_gen(n_digits=n_digits, sequence_length=sequence_length, batch_size=batch_size, context="train", device=device, adjacent_fraction=adjacent_fraction)
    test_data_gen = large_data_gen(n_digits=n_digits, sequence_length=sequence_length, batch_size=batch_size * 20, context="test", adjacent_fraction=adjacent_fraction)
    data_test = next(test_data_gen)

  for epoch in tqdm.tqdm(range(n_epochs)):
    if use_complete_data:
      train_data_gen = train_data_gen_gen()
    epoch_losses = []
    for _ in range(batches_per_epoch):
      tokens = next(train_data_gen)
      logits = model(tokens)
      losses = loss_fn(logits, tokens, return_per_token=True)
      losses.mean().backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_losses.extend(losses.detach().cpu().numpy())

    train_losses.append(np.mean(epoch_losses))

    if epoch % 10 == 0:
      print(f'Epoch {epoch} train loss: {train_losses[-1]}')

    if use_wandb or wandb_project is not None:
      wandb.log({'train_loss': train_losses[-1]})

  model.eval()
  logits = model(data_test)
  acc = acc_fn(logits, data_test)

  print(f"Test accuracy after training: {acc}")

  if save_model and (use_wandb or wandb_project is not None):
    wandb.log_artifact(model)

  if wandb_project is not None:
    run.finish()

  return train_losses

def train(fail_if_cant_load=FAIL_IF_CANT_LOAD, train_if_cant_load=TRAIN_MODEL_IF_CANT_LOAD, overwrite_data=OVERWRITE_DATA,
          always_train_model=ALWAYS_TRAIN_MODEL,
          wandb_entity=DEFAULT_WANDB_ENTITY,
          save_in_google_drive=SAVE_IN_GOOGLE_DRIVE):

    global model_is_trained

    data_train, data_test = make_testset_trainset(model, force_adjacent=FORCE_ADJACENT)
    train_data_gen_gen = lambda: make_generator_from_data(data_train, batch_size=BATCH_SIZE)

    training_losses, model_pth_path = train_or_load_model(
        f'neural-net-coq-interp-max-{model.cfg.n_ctx}-epochs-{N_EPOCHS}',
        model,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        train_data_gen_maybe_lambda=train_data_gen_gen,
        train_data_gen_is_lambda=True,
        data_test=data_test,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        adjacent_fraction=1,
        use_complete_data=True,
        batches_per_epoch=10,
        wandb_project=f'neural-net-coq-interp-max-{model.cfg.n_ctx}-epochs-{N_EPOCHS}',
        deterministic=DETERMINISTIC,
        save_in_google_drive=save_in_google_drive,
        overwrite_data=overwrite_data,
        train_model_if_cant_load=train_if_cant_load,
        model_description=f"trained max of {model.cfg.n_ctx} model",
        save_model=True,
        force_train=always_train_model,
        wandb_entity=wandb_entity,
        fail_if_cant_load=fail_if_cant_load,
    )

    model_is_trained = True
    return training_losses, model_pth_path

def get_model(train_if_necessary = False,  **kwargs):

    train(fail_if_cant_load = not train_if_necessary, train_if_cant_load = train_if_necessary, **kwargs)

    return model
