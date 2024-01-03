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

DETERMINISTIC = True # @param
N_LAYERS = 1 # @param
N_HEADS = 1 # @param
D_MODEL = 32 # @param
D_HEAD = 32 # @param
D_MLP = None # @param
D_VOCAB = 64 # @param
SEED = 123 # @param
N_CTX = 2 # @param
DEVICE = "cuda" if torch.cuda.is_available() and not DETERMINISTIC else "cpu"

MODEL_INFO = {'2-1500': ('tkwa-team', 1500), '2-32': ('team-jason', 32)}

def get_model(model_id):
    """
    '2-1500': overtrained, max-of-2, 1500 epochs
    '2-32': undertrained, max-of-2, 32 epochs
    """

    wandb_entity, n_epochs = MODEL_INFO[model_id]

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

    model_name = f'neural-net-coq-interp-max-{model.cfg.n_ctx}-epochs-{n_epochs}'
    wandb_model_path = f"{wandb_entity}/{model_name}/{model_name}:latest"
    model_dir = None
    try:
        api = wandb.Api()
        model_at = api.artifact(wandb_model_path)
        model_dir = Path(model_at.download())
    except Exception as e:
        print(f'Could not load model {wandb_model_path} from wandb:\n', e)
    if model_dir is not None:
        for model_path in model_dir.glob('*.pth'):
            try:
                cached_data = torch.load(model_path)
                model.load_state_dict(cached_data['model'])
                #model_checkpoints = cached_data["checkpoints"]
                #checkpoint_epochs = cached_data["checkpoint_epochs"]
                #test_losses = cached_data['test_losses']
                # train_losses = cached_data['train_losses']
                #train_indices = cached_data["train_indices"]
                #test_indices = cached_data["test_indices"]
                return model
            except Exception as e:
                print(f'Could not load model from {model_path}:\n', e)


def generate_all_sequences(n_digits: int, sequence_length: int = 2):
  data = list(itertools.product(range(n_digits), repeat=sequence_length))
  data = torch.tensor(data)
  return data