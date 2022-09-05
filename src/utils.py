"""
Utilities for determining paths to corpora, results, models
given config dictionaries describing an experiment, as well
as determining canonical vocabulary ordering
"""

import os
import string
import re
import copy
from tracemalloc import start
from typing import Optional
from pyrsistent import optional
import torch 
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as func 
import math 
from packaging import version
from torch import Tensor, nn

def get_identifier_iterator():
  """ Returns an iterator to provide unique ids to bracket types.
  """
  ids = iter(list(string.ascii_lowercase))
  k = 1
  while True:
    try:
      str_id = next(ids)
    except StopIteration:
      ids = iter(list(string.ascii_lowercase))
      k += 1
      str_id = next(ids)
    yield str_id*k

def get_vocab_of_bracket_types(bracket_types):
  """ Returns the vocabulary corresponding to the number of brackets.
  There are bracket_types open brackets, bracket_types close brackets,
  START, and END.
  Arguments:
    bracket_types: int (k in Dyck-(k,m))
  Returns:
    Dictionary mapping symbol string  s to int ids.
  """
  id_iterator = get_identifier_iterator()
  ids = [next(id_iterator) for x in range(bracket_types)]
  vocab = {x: c for c, x in enumerate(['(' + id_str for id_str in ids] + [id_str + ')' for id_str in ids] + ['START', 'END'])}
  return vocab, ids

def get_results_dir_of_args(args):
  """
  Takes a (likely yaml-defined) argument dictionary
  and returns the directory to which results of the
  experiment defined by the arguments will be saved
  """
  return args['reporting']['reporting_loc']

def get_corpus_paths_of_args(args):
  return {'train': args['corpus']['train_corpus_loc'],
          'dev': args['corpus']['dev_corpus_loc'],
          'test': args['corpus']['test_corpus_loc'],
          'test20': args['corpus']['test20_corpus_loc']}

def get_lm_path_of_args(args):
  results_dir = get_results_dir_of_args(args)
  return os.path.join(results_dir, args['lm']['save_path'])

def deprecated_get_results_dir_of_args(args):
  """ (Deprecated)
  Takes a (likely yaml-defined) argument dictionary
  and returns the directory to which results of the
  experiment defined by the arguments will be saved
  """
  if 'vocab_size' in args['language']:
    del args['language']['vocab_size']
  if (args['corpus']['train_override_path'] or 
      args['corpus']['dev_override_path'] or 
      args['corpus']['test_override_path']):
    language_specification_path = re.sub('train', '', args['corpus']['train_override_path'])
  else:
    language_specification_path = '_'.join([key+str(value) for key, value in args['language'].items()])

  path = os.path.join(
   args['reporting']['root'],
   '-'.join([
    '_'.join([key+str(value) for key, value in args['lm'].items()]),
    '_'.join([key+str(value) for key, value in args['training'].items()]),
    language_specification_path,
    ]))
  path = re.sub('max_stack_depth', 'msd', path)
  path = re.sub('learning_rate', 'lr', path)
  path = re.sub('sample_count', 'sc', path)
  path = re.sub('max_length', 'ml', path)
  path = re.sub('train', 'tr', path)
  path = re.sub('test', 'te', path)
  path = re.sub('num_layers', 'nl', path)
  path = re.sub('hidden_dim', 'hd', path)
  path = re.sub('embedding_dim', 'ed', path)
  path = re.sub('analytic_model', 'am', path)
  path = re.sub('max_epochs', 'me', path)
  path = re.sub('batch_size', 'bs', path)
  path = re.sub('min_length', 'ml', path)
  path = re.sub('min_state_filter_percent', 'fp', path)
  return path

def deprecated_get_corpus_paths_of_args(args):
  """ (Deprecated)
  Takes a (likely yaml-defined) argument dictionary
  and returns the paths of the train/dev/test
  corpora files.
  """
  args = copy.deepcopy(args)
  if 'vocab_size' in args['language']:
    del args['language']['vocab_size']
  if (args['corpus']['train_override_path'] or 
      args['corpus']['dev_override_path'] or 
      args['corpus']['test_override_path']):
    train_path = args['corpus']['train_override_path']
    dev_path = args['corpus']['dev_override_path']
    test_path = args['corpus']['test_override_path']
  else:
    path = os.path.join(
     args['corpus']['root'],
    '-'.join([
      '_'.join([key+str(value) for key, value in args['language'].items()])]))
    path = re.sub('max_stack_depth', 'msd', path)
    path = re.sub('learning_rate', 'lr', path)
    path = re.sub('sample_count', 'sc', path)
    path = re.sub('max_length', 'ml', path)
    path = re.sub('train', 'tr', path)
    path = re.sub('test', 'te', path)
    path = re.sub('num_layers', 'nl', path)
    path = re.sub('hidden_dim', 'hd', path)
    path = re.sub('embedding_dim', 'ed', path)
    path = re.sub('analytic_model', 'am', path)
    path = re.sub('max_epochs', 'me', path)
    path = re.sub('batch_size', 'bs', path)
    path = re.sub('min_length', 'ml', path)
    path = re.sub('min_state_filter_percent', 'fp', path)
    train_path = os.path.join(path, 'train.formal.txt')
    dev_path = os.path.join(path, 'dev.formal.txt')
    test_path = os.path.join(path, 'test.formal.txt')
      #path = re.sub('', 'te', path)
  return {'train':train_path, 'dev':dev_path, 'test':test_path}



class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.4") or use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)


class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.7"):
            self.act = self._silu_python
        else:
            self.act = nn.functional.silu

    def _silu_python(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(input)

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class MishActivation(nn.Module):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9"):
            self.act = self._mish_python
        else:
            self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        return input


ACT2FN = {
    "gelu": GELUActivation(),
    "gelu_10": ClippedGELUActivation(-10, 10),
    "gelu_fast": FastGELUActivation(),
    "gelu_new": NewGELUActivation(),
    "gelu_python": GELUActivation(use_gelu_python=True),
    "linear": LinearActivation(),
    "mish": MishActivation(),
    "quick_gelu": QuickGELUActivation(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "silu": SiLUActivation(),
    "swish": SiLUActivation(),
    "tanh": nn.Tanh(),
}

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


# For backwards compatibility with: from activations import gelu_python
gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")

#used to initialise model
class Config(object):
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs,
    ):
        self.hidden_size=n_embd
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.max_position_embeddings =n_positions
        self.num_hidden_layers=n_layer
        self.num_attention_heads=n_head
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__()