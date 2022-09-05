
import time
import math
import os
from dataclasses import dataclass

from typing import Optional, Tuple, Union
from transformers import GPT2PreTrainedModel
import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from utils import *
#from model_parallel_utils import *
        
#参考github原始代码以及transfomers库的实现，该版本不是标准的transfomer。比较像bert

#refer from Transformers Moudle

class PreTrainedModel(nn.Module):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    def __init__(self, config):
        super().__init__()
        self.config=config

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights.
        """
        # Prune heads if needed
        if self.config.pruned_heads:
            
            self.prune_heads(self.config.pruned_heads)

        self.apply(self._init_weights)
            

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()


from ast import Pass
from distutils.command.config import config
import time

import torch 
import torch.nn as nn 
from torch.nn import Module
from typing import Optional
from torch import Tensor
import torch.nn.functional as F 
from utils import *
#参考github原始代码以及transfomers库的实现，该版本不是标准的transfomer。比较像bert

#refer from Transformers Moudle
class PreTrainedModel(nn.Module):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))



class myTransformer(PreTrainedModel):
    def __init__(self,config) -> None:
        super(myTransformer,self).__init__(
        )
        self.config=config
        self.transformer=Transformer(config)
        self.mlp=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.post_init()

    def forward(self,input_id:Tensor,attention_mask:Optional[Tensor]=None,head_mask: Optional[Tensor] = None):
        return self.mlp(self.transformer(input_id,attention_mask,head_mask))

    def post_init(self):
        self.apply(self._init_weights)
#论文代码实现在训练的时候没有采用任何mask
class Transformer(PreTrainedModel):
    def __init__(self,config) -> None:
        super(Transformer,self).__init__()
        self.config=config
        self.embed_dim = config.hidden_size

        #embedding of all brackets types 
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        #embedding for the bracket position in sentence position
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([encoderBlock(config) for i in range(config.num_hidden_layers)])
        
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        
        self.post_init()

    def post_init(self):
        self.apply(self._init_weights)

    def forward(self,input_id:Tensor,attention_mask:Optional[Tensor]=None,head_mask: Optional[Tensor] = None):
        input_embed=self.wte(input_id).to("cuda")
        input_shape=input_id.size()
        position_ids = torch.arange(input_shape[-1], dtype=torch.long).cuda()
        position_embed=self.wpe(position_ids).to("cuda")

        hidden_state=input_embed+position_embed

        for block in self.h:
            hidden_state=block(
                hidden_state,
                attention_mask,
                head_mask
            ) 
        
        return  (self.ln_f(hidden_state))

class encoderBlock(PreTrainedModel):
    def __init__(self,config) -> None:
        super(encoderBlock,self).__init__()
        self.config=config
        self.hidden_size = config.hidden_size   

        self.ln1=nn.LayerNorm(self.hidden_size,eps=config.layer_norm_epsilon)
        self.attnBlock=attnBlock(config)
        self.ln2=nn.LayerNorm(self.hidden_size,eps=config.layer_norm_epsilon)
        self.mlp=MLP(self.hidden_size,config)
        

    def forward(self,tensor:Tensor,attention_mask:Optional[Tensor]=None,head_mask: Optional[Tensor] = None):
        res=tensor
        attn_output=self.attnBlock(self.ln1(tensor))
        #shortcut connect
        hidden_state=res+attn_output
        res=hidden_state
        hidden_state=self.ln2(hidden_state)
        hidden_state=self.mlp(hidden_state)+res

        return hidden_state

class attnBlock(PreTrainedModel):
    def __init__(self,config):
        super(attnBlock,self).__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.proj_attn = Conv1D(3*self.embed_dim,self.embed_dim)
        self.attn_drop= nn.Dropout(config.attn_pdrop)
        max_positions = config.max_position_embeddings
        #上三角的mask,方便广播
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )

        self.c_proj= Conv1D(self.embed_dim,self.embed_dim)
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                " {}).".format(self.num_heads)
            )
    
    

    def attn_mechanism(self,q:Tensor,k:Tensor,v:Tensor,attn_mask:Optional[Tensor]=None,head_mask:Optional[Tensor]=None)->Tensor: 
        
        attn=q@k.transpose(-1,-2)
        attn=attn/torch.tensor(
                v.size(-1) ** 0.5, dtype=attn.dtype, device=attn.device
            )

        query_length, key_length = q.size(-2), k.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
        mask_value = torch.finfo(attn.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn.dtype).to(attn.device)
        attn = torch.where(causal_mask, attn, mask_value)
        
        if(attn_mask is not None):
            attn+=attn_mask
        attn=nn.functional.softmax(attn,dim=-1)
        if(head_mask is not None):
            attn=attn@head_mask
        return  attn@v
             
    def forward(self,tensor:Tensor,attn_mask:Optional[Tensor]=None, head_mask:Optional[Tensor]=None)->Tensor:
        query,key,value=self.proj_attn(tensor).split(self.split_size,-1)
        
        query=self.split_heads(query,self.num_heads,self.head_dim)
        key = self.split_heads(key, self.num_heads, self.head_dim)
        value = self.split_heads(value, self.num_heads, self.head_dim)
        
        attn_output=self.attn_mechanism(query,key,value,attn_mask,head_mask)
        
        out=self.merge_heads(attn_output,self.num_heads,self.head_dim)
        
        out=self.c_proj(out)
        return  self.attn_drop(out)

    def merge_heads(self,tensor:Tensor,num_heads:int,attn_dim:int):
        tensor=tensor.permute(0,2,1,3)
        new_shape =tensor.size()[:-2]+(num_heads*attn_dim,)
        return tensor.view(new_shape)

    def split_heads(self,tensor:Tensor,num_heads:int,attn_dim:int)->Tensor:
        new_shape=tensor.size()[:-1]+(num_heads,attn_dim,)
        tensor=tensor.view(new_shape)
        #(batch,num_head,len_seq,attn_dim)
        return tensor.permute(0,2,1,3)


class MLP(PreTrainedModel):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states:Tensor) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


