import torch
import numpy as np
# import einops
import tqdm.auto as tqdm
import plotly.express as px

from typing import List, Union, Optional, Literal
from jaxtyping import Float, Int
from functools import partial

# from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F


def run_with_cache_and_hooks(
    model,
    fwd_hooks,
    *model_args,
    **model_kwargs,
):
    """
    Runs the model and returns model output and a Cache object.
    Applies all hooks in fwd_hooks.
    """
    cache_dict = model.add_caching_hooks()
    for name, hook in fwd_hooks:
        if type(name) == str:
            model.mod_dict[name].add_hook(hook, dir="fwd")
        else:
            # Otherwise, name is a Boolean function on names
            for hook_name, hp in model.hook_dict.items():
                if name(hook_name):
                    hp.add_hook(hook, dir="fwd")

    model_out = model(*model_args, **model_kwargs)

    model.reset_hooks(False, including_permanent=False)
    return model_out, cache_dict

def residual_stream_noise_hook(
    resid: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: int,
    noise_scale: float = 0.0001,
) -> Float[torch.Tensor, "batch pos d_model"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    # resid[:, position, :] = resid[:, position, :] + noise_scale * torch.randn_like(resid[:, position, :])
    resid[:, position, :] = resid[:, position, :] + torch.normal(0, noise_scale, size=resid[:, position, :].shape).to(resid.device)
    return resid


def residual_stream_addition_hook(
    resid: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: int,
    ref_resid: None,
    add_scale: float = 0.0001,
) -> Float[torch.Tensor, "batch pos d_model"]:

    resid[:, position, :] = resid[:, position, :] + add_scale * ref_resid[:].to(resid.dtype).to(resid.device)
    return resid


def mlp_post_suppression_hook(
    mlp_post: Float[torch.Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    position: Union[int, List[int]],
    neuron: Union[int, List[int]],
) -> Float[torch.Tensor, "batch pos d_mlp"]:

    mlp_post[:, position, neuron] = -20
    return mlp_post


def mlp_post_activate_hook(
    mlp_post: Float[torch.Tensor, "batch pos d_mlp"],
    hook: HookPoint,
    position: Union[int, List[int]],
    neuron: Union[int, List[int]],
) -> Float[torch.Tensor, "batch pos d_mlp"]:

    mlp_post[:, position, neuron] = 1.5
    return mlp_post
    

def residual_stream_replace_hook(
    resid: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: Union[int, List[int]],
    ref_resid=None,
) -> Float[torch.Tensor, "batch pos d_model"]:
    resid[:, position, :] = ref_resid[position, :]
    return resid

def head_pattern_replace_hook(
    pattern: Float[torch.Tensor, "batch head_index query_pos key_pos"],
    hook: HookPoint,
    layer_head_idx = None,
    ref_pattern = None,
) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
    pattern[:, layer_head_idx[1], :, :] = ref_pattern
    return pattern


def head_pattern_modify_hook(
    pattern: Float[torch.Tensor, "batch head_index query_pos key_pos"],
    hook: HookPoint,
    layer_head_idx,
) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
    pattern[:, layer_head_idx[1], :, :] = ref_cache["pattern", layer_head_idx[0], "attn"][layer_head_idx[1], :, :]
    return pattern


def attn_out_replace_hook(
    attn_out: Float[torch.Tensor, "batch head_index query_pos key_pos"],
    hook: HookPoint,
    position: Union[int, List[int]],
    ref_cache,
) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
    attn_out[:, :, position, :] = ref_cache["attn_out", position, "attn"]
    return attn_out


# hook functions
# zeroing residual stream, can apply to h, mlp out, but not attn out
def residual_stream_zeroing_hook(
    resid: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: Union[int, List[int]],
) -> Float[torch.Tensor, "batch pos d_model"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    resid[:, position, :] = 0
    return resid


# isolate attn effect on resid by substract attn out from resid_post
# or isolate attn effect on mlp by substract attn out from resid_mid then add back on resid post, this is the second hook
def attn_out_hook(
    resid: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: Union[int, List[int]],
    attn_out: Float[torch.Tensor, "d_model"],
    add_back: bool = False,
) -> Float[torch.Tensor, "batch pos d_model"]:
    if not add_back:
        resid[:, position, :] -= attn_out
    else:
        resid[:, position, :] += attn_out
    return resid


def residual_stream_editing_hook(
    resid: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: Union[int, List[int]],
) -> Float[torch.Tensor, "batch pos d_model"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    temp_resid = resid[:, position, :]
    variance = temp_resid.pow(2).mean(-1, keepdim=True)
    temp_resid = temp_resid * torch.rsqrt(variance + 1e-6)
    temp_resid = temp_resid @ edit_matrix.bfloat16().to("cuda")
    temp_resid = temp_resid / torch.rsqrt(variance + 1e-6)
    resid[:, position, :] = temp_resid
    return resid


# adding noise to residual stream
def residual_stream_noise_hook(
    resid_pre: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: int
) -> Float[torch.Tensor, "batch pos d_model"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    resid_pre[:, position, :] = torch.randn_like(resid_pre[:, position, :])
    return resid_pre


# replace the residual stream with bos position residual stream
def residual_stream_bos_replace_hook(
    resid_pre: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: int,
    bos_layer=0,
    cache=None,
) -> Float[torch.Tensor, "batch pos d_model"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    resid_pre[:, position, :] = cache[f"blocks.{bos_layer}.hook_resid_pre"][0]
    return resid_pre


# replace the residual stream with certain layer residual stream
# def residual_stream_replace_hook(
#     resid_pre: Float[torch.Tensor, "batch pos d_model"],
#     hook: HookPoint,
#     position: Union[int, List[int]],
#     layer: int,
#     cache=None,
# ) -> Float[torch.Tensor, "batch pos d_model"]:
#     resid_pre[:, position, :] = cache[f"blocks.{layer}.hook_resid_pre"][position, :]
#     return resid_pre


# zeroing attention pattern at position(s)
def attn_pattern_zeroing_hook(
    pattern: Float[torch.Tensor, "batch head_index query_pos key_pos"],
    hook: HookPoint,
    position: Union[int, List[int]],
    src_position: Optional[Union[int, List[int]]] = None,
):
    if src_position is not None:
        if isinstance(src_position, int):
            pattern[:, :, position, src_position] = 0
        elif isinstance(src_position, list):
            src_position = np.array(src_position).reshape(-1, 1)
            pattern[:, :, position, src_position] = 0
            
    else:
        pattern[:, :, position, :] = 0
    return pattern


def attn_pattern_amplify_hook(
    pattern: Float[torch.Tensor, "batch head_index query_pos key_pos"],
    hook: HookPoint,
    position: Union[int, List[int]],
    src_position: Optional[Union[int, List[int]]] = None,
    scale: Union[float, torch.Tensor] = 2,
):
    if src_position is not None:
        if isinstance(src_position, int):
            pattern[:, :, position, src_position] *= scale
        elif isinstance(src_position, list):
            src_position = np.array(src_position).reshape(-1, 1)
            pattern[:, :, position, src_position] *= scale
            
    else:
        pattern[:, :, position, :] = scale
    return pattern


# 
def attn_pattern_focus_myself_hook(
    pattern: Float[torch.Tensor, "batch head_index query_pos key_pos"],
    hook: HookPoint,
    position: Union[int, List[int]],
):
    pattern[:, :, position, :] = 0
    pattern[:, :, position, position] = 1
    return pattern



