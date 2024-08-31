import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import tqdm.auto as tqdm
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
import plotly.express as px
import transformer_lens.utils as utils
import seaborn as sns


def imshow_px(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line_px(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter_px(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

def imshow(
    tensor,
    tokens=None,
    title=None,
    legend=False,
    colorbar=True,
    cmap='RdBu',
    norm=None,
    figsize=(8, 6),
    colorbar_height=1,
    num_in_block=True,
    xlabel="Layer",
    ylabel="Token",
    centered_colorbar=False,
):
    
    plt.figure(figsize=figsize)
    plt.imshow(utils.to_numpy(tensor), cmap=cmap, norm=norm)
    if tokens is not None:
        plt.xticks(np.arange(len(tokens)), tokens, rotation=90)
    if title is not None:
        plt.title(title)
    if legend:
        plt.legend()
    if colorbar:
        if centered_colorbar:
            max_val = np.abs(tensor.clone().cpu()).max()
            vmin = -max_val
            vmax = max_val
        else:
            vmin = tensor.min()
            vmax = tensor.max()
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_bad(color='gray')
        plt.imshow(utils.to_numpy(tensor), cmap=cmap, norm=norm)
        plt.colorbar(ticks=[vmin, 0, vmax], format='%.4f')
        plt.clim(vmin, vmax)
    if num_in_block:
        pass
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


def line(
    list
    ,xlabel="Layer"
    ,ylabel="Prob"
    ,title=None
    ,vline=True
    ,legend=True
):
    plt.plot(range(len(list)), list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend
    plt.title(title)
    if vline:
        for i in range(len(list)):
            plt.axvline(x=i, color='gray', linestyle='--')
    if legend:
        plt.legend()
    plt.show()


def hidden_states_projection(
    cache,
    tokens,
    tl_model,
    str_tokens,
    tokenizer=None,
    is_without_bos=0,
    use_cache=False,
    is_save=False,
    save_path=None,
    is_return=False,
):
    sequence_length = len(tokens[0])
    num_components = sequence_length
    num_layers = tl_model.cfg.n_layers
    norm = tl_model.ln_final
    unembed = tl_model.unembed

    if not use_cache:
        h_matrix_resid = np.empty((sequence_length, num_layers), dtype=object)
        for layer in tqdm.tqdm(range(num_layers)):
            
            for i in range(num_components):

                # i_logits = torch.matmul(norm(cache[f"blocks.{layer}.hook_resid_post"][i].float()), unembed_weight)
                
                hs = norm(cache[f"blocks.{layer}.hook_resid_post"][i].float())
                i_logits = unembed(hs.unsqueeze(0).unsqueeze(0))[0, 0]
                max_prob = max(torch.softmax(i_logits, dim=-1))
                meaning = tokenizer.decode(torch.softmax(i_logits, dim=-1).argmax(dim=-1))
                
                h_matrix_resid[i, layer] = {"prob": max_prob.cpu(), "meaning": meaning, "logits": i_logits.cpu()}

        prob_matrix_resid = np.zeros((sequence_length, num_layers))
        for i in range(sequence_length):
            for j in range(num_layers):
                prob_matrix_resid[i, j] = h_matrix_resid[i, j]["prob"]
    else:
        pass

    generated_tokens = str_tokens[is_without_bos:]  # get rid of <s>
    generated_tokens = [token.replace("Ġ", "") for token in generated_tokens]
    print(generated_tokens)

    cmap = sns.color_palette("Blues", as_cmap=True)

    plt.figure(figsize=(24, 8))
    plt.xlabel("Layer", fontsize=26)
    plt.ylabel("Tokens", fontsize=26)

    sns.heatmap(prob_matrix_resid[is_without_bos:, :], cmap=cmap, cbar_kws={"shrink": 0.8, "pad": 0.02, "aspect": 10})

    for i in range(is_without_bos, num_components):
        for j in range(num_layers):
            if prob_matrix_resid[i, j] > 0.2:
                plt.text(j + 0.5, i + 0.5 - is_without_bos, 
                        f"{h_matrix_resid[i, j]['meaning']}", 
                        ha="center", va="center", 
                        color="white", fontsize=10)

    plt.yticks(np.arange(is_without_bos, num_components), generated_tokens, rotation=20, fontsize=16)
    plt.xticks(fontsize=16)  # Increase the font size of the x-axis ticks

    plt.tight_layout()

    if is_save:
        plt.savefig(save_path, dpi=300)

    plt.show()

    if is_return:
        return h_matrix_resid, prob_matrix_resid