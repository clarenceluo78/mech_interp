import argparse, json
import random
import torch
import numpy as np
from transformers import (
    WEIGHTS_NAME,
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,

)

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 10]

# config = GPT2Config.from_pretrained("gpt2")
# VOCAB_SIZE = config.vocab_size


def model_preds(model, input_ids, input_mask, pos, tokenizer, foils=None, k=10, verbose=False):
    # Obtain model's top predictions for given input
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(model.device)
    softmax = torch.nn.Softmax(dim=0)
    A = model(input_ids[:, :pos], attention_mask=input_mask[:, :pos])
    probs = softmax(A.logits[0][pos-1])
    top_preds = probs.topk(k)
    if verbose:
        if foils:
            for foil in foils:
                print("Contrastive loss: ", A.logits[0][pos-1][input_ids[0, pos]] - A.logits[0][pos-1][foil])
                print(f"{np.round(probs[foil].item(), 3)}: {tokenizer.decode(foil)}")
        print("Top model predictions:")
        for p,i in zip(top_preds.values, top_preds.indices):
            print(f"{np.round(p.item(), 3)}: {tokenizer.decode(i)}")
    return top_preds.indices


## hook for every layer
def register_forward_hook(model, layer_id, embeddings_list):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().float().numpy())
    layer = model.model.layers[layer_id].input_layernorm
    handle = layer.register_forward_hook(forward_hook)
    return handle


# Adapted from AllenNLP Interpret and Han et al. 2020
def register_embedding_list_hook(model, embeddings_list):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().float().numpy())
    embedding_layer = model.model.embed_tokens
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle


# hook for every layer
def register_backward_hook(model, layer_id, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].detach().cpu().float().numpy())
    layer = model.model.layers[layer_id].input_layernorm
    hook = layer.register_backward_hook(hook_layers)
    return hook


def register_embedding_gradient_hooks(model, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].detach().cpu().float().numpy())
    embedding_layer = model.model.embed_tokens
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook


# get every hooked values layer wise
def register_hooks(model, num_layers):
    embeddings_list = [[] for _ in range(num_layers + 1)]
    gradients_list = [[] for _ in range(num_layers + 1)]

    # Embedding layer
    forward_hook = register_embedding_list_hook(model, embeddings_list[0])
    backward_hook = register_embedding_gradient_hooks(model, gradients_list[0])
    forward_hook.remove()
    backward_hook.remove()
    
    for i in range(num_layers - 1):
        forward_hook = register_forward_hook(model, i+1, embeddings_list[i+1])
        backward_hook = register_backward_hook(model, i+1, gradients_list[i+1])
        model.model.layers[i+1].input_layernorm.register_forward_hook(forward_hook)
        model.model.layers[i+1].input_layernorm.register_backward_hook(backward_hook)
    
    # Last layer
    forward_hook = register_forward_hook(model, num_layers - 1, embeddings_list)
    backward_hook = register_backward_hook(model, num_layers - 1, gradients_list)
    model.norm.register_forward_hook(forward_hook)
    model.norm.register_backward_hook(backward_hook)
    
    return embeddings_list, gradients_list


def compute_scores(embeddings_list, gradients_list, score_type='input_x_gradient'):
    num_layers = len(embeddings_list)
    num_tokens = embeddings_list[0][0].shape[0]  # Assuming shape is [num_tokens, embedding_dim]
    
    scores = np.zeros((num_tokens, num_layers))
    
    for layer_id in range(num_layers):
        embeddings = np.array(embeddings_list[layer_id]).squeeze()
        gradients = np.array(gradients_list[layer_id]).squeeze()
        
        if score_type == 'input_x_gradient':
            layer_scores = np.sum(embeddings * gradients, axis=-1)
        elif score_type == 'l1_grad_norm':
            layer_scores = np.linalg.norm(gradients, ord=1, axis=-1)
        else:
            raise ValueError("Unsupported score type")
        
        scores[:, layer_id] = layer_scores
    
    return scores


def saliency(model, tokens, batch=0, correct_id=None, foil_id=None):
    # Get model gradients and input embeddings
    torch.enable_grad()
    model.eval()
    embeddings_list = []
    handle = register_embedding_list_hook(model, embeddings_list)
    embeddings_gradients = []
    hook = register_embedding_gradient_hooks(model, embeddings_gradients)
    
    # if correct is None:
    #     correct = input_ids[-1]
    # input_ids = input_ids[:-1]
    # input_mask = input_mask[:-1]
    # input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    # input_mask = torch.tensor(input_mask, dtype=torch.long).to(model.device)

    model.zero_grad()
    print(tokens)
    A = model(**tokens)
    print(A.logits.shape)

    if foil_id is not None and correct_id != foil_id:
        (A.logits[0, -1][correct_id]-A.logits[-1][foil]).backward()
    else:
        (A.logits[0, -1][correct_id]).backward()
    handle.remove()
    hook.remove()

    return np.array(embeddings_gradients).squeeze(), np.array(embeddings_list).squeeze()


def input_x_gradient(grads, embds, normalize=False):
    input_grad = np.sum(grads * embds, axis=-1).squeeze()

    if normalize:
        norm = np.linalg.norm(input_grad, ord=1)
        input_grad /= norm
        
    return input_grad


def l1_grad_norm(grads, normalize=False):
    l1_grad = np.linalg.norm(grads, ord=1, axis=-1).squeeze()

    if normalize:
        norm = np.linalg.norm(l1_grad, ord=1)
        l1_grad /= norm
    return l1_grad


def erasure_scores(model, input_ids, input_mask, correct=None, foil=None, remove=False, normalize=False):
    model.eval()
    if correct is None:
        correct = input_ids[-1]
    input_ids = input_ids[:-1]
    input_mask = input_mask[:-1]
    input_ids = torch.unsqueeze(torch.tensor(input_ids, dtype=torch.long).to(model.device), 0)
    input_mask = torch.unsqueeze(torch.tensor(input_mask, dtype=torch.long).to(model.device), 0)
    
    A = model(input_ids, attention_mask=input_mask)
    softmax = torch.nn.Softmax(dim=0)
    logits = A.logits[0][-1]
    probs = softmax(logits)
    if foil is not None and correct != foil:
        base_score = (probs[correct]-probs[foil]).detach().cpu().numpy()
    else:
        base_score = (probs[correct]).detach().cpu().numpy()

    scores = np.zeros(len(input_ids[0]))
    for i in range(len(input_ids[0])):
        if remove:
            input_ids_i = torch.cat((input_ids[0][:i], input_ids[0][i+1:])).unsqueeze(0)
            input_mask_i = torch.cat((input_mask[0][:i], input_mask[0][i+1:])).unsqueeze(0)
        else:
            input_ids_i = torch.clone(input_ids)
            input_mask_i = torch.clone(input_mask)
            input_mask_i[0][i] = 0

        A = model(input_ids_i, attention_mask=input_mask_i)
        logits = A.logits[0][-1]
        probs = softmax(logits)
        if foil is not None and correct != foil:
            erased_score = (probs[correct]-probs[foil]).detach().cpu().numpy()
        else:
            erased_score = (probs[correct]).detach().cpu().numpy()
                    
        scores[i] = base_score - erased_score # higher score = lower confidence in correct = more influential input
    if normalize:
        norm = np.linalg.norm(scores, ord=1)
        scores /= norm
    return scores


def visualize(attention, tokenizer, input_ids, gold=None, normalize=False, print_text=True, save_file=None, title=None, figsize=60, fontsize=36):
    tokens = [tokenizer.decode(i) for i in input_ids[0][:len(attention) + 1]]
    if gold is not None:
        for i, g in enumerate(gold):
            if g == 1:
                tokens[i] = "**" + tokens[i] + "**"

    # Normalize to [-1, 1]
    if normalize:
        a,b = min(attention), max(attention)
        x = 2/(b-a)
        y = 1-b*x
        attention = [g*x + y for g in attention]
    attention = np.array([list(map(float, attention))])

    fig, ax = plt.subplots(figsize=(figsize,figsize))
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(attention, cmap='seismic', norm=norm)

    if print_text:
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=fontsize)
    else:
        ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for (i, j), z in np.ndenumerate(attention):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=fontsize)


    ax.set_title("")
    fig.tight_layout()
    if title is not None:
        plt.title(title, fontsize=36)
    
    if save_file is not None:
        plt.savefig(save_file, bbox_inches = 'tight',
        pad_inches = 0)
        plt.close()
    else:
        plt.show()


def main():
    pass


if __name__ == "__main__":
    main()
