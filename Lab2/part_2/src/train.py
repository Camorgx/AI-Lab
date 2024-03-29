#!/usr/bin/env python3
import math
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as func
from tqdm import tqdm
from typing import List


class CharTokenizer:
    """
    a very simple char-based tokenizer. the tokenizer turns a string into a list of integers.
    """

    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.n_vocab = len(corpus)
        self.encode_dict = dict()
        self.decode_dict = dict()
        index = 0
        for word in corpus:
            self.encode_dict[word] = index
            self.decode_dict[index] = word
            index += 1
        # End of your code

    def encode(self, string: str):
        return list(map(lambda x: self.encode_dict[x], string))
        # End of your code

    def decode(self, codes: List[int]):
        return ''.join(list(map(lambda x: self.decode_dict[x], codes)))
        # End of your code


class Head(nn.Module):
    """single head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.Key = nn.Linear(n_embd, head_size, bias=False)
        self.Query = nn.Linear(n_embd, head_size, bias=False)
        self.Value = nn.Linear(n_embd, head_size, bias=False)
        # End of your code
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size), dtype=torch.bool)))

    def forward(self, inputs, mask=None):
        """
        input: tensor(batch, block_size, n_embd)
        return out: tensor(batch, block_size, head_size)
        """
        q = self.Query(inputs)
        k = self.Key(inputs)
        v = self.Value(inputs)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
        seq_mask = torch.tril(torch.ones((scores.shape[1], scores.shape[2]), dtype=torch.bool)).to(device)
        if mask is None:
            mask = seq_mask
        else:
            mask |= seq_mask
        scores.masked_fill(mask, -1e9)
        attention = nn.Softmax(dim=-1)(scores)
        out = torch.matmul(attention, v)
        # End of your code
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.linear = nn.Linear(n_heads * head_size, n_embd)
        self.layer_norm = nn.LayerNorm(n_embd)
        # End of your code

    def forward(self, inputs, mask=None):
        batch = inputs.shape[0]
        residual = inputs
        attentions = torch.stack([head(inputs, mask) for head in self.heads])
        attentions = attentions.permute(1, 2, 0, 3)
        attentions = attentions.transpose(1, 2).contiguous().view(batch, -1, n_heads * self.head_size)
        attentions = self.linear(attentions)
        out = self.layer_norm(residual + attentions)
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_embd, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=n_embd, kernel_size=1)
        self.layer_norm = nn.LayerNorm(n_embd)
        # End of your code

    def forward(self, inputs):
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        out = self.layer_norm(output + residual)
        return out


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_embd // n_heads)
        self.feed_forward = FeedForward()
        # End of your code

    def forward(self, inputs, mask=None):
        attention = self.self_attention(inputs, mask)
        inputs = self.feed_forward(attention)
        # End of your code
        return inputs


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        def get_encoding_table(n_position):
            def cal_angle(position, hid_idx):
                return position / np.power(10000, 2 * (hid_idx // 2) / n_embd)

            def get_angle_vec(position):
                return [cal_angle(position, hid_j) for hid_j in range(n_embd)]

            sinusoid_table = np.array([get_angle_vec(pos_i) for pos_i in range(n_position)])
            # dim 2i
            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
            # dim 2i+1
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
            return torch.FloatTensor(sinusoid_table)

        self.emb = nn.Embedding(n_vocab, n_embd)
        self.pos_emb = get_encoding_table(block_size + 1)
        self.layers = nn.ModuleList([Block() for _ in range(n_layers)])
        self.linear = nn.Linear(n_embd, n_vocab, bias=False)
        # End of your code

    def forward(self, inputs, labels=None, mask=None):
        """
        inputs: tensor(batch_size, context)
        """
        # embedding:(batch, context, embedding)
        batch, context = inputs.shape
        attention = self.emb(inputs) + torch.stack([self.pos_emb[:context]] * batch).to(device)
        for layer in self.layers:
            attention = layer(attention, mask)
        logits = self.linear(attention)
        # End of your code

        # compute the loss
        if labels is None:
            loss = None
        else:
            batch, time, channel = logits.shape
            logits = logits.view(batch * time, channel)
            labels = labels.view(batch * time)
            loss = func.cross_entropy(logits, labels)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                padding = inputs + [0] * (block_size - len(inputs))
                mask = [[False] * len(inputs) + [True] * (block_size - len(inputs))] * block_size
                mask = torch.tensor(mask, dtype=torch.bool, device=device)
                tensor = torch.stack([torch.tensor(padding)]).to(device)
                logits = self(tensor, mask=mask)[0][0][len(inputs) - 1]
                softmax = func.softmax(logits, dim=-1)
                res = random.choices(range(n_vocab), weights=softmax, k=1)[0]
                inputs.append(res)
        # End of your code
        return inputs


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


def generate(model, prefix):
    context = encode(prefix)
    print(decode(model.generate(context, max_new_tokens=200)))


def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    record = []

    for it in tqdm(range(max_iters)):

        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"\nstep {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            record.append([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}"])

        inputs, labels = get_batch("train")

        logits, loss = model(inputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(model, f'../output/model_{max_iters}.pth')
    pd.DataFrame(record, columns=['iter', 'train loss', 'val loss'])\
        .to_csv(f'../output/loss_{max_iters}.csv', index=False)


# define the hyperparameters
batch_size = 16
block_size = 256
max_iters = 10000  # set the number of training iterations as you like
eval_interval = 200
learning_rate = 1e-4
device = "cpu" if not torch.cuda.is_available() else "cuda"
eval_iters = 200
n_embd = 64  # d_model
n_heads = 8
n_layers = 6

# read the dataset
with open("../data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))

# initialize the vocabulary
tokenizer = CharTokenizer(chars)
encode = tokenizer.encode
decode = tokenizer.decode
n_vocab = tokenizer.n_vocab

# separate the dataset into train and validation
train_data = torch.tensor(encode(text[: -len(text) // 10]), dtype=torch.long)
val_data = torch.tensor(encode(text[-len(text) // 10:]), dtype=torch.long)

if __name__ == '__main__':
    # define the model
    transformer = Transformer().to(device)
    train(transformer)
    generate(transformer, "What studied torments, tyrant, hast")
