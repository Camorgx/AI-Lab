#!/usr/bin/env python3
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
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
        for word in corpus:
            num = ord(word)
            self.encode_dict[word] = num
            self.decode_dict[num] = word
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
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, inputs, mask):
        """
        input: tensor(batch, time, n_embd)
        return out: tensor(batch, time, head_size)
        """
        q = self.Query(inputs)
        k = self.Key(inputs)
        v = self.Value(inputs)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
        scores.masked_fill_(mask, -1e9)
        attention = nn.Softmax(dim=-1)(scores)
        out = torch.matmul(attention, v)
        # End of your code
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.linear = nn.Linear(n_heads * head_size, n_embd)
        self.layer_norm = nn.LayerNorm(n_embd)
        # End of your code

    def forward(self, inputs):
        attentions = [head(inputs) for head in self.heads]
        attentions = torch.cat(attentions, dim=1)
        attentions = self.linear(attentions)
        out = self.layer_norm(attentions)
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

    def forward(self, inputs):
        attention = self.self_attention(inputs)
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

        self.pos_emb = get_encoding_table(block_size + 1)
        self.layers = nn.ModuleList([Block() for _ in range(n_layers)])
        self.linear = nn.Linear(n_embd, n_vocab)
        # End of your code

    def forward(self, inputs, labels=None):
        """
        inputs: tensor(batch_size, block_size)
        """
        # embedding:(batch, context, embedding)
        attention = inputs + torch.stack([self.pos_emb] * batch_size)
        for layer in self.layers:
            attention = layer(attention)
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
        for _ in range(max_new_tokens):
            inputs = self(inputs)
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


def generate(model):
    context = torch.zeros((1, 1), device=device, dtype=torch.long)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))


def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for it in range(max_iters):

        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        inputs, labels = get_batch("train")

        logits, loss = model(inputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


# define the hyperparameters
batch_size = 16
block_size = 256
max_iters = 5000  # set the number of training iterations as you like
eval_interval = 50
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
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

# define the model
transformer = Transformer().to(device)
train(transformer)
generate(transformer)
