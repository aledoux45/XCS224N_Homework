#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1dimport torch

import torch
import torch.nn as nn


class Highway(nn.Module):
    def __init__(self, word_embed_size):
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        self.w_proj = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)
        self.w_gate = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)

    def forward(self, x_conv_out):
        """
        Maps from x_conv_out to x_highway
        x_conv_out = (batch_size, word_embed_size)
        x_highway = (batch_size, word_embed_size)
        """
        x_proj = torch.relu(self.w_proj(x_conv_out))
        x_gate = torch.sigmoid(self.w_gate(x_conv_out))

        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        # x_highway = torch.mul(x_proj, x_gate) + torch.mul(x_conv_out, 1 - x_gate)
        return x_highway

### END YOUR CODE 

