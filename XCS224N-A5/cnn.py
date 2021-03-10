#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, char_embed_size, word_embed_size, kernel_size=5):
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.word_embed_size = word_embed_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(char_embed_size, word_embed_size, self.kernel_size, stride=1, padding=0, bias=True)
        # self.max_pool_1d = nn.MaxPool1d(max_word_length - kernel_size + 1)
        
    def forward(self, x_reshaped):
        """
        Maps from x_reshaped to x_conv_out
        x_reshaped = (batch_size, char_embed_size, seq_length)
        x_conv_out = (batch_size, word_embed_size)
        """
        x_conv = torch.relu(self.conv(x_reshaped)) # x_reshaped = (batch_size, embed_size, seq_length)
        x_conv_out, _ = torch.max(x_conv, 2)
        # x_conv_out = self.max_pool_1d(torch.relu_(x_conv)).squeeze()
        return x_conv_out

### END YOUR CODE

