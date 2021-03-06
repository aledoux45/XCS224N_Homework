#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.vocab = vocab
        self.embed_size = embed_size
        self.char_embed_size = 50
        self.dropout = nn.Dropout(0.3)

        self.char_embedding  = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx=vocab.char2id['<pad>'])
        
        self.highway = Highway(self.embed_size)
        self.cnn = CNN(self.char_embed_size, self.embed_size, kernel_size=5)
        
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        
        input = (sentence_length, batch_size, max_word_length)
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        char_embeddings = self.char_embedding(input) # sentence_length, batch_size, max_word_length,
        seq_len, batch_size, max_word_len, _ = char_embeddings.shape
        x = char_embeddings.view(seq_len * batch_size, max_word_len, self.char_embed_size).permute(0,2,1)

        # (batch_size, char_embed_size, seq_length)
        x_conv = self.cnn(x) # cnn input = (batch_size, embed_size, seq_length)
        x_highway = self.highway(x_conv) # x_highway = (batch_size, word_embed_size)
        out = self.dropout(x_highway)
        out = out.view(seq_len, batch_size, self.embed_size)
        return out
        ### END YOUR CODE
