import torch.nn as nn
import torch

class BaseModel(nn.Module):
    def __init__(self, word_vocab_size, output_size):
        super(BaseModel, self).__init__()
        ### START YOUR CODE ###
        pass
        ### END YOUR CODE ###
    
    def forward(self, x):
        ### START YOUR CODE ###
        pass
        ### END YOUR CODE ###
        return x


class WordPOSModel(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, output_size):
        super(WordPOSModel, self).__init__()
        ### START YOUR CODE ###
        pass
        ### END YOUR CODE ###

    def forward(self, x):
        ### START YOUR CODE ###
        pass
        ### END YOUR CODE ###
        return x
