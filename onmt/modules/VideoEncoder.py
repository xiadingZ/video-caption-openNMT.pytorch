import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class VideoEncoder(nn.Module):
    """
    A simple encoder convolutional -> recurrent neural network for
    image input.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.
        dim_vid (int): dimmension of video features inpur to encoder
    """
    def __init__(self, num_layers, bidirectional, rnn_size, dropout, dim_vid=2048):
        super(VideoEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.dim_hidden = rnn_size
        self.linear = nn.Linear(dim_vid, self.dim_hidden)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(self.dim_hidden, self.dim_hidden,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)

    def load_pretrained_vectors(self, opt):
        # Pass in needed options only when modify function definition.
        pass

    def forward(self, input, lengths=None):
        "See :obj:`onmt.modules.EncoderBase.forward()`"

        batch_size, seq_len, dim_vid = input.size()
        input = self.dropout(self.linear(input.view(-1, dim_vid))).view(batch_size, seq_len, self.dim_hidden)
        input = input.transpose(0, 1)
        # seq_len, batch_size, dim_hidden
        out, hidden_t = self.rnn(input)

        return hidden_t, out
