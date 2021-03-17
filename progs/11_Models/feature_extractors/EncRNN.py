import torch
import torch.nn as nn
import torch.nn.functional as F
from sru import SRU
import sys
sys.path.append('../../modules')
from better_lstm import LSTM

class EncRNN(nn.Module):
  def __init__(self, input_size=1,
                     hidden_size=32,
                     num_layers=4,
                     dropout_rnn=0,
                     bidirectional=True,
                     rnn_archi='gru'
                     ):
    super(EncRNN, self).__init__()
    self.dropout = nn.Dropout(p=dropout_rnn)
    self.rnn_archi = rnn_archi

    if rnn_archi == 'gru':
      self.rnn = nn.GRU(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=False,
                        dropout=dropout_rnn,
                        bidirectional=True)

    if rnn_archi == 'lstm':
      # self.rnn = LayerNormLSTM(input_size=input_size,
      #                          hidden_size=hidden_size,
      #                          num_layers=num_layers,
      #                          bidirectional=bidirectional)
      self.rnn = LSTM(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      bidirectional=bidirectional,
                      dropouti=0.2,
                      dropoutw=0.2,
                      dropouto=0.5

      )

    if rnn_archi == 'sru':
      self.rnn = SRU(input_size=input_size,
                     hidden_size=hidden_size,
                     num_layers=num_layers,
                     bidirectional=bidirectional,
                     dropout=dropout_rnn,
                     layer_norm=True,
                     )

    self.outsize = (1+bidirectional) * hidden_size

  def forward(self, inp):
    inp = self.dropout(inp)
    inp = inp.transpose(0,1) # -> [seq, batch, n_features]
    if self.rnn_archi != 'sru':
      self.rnn.flatten_parameters()
    rnn_out, enc_h_n = self.rnn(inp, None)
    rnn_out = self.dropout(rnn_out)
    context_vec = rnn_out[-1, :, :] # context_vec: [batch, enc_rnn_outsize]
    return context_vec, enc_h_n

  def initHidden(self):
      return torch.zeros(1, 1, self.hidden_size, device=device)
