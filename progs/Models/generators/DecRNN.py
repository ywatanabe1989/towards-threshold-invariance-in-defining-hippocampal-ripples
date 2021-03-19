import torch
import torch.nn as nn
import torch.nn.functional as F

class DecRNN(nn.Module): # Attn_Dec https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
  def __init__(self, input_size=1,
                     hidden_size=32,
                     num_layers=4,
                     bidirectional=True,
                     dropout_rnn=0,
                     dropout_fc=0,
                     rnn_archi='gru',
                     ):
    super(DecRNN, self).__init__()
    self.rnn_archi = rnn_archi
    self.outsize = 1

    if rnn_archi == 'gru':
      self.rnn = nn.GRU(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=False,
                        dropout=dropout_rnn,
                        bidirectional=True)
    if rnn_archi == 'lstm':
      self.rnn = LayerNormLSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional)

    if rnn_archi == 'sru':
      self.rnn = SRU(input_size=input_size,
                     hidden_size=hidden_size,
                     num_layers=num_layers,
                     bidirectional=bidirectional,
                     dropout=dropout_rnn,
                     layer_norm=True,
                     )

    self.dropout = nn.Dropout(p=dropout_fc)
    self.fc = nn.Linear((1+bidirectional)*hidden_size, self.outsize)

  def forward(self, input, h_n):
    # input = input.unsqueeze(0) # [1, batch]
    # input = F.relu(input)
    if self.rnn_archi == 'gru':
      self.rnn.flatten_parameters()
    output, h_n = self.rnn(input, h_n) # output [1, batch, (1+bidirectional)*hidden_size]
    # output = nn.ReLU(self.dropout(output))
    prediction = self.fc(output.squeeze(0))
    return prediction, h_n

  def initHidden(self):
      return torch.zeros(1, 1, self.hidden_size, device=device)
