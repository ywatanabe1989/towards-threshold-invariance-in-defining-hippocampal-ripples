import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm
import math

def fc_block(n_in, n_out, activation, dropout, *args, **kwargs):
    return nn.Sequential(
        nn.Linear(n_in, n_out, *args, **kwargs),
        activation(),
        nn.Dropout(p=dropout)
        )

class MLP(nn.Module):
  def __init__(self, n_in, n_outs, acts, dropouts,):
    super(MLP, self).__init__()
    n_nodes = [n_in] + n_outs
    # fc_blocks = nn.ModuleList([fc_block(n_nodes[i], n_nodes[i+1], acts[i], dropouts[i]) \
    #              for i in range(len(n_outs)) ])
    fc_blocks = [fc_block(n_nodes[i], n_nodes[i+1], acts[i], dropouts[i]) \
                 for i in range(len(n_outs)) ]
    self.fc = nn.Sequential(*fc_blocks)

  def forward(self, x):
    return self.fc(x)

class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe_tensor = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = x + pe_tensor
        return self.dropout(x), pe_tensor

class EncTransformer(nn.Module):
  def __init__(self, n_in, seq_len, quasi_tau=3, d_model=512, nhead=8, dim_feedforward=2048, dropout=0, n_enc_layers=6):
    super().__init__()
    self.outsize = d_model
    self.quasi_tau = quasi_tau
    self.adjuster = MLP(quasi_tau, [d_model,], [nn.ReLU,], [dropout,])
    self.pe = PositionalEncoding(d_model, dropout, max_len=seq_len)
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
    encoder_norm = LayerNorm(d_model)
    self.encoder = TransformerEncoder(encoder_layer, n_enc_layers, encoder_norm)
    self.gap = torch.mean

  def quasi_attractor_transformation(self, x, tau=3): # data [bs, seq_len, n_features=1]
      qa = []
      for i in range(tau):
        qa.append(x[:, i:-tau+i])
      qa = torch.stack(qa).transpose(0,-1).squeeze()
      return qa

  def forward(self, x):
    x = self.quasi_attractor_transformation(x, tau=self.quasi_tau)
    x = self.adjuster(x)
    x, _ = self.pe(x)
    x = self.encoder(x)
    x = self.gap(x, dim=1)
    return x


'''
# Generate 1D data (as Time Series One)
bs = 64
seq_len = 1024
n_features = 1
d_model = 256
dim_feedforward = 256
data = torch.randn(bs, seq_len, n_features)
enc_transformer = EncTransformer(n_features, seq_len=seq_len, d_model=d_model, nhead=8, dim_feedforward=2048, dropout=0, n_enc_layers=6)
out = enc_transformer(data)

# Quasi Attractor
tau = 3
quasi_attractor = get_quasi_attractor(data, tau=tau)

# Adjuster
adjuster = MLP(tau, [d_model,], [nn.ReLU,], [0.5,])
data = adjuster(quasi_attractor)

# Positional Encoding
pe = PositionalEncoding(d_model, 0, max_len=seq_len) # fixme, scale
out, pe_tensor = pe(data)
import seaborn as sns
ax = sns.heatmap(pe_tensor.numpy().squeeze())
ax = sns.heatmap(out.numpy().squeeze())
'''
