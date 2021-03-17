import torch
import torch.nn as nn

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

'''
n_in = 10
bs = 64
dropout = 0.5

mlp = MLP(n_in, [100, 3], [nn.ReLU, nn.ReLU], [dropout, dropout])
data = torch.randn(bs, n_in)
print(mlp)

# train
mlp.train()
print(mlp(data))

mlp.eval()
print(mlp(data))
'''
