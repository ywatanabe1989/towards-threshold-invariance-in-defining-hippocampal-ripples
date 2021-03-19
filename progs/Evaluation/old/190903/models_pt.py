import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from sru import SRU
from torchqrnn import QRNN

## CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if  use_cuda else "cpu")
# torch.backends.cudnn.enabled = False # fixme
torch.backends.cudnn.benchmark = True
n_gpus = int(torch.cuda.device_count())


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        in_size = input.size(0) # batch_size
        h_t = torch.zeros(in_size, 51).to(device) #, dtype=torch.float)
        c_t = torch.zeros(in_size, 51).to(device)
        h_t2 = torch.zeros(in_size, 51).to(device)
        c_t2 = torch.zeros(in_size, 51).to(device)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)): # input_t.shape -> [97,1]
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

class Model1_(nn.Module):
    def __init__(self):
        super(Model1_, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        in_size = input.size(0) # batch_size
        h_t = Variable(torch.zeros(in_size, 51)).cuda()
        c_t = Variable(torch.zeros(in_size, 51)).cuda()
        h_t2 = Variable(torch.zeros(in_size, 51)).cuda()
        c_t2 = Variable(torch.zeros(in_size, 51)).cuda()

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)): # input_t.shape -> [97,1]
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

class Model2(nn.Module): # BiRNN
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        # input_size: the number of input features in the input x
        super(Model2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

    def forward(self, x, future=0):
        outputs = [] # added for applying classification model to AR model
        in_size = x.size(0) # added
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        for i, x_t in enumerate(x.chunk(x.size(1), dim=1)): #?, x_t -> [97,1]
          x_t = x_t.unsqueeze(1)
          out, _ = self.lstm(x_t, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
          out = self.fc(out[:, -1, :])
          outputs += [out]
        for i in range(future): # if we should predict the future
          # Forward propagate LSTM
          out =  out.unsqueeze(1)
          out, _ = self.lstm(out, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
          # Decode the hidden state of the last time step
          out = self.fc(out[:, -1, :])
          outputs += [out]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

class Model3(nn.Module):
    def __init__(self, num_layers=2):
        super(Model3, self).__init__()
        self.num_layers = num_layers #

        self.lstm1_fw = nn.LSTMCell(1, 51)
        self.lstm2_fw = nn.LSTMCell(51, 51)
        self.linear_fw = nn.Linear(51, 1)

        self.lstm1_bw = nn.LSTMCell(1, 51)
        self.lstm2_bw = nn.LSTMCell(51, 51)
        self.linear_bw = nn.Linear(51, 1)

        self.linear = nn.Linear(2, 1)
        # self.linear = nn.Linear(51*2, 1)

    def forward(self, input, future = 0):
        outputs = []
        in_size = input.size(0) # batch_size

        h_t1_fw = torch.zeros(in_size, 51).to(device)
        c_t1_fw = torch.zeros(in_size, 51).to(device)
        h_t2_fw = torch.zeros(in_size, 51).to(device)
        c_t2_fw = torch.zeros(in_size, 51).to(device)

        h_t1_bw = torch.zeros(in_size, 51).to(device)
        c_t1_bw = torch.zeros(in_size, 51).to(device)
        h_t2_bw = torch.zeros(in_size, 51).to(device)
        c_t2_bw = torch.zeros(in_size, 51).to(device)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)): # input_t.shape -> [97,1]
            input_t_fw = input_t
            h_t1_fw, c_t1_fw = self.lstm1_fw(input_t_fw, (h_t1_fw, c_t1_fw))
            h_t2_fw, c_t2_fw = self.lstm2_fw(h_t1_fw, (h_t2_fw, c_t2_fw))

            input_t_bw = torch.flip(input_t, [1])
            h_t1_bw, c_t1_bw = self.lstm1_bw(input_t_bw, (h_t1_bw, c_t1_bw))
            h_t2_bw, c_t2_bw = self.lstm2_bw(h_t1_bw, (h_t2_bw, c_t2_bw))

            output_fw = self.linear_fw(h_t2_fw)
            output_bw = self.linear_bw(h_t2_bw)
            output = self.linear(torch.cat((output_fw, output_bw), 1))
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t1_fw, c_t1_fw = self.lstm1_fw(output_fw, (h_t1_fw, c_t1_fw))
            h_t2_fw, c_t2_fw = self.lstm2_fw(h_t1_fw, (h_t2_fw, c_t2_fw))
            output_fw = self.linear_fw(h_t2_fw)

            h_t1_bw, c_t1_bw = self.lstm1_bw(output_bw, (h_t1_bw, c_t1_bw))
            h_t2_bw, c_t2_bw = self.lstm2_bw(h_t1_bw, (h_t2_bw, c_t2_bw))
            output_bw = self.linear_fw(h_t2_bw)

            output = self.linear(torch.cat((output_fw, output_bw), 1))
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

class Model4(nn.Module): # EncoderBiRNN, input_size = n_features
    def __init__(self, input_size, hidden_size=32, n_class=10, num_layers=4, dropout=0, bidirectional=True):
        super(Model4, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.n_class = n_class
        # self.bi_gru = nn.GRU(input_size, self.hidden_size, num_layers=self.num_layers, \
        #                      batch_first=True, bidirectional=True, dropout=self.dropout)
        self.bi_gru = nn.GRU(input_size, self.hidden_size, num_layers=self.num_layers, \
                             batch_first=True, bidirectional=True, dropout=self.dropout)
        self.fc = nn.Linear((1+self.bidirectional)*self.hidden_size, n_class)
        self.softmax = F.softmax

    def forward(self, input, h_n=None):
        # input: (batch, seq, feature)
        # The input can also be a packed variable length sequence.
        # See torch.nn.utils.rnn.pack_padded_sequence() for details.
        # h_0 = None
        ### self.bi_gru.flatten_parameters() # for parallel computing
        output, h_n = self.bi_gru(input, h_n) # [(1+num_directions)*num_layers, ?, hidden_size]
        # output: [batch, seq_len, num_directions(=2)*hidden_size]
        # tensor containing the output features h_t from the last layer of the GRU, for each t.
        # If a torch.nn.utils.rnn.PackedSequence has been given as the input,
        # the output will also be a packed sequence.
        # For the unpacked case, the directions can be separated using
        # output.view(seq_len, batch, num_directions, hidden_size),
        # with forward and backward being direction 0 and 1 respectively.

        # h_n: (num_layers * num_directions, batch, hidden_size)
        # Like output, the layers can be separated using
        # h_n.view(num_layers, num_directions, batch, hidden_size).
        last_output = output[:, -1, :] # last_output: [batch_size, (1+bidirectional)*hidden_size]

        class_hat = self.softmax(self.fc(last_output), dim=1)
        return output, class_hat, h_n

    def backward(self, ):
        pass


class EncBiGRU_varlen(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, n_class=10, num_layers=4, \
                                 dropout=0, bidirectional=True):
        super(EncBiGRU_varlen, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.n_class = n_class
        self.batch_first = True
        ### layer_norm is not implemented.  ###
        self.bi_gru = nn.GRU(input_size, self.hidden_size, num_layers=self.num_layers, \
                             batch_first=self.batch_first, bidirectional=True, dropout=self.dropout)
        self.fc_c = nn.Linear((1+self.bidirectional)*self.hidden_size, n_class)
        self.fc_d = nn.Linear((1+self.bidirectional)*self.hidden_size, 1)
        self.softmax = F.softmax

    # https://discuss.pytorch.org/t/batched-index-select/9115/7
    def batched_index_select(self, input, dim, index):
      views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
      expanse = list(input.shape)
      expanse[0] = -1
      expanse[dim] = -1
      index = index.view(views).expand(expanse)
      return torch.gather(input, dim, index).squeeze()

    def forward(self, input, lengths, h_n=None):
        self.bi_gru.flatten_parameters()
        # max_len = lengths.max()
        total_length = input.size(1)
        input = nn.utils.rnn.pack_padded_sequence(input, lengths.squeeze().cpu(), \
                                                                                    batch_first=self.batch_first)
        output, h_n = self.bi_gru(input, h_n)
        output, input_sizes = nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first, \
                                                                                                        total_length=total_length)
        # Extract the outputs for the last timestep of each example
        idx = (lengths -1).cuda()
        # extract last outputs from variable length inputs
        self.context_vec = self.batched_index_select(output, 1, idx)
        # Validation
        # for i in range(10):
        #   print(output[i, lengths[i] -1] == context_vec[i])
        class_hat = self.softmax(self.fc_c(self.context_vec), dim=1)
        dur_hat = self.fc_d(self.context_vec)
        return class_hat, dur_hat

class EncBiGRU_pack(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, n_class=10, num_layers=4, \
                                 dropout=0, bidirectional=True):
        super(EncBiGRU_pack, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.n_class = n_class
        self.batch_first = True
        ### layer_norm is not implemented.  ###
        self.bi_gru = nn.GRU(input_size, self.hidden_size, num_layers=self.num_layers, \
                             batch_first=self.batch_first, bidirectional=True, dropout=self.dropout)
        self.bi_gru.flatten_parameters()
        self.fc_c = nn.Linear((1+self.bidirectional)*self.hidden_size, n_class)
        self.fc_d = nn.Linear((1+self.bidirectional)*self.hidden_size, 1)
        self.softmax = F.softmax

    # https://discuss.pytorch.org/t/batched-index-select/9115/7
    def batched_index_select(self, input, dim, index):
      views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
      expanse = list(input.shape)
      expanse[0] = -1
      expanse[dim] = -1
      index = index.view(views).expand(expanse)
      return torch.gather(input, dim, index).squeeze()

    def forward(self, packed, lengths, h_n=None):
        total_length = max(lengths)
        output, h_n = self.bi_gru(packed, h_n)
        output, input_sizes = nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first, \
                                                                                                        total_length=total_length)
        # Extract the outputs for the last timestep of each example
        idx = (lengths -1).cuda()
        # extract last outputs from variable length inputs
        self.context_vec = self.batched_index_select(output, 1, idx)
        # Validation
        # for i in range(10):
        #   print(output[i, lengths[i] -1] == context_vec[i])
        class_hat = self.softmax(self.fc_c(self.context_vec), dim=1)
        dur_hat = self.fc_d(self.context_vec)
        return class_hat, dur_hat

class EncBiSRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=4, dropout=0, bidirectional=True):
        super(EncBiSRU, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.batch_first = False
        # SRU -> batch_first is not impremented. [seq, batch, input_dim]
        self.bi_sru = SRU(input_size, self.hidden_size, num_layers=self.num_layers, \
                          bidirectional=True, dropout=self.dropout, layer_norm=True)
        self.fc_d = nn.Linear((1+self.bidirectional)*self.hidden_size, 1)
        self.fc_l = nn.Linear((1+self.bidirectional)*self.hidden_size, 1)

    def forward(self, input, h_n=None):
        input = input.transpose(0,1) # -> [seq, batch, n_features]
        output, h_n = self.bi_sru(input, h_n) # input: [seq, batch, n_features]
        # output: [seq, batch, hidden_size * (1 + bidirectional)]
        self.context_vec = output[-1, :, :] # self.context_vec: [batch, hidden_size * (1 + bidirectional)]
        # class_hat = self.softmax(self.fc_c(self.context_vec), dim=1)
        dur_hat = self.fc_d(self.context_vec)
        lat_hat = self.fc_l(self.context_vec)
        return dur_hat, lat_hat, h_n

    def backward(self, ):
        pass

class EncBiSRU_binary(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=4, dropout=0, bidirectional=True):
        super(EncBiSRU_binary, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.batch_first = False
        # SRU -> batch_first is not impremented. [seq, batch, input_dim]
        self.bi_sru = SRU(input_size, self.hidden_size, num_layers=self.num_layers, \
                          bidirectional=True, dropout=self.dropout, layer_norm=True)
        self.fc_d = nn.Linear((1+self.bidirectional)*self.hidden_size, 1)
        self.fc_l = nn.Linear((1+self.bidirectional)*self.hidden_size, 1)
        self.fc_isnext = nn.Linear((1+self.bidirectional)*self.hidden_size, 2)
        self.softmax = F.softmax

    def forward(self, input, h_n=None):
        input = input.transpose(0,1) # -> [seq, batch, n_features]
        output, h_n = self.bi_sru(input, h_n) # input: [seq, batch, n_features]
        # output: [seq, batch, hidden_size * (1 + bidirectional)]
        self.context_vec = output[-1, :, :] # self.context_vec: [batch, hidden_size * (1 + bidirectional)]
        # class_hat = self.softmax(self.fc_c(self.context_vec), dim=1)
        dur_hat = self.fc_d(self.context_vec)
        lat_hat = self.fc_l(self.context_vec)
        isnext_score_hat = self.fc_isnext(self.context_vec)
        isnext_prob_hat = self.softmax(isnext_score_hat, dim=1)
        return dur_hat, lat_hat, isnext_prob_hat

    def backward(self, ):
        pass


class EncBiSRU_mnist(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, n_class=10, num_layers=4, dropout=0, bidirectional=True):
        super(EncBiSRU, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.n_class = n_class
        self.batch_first = False
        # SRU -> batch_first is not impremented. [seq, batch, input_dim]
        self.bi_sru = SRU(input_size, self.hidden_size, num_layers=self.num_layers, \
                             bidirectional=True, dropout=self.dropout, layer_norm=True)
        self.fc_c = nn.Linear((1+self.bidirectional)*self.hidden_size, n_class)
        self.fc_d = nn.Linear((1+self.bidirectional)*self.hidden_size, 1)
        self.softmax = F.softmax

    def forward(self, input, h_n=None):
        input = input.transpose(0,1) # -> [seq, batch, n_features]
        output, h_n = self.bi_sru(input, h_n) # input: [seq, batch, n_features]
        # output: [seq, batch, hidden_size * (1 + bidirectional)]
        self.context_vec = output[-1, :, :] # self.context_vec: [batch, hidden_size * (1 + bidirectional)]
        class_hat = self.softmax(self.fc_c(self.context_vec), dim=1)
        dur_hat = self.fc_d(self.context_vec)
        return output, class_hat, dur_hat, h_n
        # output: [batch*n_gpus, seq, hidden_size * (1 + bidirectional)]
        # class_hat: [batch, n_class]

    def backward(self, ):
        pass

class Dec(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Dec, self).__init__()
        self.hidden_size = hidden_size
        self.bi_sru = SRU(input_size, self.hidden_size, num_layers=self.num_layers, \
                             bidirectional=True, dropout=self.dropout, layer_norm=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.bi_sru(context_vector, hidden)
        last_output = output[-1, :, :]
        output = self.fc(last_output)
        return output


class EncDec(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncDec, self).__init__()
        self.enc = encoder
        self.dec = decoder

    def forward(self, input, h_n=None):
        enc_out, n_h = self.enc(input)
        encoded_vector = enc_out[:, -1, :] # [batch, seq, enc_hidden]
        dec_out = self.dec(enc_out)
        return dec_out


'''
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True, n_layers=2, dropout=0.1):
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.bidirectional = bidirectional
    self.n_layers = n_layers
    self.dropout = dropout

    self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional)
    # self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=bidirectional)

  def forward(self, inputs, hidden):
    # Note: we run this all at once (over multiple batches of multiple sequences)
    # packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, self.input_size)
    # packed == inputs.view(1, 1, self.input_size) ?

    output, hidden = self.lstm(inputs.view(1, 1, self.input_size), hidden)
    return output, hidden

  def init_hidden(self):
    return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
      torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size))

input_size = 1
hidden_size = 32
output_size = 1
enc = Encoder(input_size, hidden_size)
dec = AttentionDecoder(hidden_size, output_size)

class AttentionDecoder(nn.Module):

  def __init__(self, hidden_size, output_size):
    super(AttentionDecoder, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.attn = nn.Linear(hidden_size + output_size, 1)
    self.lstm = nn.LSTM(hidden_size, output_size)
    self.final = nn.Linear(output_size, output_size)

  def init_hidden(self):
    return (torch.zeros(1, 1, self.output_size),
      torch.zeros(1, 1, self.output_size))

  def forward(self, decoder_hidden, encoder_outputs, input):

    weights = []
    for i in range(len(encoder_outputs)):
      print(decoder_hidden[0][0].shape)
      print(encoder_outputs[0].shape)
      weights.append(self.attn(torch.cat((decoder_hidden[0][0],
                                          encoder_outputs[i]), dim = 1)))
    normalized_weights = F.softmax(torch.cat(weights, 1), 1)

    attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                             encoder_outputs.view(1, -1, self.hidden_size))

    input_lstm = torch.cat((attn_applied[0], input[0]), dim = 1) #if we are using embedding, use embedding of input here instead

    output, hidden = self.lstm(input_lstm.unsqueeze(0), decoder_hidden)

    output = self.final(output[0])

    return output, hidden, normalized_weights
'''

class Model6(nn.Module):
    def __init__(self, input_size, hidden_size=32, n_class=10, num_layers=4, dropout=0):
        super(Model6, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.n_class = n_class
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.qrnn = QRNN(self.hidden_size, self.hidden_size, num_layers=self.num_layers, \
                             dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, n_class)
        self.softmax = F.softmax

    def forward(self, input, h_n=None):
        input = self.fc1(input)
        input = input.transpose(0,1) # input: 392, 512, 1
        output, h_n = self.qrnn(input, h_n)
        last_output = output[-1, :, :] # last_output: [batch_size, (1+bidirectional)*hidden_size]

        class_hat = self.softmax(self.fc(last_output), dim=1)
        return output, class_hat, h_n # output: [seq*n_gpus, 512, 128], class_hat: [seq*n_gpus, n_class]

    def backward(self, ):
        pass
