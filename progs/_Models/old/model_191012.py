import sys
sys.path.append('./utils')
import myfunc as mf
sys.path.append('./11_Models')
sys.path.append('./11_Models/modules')

sys.path.append('./11_Models/feature_extractors')
# from MultiScaleResNet1D import MultiScaleResNet1D
from ResNet1D import ResNet1D
from EncRNN import EncRNN
from EncTransformer import EncTransformer
# from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm
sys.path.append('./11_Models/classifiers')
from MLP import MLP
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,\
                             RandomForestClassifier, VotingClassifier

sys.path.append('./11_Models/regressors')
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor,\
                             RandomForestRegressor, VotingRegressor

sys.path.append('./11_Models/generators')
from DecRNN import DecRNN

import torch
import torch.nn as nn
import torch.nn.functional as F

import socket
hostname = socket.gethostname()
if hostname == 'localhost.localdomain':
  from delogger import Delogger
  Delogger.is_debug_stream = True
  debuglog = Delogger.line_profiler




class Model(nn.Module):
    def __init__(self, input_size=1,
                       output_size_isRipple=2,
                       max_seq_len=1000,
                       samp_rate=1000,
                       hidden_size=32,
                       num_layers=4,
                       dropout_rnn=0,
                       dropout_fc=0,
                       bidirectional=True,
                       use_input_bn=True,
                       use_rnn=True,
                       use_cnn=True,
                       use_transformer=True,
                       transformer_d_model=256,
                       use_fft=True,
                       use_wavelet_scat=True,
                       rnn_archi='gru',
                       ):
        super(Model, self).__init__()

        self.samp_rate = samp_rate

        self.dropout_fc = nn.Dropout(p=dropout_fc)

        # Input BN
        self.use_input_bn = use_input_bn
        if use_input_bn:
          self.input_bn = nn.BatchNorm1d(max_seq_len)

        ########## Feature Extractors ##########
        last_shared_features_size = 0

        # Encoder RNN
        self.use_rnn = use_rnn
        if use_rnn:
          self.enc_rnn = EncRNN(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=bidirectional,
                                rnn_archi=rnn_archi,
                                )
          last_shared_features_size += self.enc_rnn.outsize

        # Encoder CNN
        self.use_cnn = use_cnn
        if use_cnn:
          self.cnn = ResNet1D(input_size)
          last_shared_features_size += self.cnn.outsize # 128

        # Encoder Transformer
        self.use_transformer = use_transformer
        if use_transformer:
          self.enc_tf = EncTransformer(input_size, seq_len=max_seq_len, d_model=transformer_d_model, dim_feedforward=256)
          last_shared_features_size += self.enc_tf.outsize

        # FFT
        self.use_fft = use_fft
        if use_fft:
          last_shared_features_size += int(max_seq_len/2)

        # Wavelet Scattering
        self.use_wavelet_scat = use_wavelet_scat
        if use_wavelet_scat:
          last_shared_features_size += 336 # Fixme

        ########## Adaptor ##########
        # Shared FC
        self.fc_shared = nn.Sequential(nn.Linear(last_shared_features_size, int(last_shared_features_size/2)),
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_fc)
                                      )
                     # MLP(last_shared_features_size, [int(last_shared_features_size/2)),], [nn.ReLU(),], [nn.Dropout(p=dropout_fc),])

        ########## Regressors ##########
        # Specific FCs
        # self.fc_dur = nn.Sequential(nn.Linear(int(last_shared_features_size/2), int(last_shared_features_size/4)),
        #                             nn.ReLU(),
        #                             nn.Dropout(p=dropout_fc),
        #                             nn.Linear(int(last_shared_features_size/4), 2)
        #                            )
        # self.fc_lat = nn.Sequential(nn.Linear(int(last_shared_features_size/2), int(last_shared_features_size/4)),
        #                             nn.ReLU(),
        #                             nn.Dropout(p=dropout_fc),
        #                             nn.Linear(int(last_shared_features_size/4), 2)
        #                            )

        # ########## Classifiers ##########
        # self.fc_isn = nn.Sequential(nn.Linear(int(last_shared_features_size/2), int(last_shared_features_size/4)),
        #                             nn.ReLU(),
        #                             nn.Dropout(p=dropout_fc),
        #                             nn.Linear(int(last_shared_features_size/4), 2)
        #                            )


        self.fc_isRipple = nn.Sequential(nn.Linear(int(last_shared_features_size/2), output_size_isRipple),
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout_fc),
                                         nn.Linear(last_shared_features_size-int(size_d/4), last_shared_features_size-int(size_d/2)),
                                        )

    def collect_features(self, last_shared_features, new_features):
        if last_shared_features is None:
          last_shared_features = new_features
        else:
          last_shared_features = torch.cat((last_shared_features, new_features), dim=-1)
        return last_shared_features

    # @Delogger.line_memory_profiler
    def forward(self, inp): # inp: [batch, seq_len, n_features]
        n_features = inp.shape[-1]
        last_shared_features = None

        # input dropout
        # inp = self.dropout_fc(inp) # fixme

        # Batch Normalization
        if self.use_input_bn:
          inp = self.input_bn(inp)

        ########## Feature Extractors ##########
        ### Trainable ###
        # Encoder
        if self.use_rnn:
          enc_rnn_out, enc_rnn_h_n = self.enc_rnn(inp)
          last_shared_features = self.collect_features(last_shared_features, enc_rnn_out)

        # CNN
        if self.use_cnn:
          cnn_out = self.cnn(inp)
          last_shared_features = self.collect_features(last_shared_features, cnn_out)

        # Transformer
        if self.use_transformer:
          enc_tf_out = self.enc_tf(inp) # fixme
          last_shared_features = self.collect_features(last_shared_features, enc_tf_out)

        ### Untrainable ###
        # FFT
        if self.use_fft:
          sp_powers = mf.fft_torch(inp)
          last_shared_features = self.collect_features(last_shared_features, sp_powers.to(inp.dtype))

        # Wavelet Scattering
        if self.use_wavelet_scat:
          sx = mf.wavelet_scattering_1D(inp)
          last_shared_features = self.collect_features(last_shared_features, sx.to(inp.dtype))

        ########## Adaptor ##########
        # Shared FC
        shared_fc_output = self.fc_shared(last_shared_features)
        shared_fc_output = self.dropout_fc(shared_fc_output)
        shared_fc_output = F.relu(shared_fc_output)

        ########## Regressors ##########
        # dur_hat = self.fc_dur(shared_fc_output)
        # dur_hat_mu = dur_hat[:,0]
        # dur_hat_sigma = dur_hat[:,1]

        # lat_logn_hat = self.fc_lat(shared_fc_output)
        # # lat_logn_hat = torch.clamp(lat_logn_hat, min=1e-3, max=1e3)
        # lat_logn_hat_mu = lat_logn_hat[:,0]
        # lat_logn_hat_sigma = lat_logn_hat[:,1]

        # ########## Classifiers ##########
        # isn_score_hat = self.fc_isn(shared_fc_output)
        isRipple_logits = self.fc_isRipple(shared_fc_output)

        # return dur_hat_mu, dur_hat_sigma, lat_logn_hat_mu, lat_logn_hat_sigma, shared_fc_output
        return isRipple_logits
