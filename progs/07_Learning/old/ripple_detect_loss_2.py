import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def mk_label(distances_ms, max_distances_ms=100):

  def N(t, N0=1, T=10):
    return N0 * (1/2) ** (t/T)

  clamped_distances_ms = torch.clamp(distances_ms, 0, max_distances_ms)
  return N(clamped_distances_ms)

def plot_distances_ms_probs(distances_ms, ripple_probs, title=None):
  plt.scatter(distances_ms, ripple_probs.detach().numpy())
  plt.xlabel('Distance [ms]')
  plt.ylabel('$\hat{Ripple Probability}$')
  plt.title(title)


def plot_probs_distances_ms_loss_detect_in_3D(ripple_probs, distances_ms, loss_detect, title=None):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(ripple_probs.detach().numpy(), distances_ms, loss_detect.detach().numpy())
  ax.set_xlabel('Ripple Probability')
  ax.set_ylabel('Distance [ms]')
  ax.set_zlabel('Loss_detect [a.u.]')
  plt.title(title)


bs = 64*32*4*10
n_classes = 1
sigmoid = nn.Sigmoid()

criterion = nn.BCEWithLogitsLoss() # Sigmoid + BinaryCrossEntropy for numerical stability

distances_ms = abs(torch.randn(bs))*100
labels = mk_label(distances_ms).unsqueeze(dim=-1)

'''
import pandas as pd
df = pd.DataFrame()

criterion = nn.BCELoss(reduction='none')

targets = torch.linspace(0, 1, 1000)
inputs = torch.linspace(0, 1, 1000)
bceloss = criterion(targets, inputs)

df['targets'] = targets.numpy()
df['inputs'] = inputs.numpy()
df['bceloss'] = bceloss.numpy()

df_pivot = pd.pivot_table(data=df, values='bceloss', columns='targets', index='inputs', aggfunc=np.mean)
sns.heatmap(df_pivot)
'''






## Random Simulation ##
ripple_scores_random = torch.randn(bs, n_classes, requires_grad=True)
ripple_probs_random = sigmoid(ripple_scores_random)
loss_detect_random = criterion(ripple_scores_random, labels)


plot_distances_ms_probs(distances_ms, ripple_probs_random,
                        title='(Random Simulation) Scatter Plot of Distances [ms] and $\hat{Ripple Probability}$')
plot_probs_distances_ms_loss_detect_in_3D(ripple_probs_random, distances_ms, loss_detect_random,
                                          title='Random Simulation (Loss_sum: {:.2f})'.format(loss_detect_random.sum()))


## After Learning Simulation ##
ripple_scores_learned = (-0.10*distances_ms + 0.1*torch.randn(bs, requires_grad=True) + 3).unsqueeze(dim=-1)
ripple_probs_learned = sigmoid(ripple_scores_learned)
loss_detect_learned = criterion(ripple_scores_learned, labels)

plot_distances_ms_probs(distances_ms, ripple_probs_learned,
                        title='(Learned Simulation) Scatter Plot of Distances [ms] and $\hat{Ripple Probability}$')
plot_probs_distances_ms_loss_detect_in_3D(ripple_probs_learned, distances_ms, loss_detect_learned,
                                          title='Learned Simulation (Loss_sum: {:.2f})'.format(loss_detect_learned.sum()))
















class RippleDetectLoss():
  def __init__(self, max_distances_ms=100):
    self.bce_with_logits_loss = nn.BCEWithLogitsLoss() # Sigmoid + BinaryCrossEntropy for numerical stability
    self.max_distances_ms = max_distances_ms

  def __call__(self, scores, distances_ms):
      target_all_one = torch.ones_like(scores)
      bce_loss = self.bce_with_logits_loss(scores, target_all_one)
      ripple_detect_loss = (1-self.lambda_1(distances_ms)) * bce_loss # fixme: How do you use lambda 1?
      return ripple_detect_loss

  def N(self, t, N0=1, T=10):
    return N0 * (1/2) ** (t/T)

  def lambda_1(self, distances_ms):
      clamped_distances_ms = torch.clamp(distances_ms, 0, self.max_distances_ms)
      return self.N(clamped_distances_ms)

def plot_distances_ms_lambda1_curve(distances_ms, lambda1):
  plt.scatter(distances_ms, lambda1)
  plt.xlabel('Distance [ms]')
  plt.ylabel('Lambda1')
  plt.title('Lambda1 = 1*(1/2)**(Distances[ms]/10)')
  plt.xlim([0, 150])





bs = 64*32*4*10
n_classes = 1
sigmoid = nn.Sigmoid()

criterion = RippleDetectLoss()
distances_ms = abs(torch.randn(bs))*100
lambda1 = criterion.lambda_1(distances_ms)
plot_distances_ms_lambda1_curve(distances_ms, lambda1)







# loss_detect.backward()




'''
## Check the BCELoss Implementation
bceloss = nn.BCELoss()
input = torch.tensor([0.])
target = torch.tensor([1.])
bceloss(input, target)

def mybceloss(input, target, eps=1e-12):
  return -(input*np.log(target+eps) + (1-input)*np.log(1-target+eps))
mybceloss(input, target)
'''












plt.scatter(probs[:, 1], loss_detect)
plt.xlabel('probs')
plt.ylabel('distances [ms]')

def N(t, N0=1, T=10):
  return N0 * (1/2) ** (t/T)

tau=10
t = np.arange(100)
n = N(t, T=tau)
plt.plot(t, n)
plt.title('Decay Curve (Half-life: Tau = {})'.format(tau))


# #############################################
# ## https://scipython.com/book/chapter-7-matplotlib/examples/lifetimes-of-an-exponential-decay/

# def plot_exponential_decay(N=10000, tau=28, tmax=100):
#   import numpy as np
#   import matplotlib.pyplot as plt

#   # Initial value of y at t=0, lifetime in s
#   # N, tau = 10000, 28
#   # Maximum time to consider (s)
#   # tmax = 100
#   # A suitable grid of time points, and the exponential decay itself
#   t = np.linspace(0, tmax, 1000)
#   y = N * np.exp(-t/tau)

#   fig = plt.figure()
#   ax = fig.add_subplot(111)
#   ax.plot(t, y)

#   # The number of lifetimes that fall within the plotted time interval
#   ntau = tmax // tau + 1
#   # xticks at 0, tau, 2*tau, ..., ntau*tau; yticks at the corresponding y-values
#   xticks = [i*tau for i in range(ntau)]
#   yticks = [N * np.exp(-i) for i in range(ntau)]
#   ax.set_xticks(xticks)
#   ax.set_yticks(yticks)

#   # xtick labels: 0, tau, 2tau, ...
#   xtick_labels = [r'$0$', r'$\tau$'] + [r'${}\tau$'.format(k) for k in range(2,ntau)]
#   ax.set_xticklabels(xtick_labels)
#   # corresponding ytick labels: N, N/e, N/2e, ...
#   ytick_labels = [r'$N$',r'$N/e$'] + [r'$N/{}e$'.format(k) for k in range(2,ntau)]
#   ax.set_yticklabels(ytick_labels)

#   ax.set_xlabel(r'$t\;/\mathrm{s}$')
#   ax.set_ylabel(r'$y$')
#   ax.grid()
#   plt.show()
# #############################################

# plot_exponential_decay()


# class RippleDetectLoss():
#   def __init__(self, ):
#     self.loss = nn.BCEWithLogitsLoss() # Sigmoid + BinaryCrossEntropy for numerical stability

#   def __call__(self, scores, distances_ms):
#       loss = self.lambda_0(distances_ms)*self.loss_0(scores) \
#            + self.lambda_1(distances_ms)*self.loss_1(scores)
#       return loss

#   def N(self, t, N0=1, T=10):
#     return N0 * (1/2) ** (t/T)

#   def lambda_0(self, distances_ms):
#       return 1 - self.lambda_1(distances_ms)

#   def lambda_1(self, distances_ms):
#       return self.N(distances_ms)

#   def loss_0(self, scores):
#       target_0 = torch.zeros_like(scores)
#       return self.loss(scores, target_0)

#   def loss_1(self, scores):
#       target_1 = torch.ones_like(scores)
#       return self.loss(scores, target_1)
