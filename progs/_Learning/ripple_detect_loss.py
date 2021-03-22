import torch
import torch.nn as nn

def get_decay_curve(t, N0=1, T=10): # N
  decay_curve = N0 * (1/2) ** (t/T)
  return decay_curve

def get_lambda(distances_ms, max_distance_ms=None):
    distances_ms[torch.isnan(distances_ms)] = max_distance_ms
    clamped_distances_ms = torch.clamp(distances_ms, 0, max_distance_ms)
    lam = get_decay_curve(clamped_distances_ms)
    lam /= lam.mean()
    return lam




# class RippleDetectLoss():
#   def __init__(self, max_distances_ms=100):
#     self.cross_entropy_loss = nn.CrossEntropyLoss() # nn.LogSoftmax() and nn.NLLLoss() for numerical stability
#     self.max_distances_ms = max_distances_ms

#   def __call__(self, scores, distances_ms):
#       lam = self.get_lambda(distances_ms)
#       loss_0 = self.get_loss_0(scores)
#       loss_1 = self.get_loss_1(scores)
#       ripple_detect_loss = lam*loss_0 + (1-lam)*loss_1
#       return ripple_detect_loss

#   def get_decay_curve(self, t, N0=1, T=10): # N
#     decay_curve = N0 * (1/2) ** (t/T)
#     return decay_curve

#   def get_lambda(self, distances_ms):
#       clamped_distances_ms = torch.clamp(distances_ms, 0, self.max_distances_ms)
#       lam = self.get_decay_curve(clamped_distances_ms)
#       return lam

#   def get_loss_0(self, scores):
#       target_0 = torch.zeros(len(scores), dtype=torch.long)
#       loss_0 = self.cross_entropy_loss(scores, target_0)
#       return loss_0

#   def get_loss_1(self, scores):
#       target_1 = torch.ones(len(scores), dtype=torch.long)
#       loss_1 = self.cross_entropy_loss(scores, target_1)
#       return loss_1

# def plot_distances_ms_lambda(distances_ms, lam):
#   plt.scatter(distances_ms, lam)
#   plt.xlabel('Distance [ms]')
#   plt.ylabel('Lambda')
#   plt.title('Lambda = 1*(1/2)**(Distances[ms]/10)')
#   plt.xlim([0, 150])

# def plot_distances_ms_probs(distances_ms, ripple_probs, title=None):
#   plt.scatter(distances_ms, ripple_probs.detach().numpy())
#   plt.xlabel('Distance [ms]')
#   plt.ylabel('$\hat{Ripple Probability}$')
#   plt.title(title)

# def plot_probs_distances_ms_loss_detect_in_3D(ripple_probs, distances_ms, loss_detect, title=None):
#   import matplotlib.pyplot as plt
#   from mpl_toolkits.mplot3d import Axes3D
#   fig = plt.figure()
#   ax = fig.add_subplot(111, projection='3d')
#   ax.scatter(ripple_probs.detach().numpy(), distances_ms, loss_detect.detach().numpy())
#   ax.set_xlabel('Ripple Probability')
#   ax.set_ylabel('Distance [ms]')
#   ax.set_zlabel('Loss_detect [a.u.]')
#   plt.title(title)



# ## Preparation
# bs = 64*32*4*10
# n_classes = 2
# distances_ms = abs(torch.randn(bs))*100
# # Funcs
# sigmoid = nn.Sigmoid()
# softmax = nn.Softmax(dim=-1)
# cross_entropy_criterion = nn.CrossEntropyLoss() # nn.LogSoftmax() and nn.NLLLoss() for numerical stability
# ripple_detect_criterion = RippleDetectLoss()

# # Confirm custom function
# lam = ripple_detect_criterion.get_lambda(distances_ms)
# # plot_distances_ms_lambda(distances_ms, lam)






# ## Random Simulation ##
# scores_random = torch.randn(bs, n_classes, requires_grad=True)
# probs_random = softmax(scores_random)
# loss_detect_random = ripple_detect_criterion(scores_random, distances_ms)
# plot_distances_ms_probs(distances_ms, probs_random[:,1],
#                         title='(Random Simulation) Scatter Plot of Distances [ms] and $\hat{Ripple Probability}$')
# plot_probs_distances_ms_loss_detect_in_3D(probs_random[:,1], distances_ms, loss_detect_random,
#                                           title='Random Simulation (Loss_sum: {:.2f})'.format(loss_detect_random.sum()))

# ## Learned Simulation ##
# scores_learned = scores_random*distances_ms.unsqueeze(dim=-1) # distances_ms + 0.1*torch.randn(bs, requires_grad=True)
# probs_learned = softmax(scores_learned)
# loss_detect_learned = ripple_detect_criterion(scores_learned, distances_ms)
# plot_distances_ms_probs(distances_ms, probs_learned[:,1],
#                         title='(Learned Simulation) Scatter Plot of Distances [ms] and $\hat{Ripple Probability}$')
# plot_probs_distances_ms_loss_detect_in_3D(probs_learned[:,1], distances_ms, loss_detect_learned,
#                                           title='Learned Simulation (Loss_sum: {:.2f})'.format(loss_detect_learned.sum()))

# ripple_detect_criterion.get_loss_0(scores_random)
# ripple_detect_criterion.get_loss_1(scores_random)
# ripple_detect_criterion.get_loss_0(scores_learned)
# ripple_detect_criterion.get_loss_1(scores_learned)



# # loss_detect.backward()




# '''
# ## Check the BCELoss Implementation
# bceloss = nn.BCELoss()
# input = torch.tensor([0.])
# target = torch.tensor([1.])
# bceloss(input, target)

# def mybceloss(input, target, eps=1e-12):
#   return -(input*np.log(target+eps) + (1-input)*np.log(1-target+eps))
# mybceloss(input, target)
# '''












# plt.scatter(probs[:, 1], loss_detect)
# plt.xlabel('probs')
# plt.ylabel('distances [ms]')

# def N(t, N0=1, T=10):
#   return N0 * (1/2) ** (t/T)

# tau=10
# t = np.arange(100)
# n = N(t, T=tau)
# plt.plot(t, n)
# plt.title('Decay Curve (Half-life: Tau = {})'.format(tau))


# # #############################################
# # ## https://scipython.com/book/chapter-7-matplotlib/examples/lifetimes-of-an-exponential-decay/

# # def plot_exponential_decay(N=10000, tau=28, tmax=100):
# #   import numpy as np
# #   import matplotlib.pyplot as plt

# #   # Initial value of y at t=0, lifetime in s
# #   # N, tau = 10000, 28
# #   # Maximum time to consider (s)
# #   # tmax = 100
# #   # A suitable grid of time points, and the exponential decay itself
# #   t = np.linspace(0, tmax, 1000)
# #   y = N * np.exp(-t/tau)

# #   fig = plt.figure()
# #   ax = fig.add_subplot(111)
# #   ax.plot(t, y)

# #   # The number of lifetimes that fall within the plotted time interval
# #   ntau = tmax // tau + 1
# #   # xticks at 0, tau, 2*tau, ..., ntau*tau; yticks at the corresponding y-values
# #   xticks = [i*tau for i in range(ntau)]
# #   yticks = [N * np.exp(-i) for i in range(ntau)]
# #   ax.set_xticks(xticks)
# #   ax.set_yticks(yticks)

# #   # xtick labels: 0, tau, 2tau, ...
# #   xtick_labels = [r'$0$', r'$\tau$'] + [r'${}\tau$'.format(k) for k in range(2,ntau)]
# #   ax.set_xticklabels(xtick_labels)
# #   # corresponding ytick labels: N, N/e, N/2e, ...
# #   ytick_labels = [r'$N$',r'$N/e$'] + [r'$N/{}e$'.format(k) for k in range(2,ntau)]
# #   ax.set_yticklabels(ytick_labels)

# #   ax.set_xlabel(r'$t\;/\mathrm{s}$')
# #   ax.set_ylabel(r'$y$')
# #   ax.grid()
# #   plt.show()
# # #############################################

# # plot_exponential_decay()
