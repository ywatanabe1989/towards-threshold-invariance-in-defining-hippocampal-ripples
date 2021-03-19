#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
sys.path.append('./07_Learning/')
import utils.myfunc as mf

############################################################
log_lat_mean = mf.pkl_load('07_Learning/log(lat)_mean_std.pkl')['log(lat+1e-5)_mean'] # 0.9229334
log_lat_std = mf.pkl_load('07_Learning/log(lat)_mean_std.pkl')['log(lat+1e-5)_std'] # 1.7429373
p = {'samp_rate':1000}

mu = 10
sigma = 3
N = 1000000

rv_x = scipy.stats.norm(loc=mu, scale=sigma)
x = rv_x.rvs(size=N)
plt.hist(x)
plt.show()
print(x.mean())
print(x.std())


y = log_lat_std * x + log_lat_mean
mean_y = log_lat_std * x.mean() + log_lat_mean
sigma_y = log_lat_std * x.std()
plt.hist(y)
plt.show()
print(y.mean())
print(mean_y)
print(y.std())
print(sigma_y)


z = np.exp(y)
plt.hist(z, bins=np.linspace(0, 100000, 100))
plt.show()
mean_z = np.exp(mean_y + (sigma_y**2)/2)
median_z = np.exp(mean_y)
mode_z = np.exp(mean_y - sigma_y**2)
var_z = (np.exp(sigma_y**2)-1) * np.exp(2*mean_y + (mean_y**2))

print(z.mean())
print(mean_z)

print(z[np.argsort(z)[len(z)//2]])
print(median_z)

print(scipy.stats.mode(z))
print(mode_z)


print(z.var())
print(var_z)






























def f1(x):
  y = np.exp(log_lat_std * x + log_lat_mean) - 1e-5
  return y

def g1(y):
  x = ( np.log(y + 1e-5) - log_lat_mean )/ log_lat_std
  return x

def f2(y):
  z = y * p['samp_rate']
  return z

def g2(z):
  y = z / p['samp_rate']
  return y

def normal_pdf(x, mu, sigma):
  var = sigma**2
  return 1./((2*np.pi*var)**0.5) * np.exp(-(x-mu)**2 / (2*var))

def pdf_x(x, mu, sigma):
  return normal_pdf(x, mu, sigma)

def pdf_y(y, mu, sigma):
  x = g1(y)
  dxdy = 1 / (log_lat_std * (y + 1e-5))
  return pdf_x(x, mu, sigma) * abs(dxdy)

def pdf_z(z, mu, sigma):
  y = g2(z)
  dydz = 1 / p['samp_rate']
  return pdf_y(y, mu, sigma) * abs(dydz)

def transformed_pdf(z, mu, sigma):
  y = z / 1000 # p['samp_rate']
  x = g(y)
  dxdy = 1 / (log_lat_std * (y + 1e-5))
  dydz = 1 / p['samp_rate']
  return normal_pdf(x, mu, sigma) * abs(dxdy) * abs(dydz)
############################################################


# The transformation function of random variables
point_num = 1000
x = np.linspace(0, 50, point_num, endpoint=False)
y = f1(x)
plt.plot(x, y, 'b', label='The transformation function (y = f1(x) (= exp({:.2f} * x + {:.2f}) - 1e-5) )'\
         .format(log_lat_std, log_lat_mean))
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('The Jacobian Transformation (y = f1(x))')
plt.show()

# px(x)
mu = 10
sigma = 1
x_px = np.linspace(0, mu*2, point_num, endpoint=False)
y_px = pdf_x(x_px, mu, sigma)
delta_x = x_px[1] - x_px[0]
auc = y_px.sum() * delta_x
plt.plot(x_px, y_px, label='h = px(x) ($\mu$={}, $\sigma$={}) (AUC={:.2f})'.format(mu, sigma, auc))
plt.xlim([0, 20])
plt.xlabel('X')
plt.legend()
plt.show()

# py(y)
# x_py = np.linspace(0, 2e9, point_num, endpoint=False)
y_py = pdf_y(x_px, mu, sigma)
delta_x = x_px[1] - x_px[0]
auc = y_py.sum() * delta_x
plt.plot(x_py, y_py, label='h = py(y) ($\mu$={}, $\sigma$={}) (AUC={:.2f})'.format(mu, sigma, auc))
plt.xlabel('Y')
plt.legend()
plt.show()


# pz(z)
y_pz = pdf_z(x0, mu, sigma)
plt.plot(x0, y_pz, label='h = pz(z)')

plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()

# plt.yscale('log')



y_x = f2(f1(x_ax))
plt.plot(y2, label='x vs z')
y = f1(x_ax)
plt.plot(y, label='x vs y')
plt.yscale('log')
plt.legend()
plt.show()

## x ##
yx = pdf_x(x_ax, mu, sigma)
plt.plot(yx, label='PDF x (sum: {:.2f})'.format(yx.sum()))
# plt.legend()
# plt.show()

## y ##
yy = pdf_y(x_ax, mu, sigma)
plt.plot(yy, label='PDF y (sum: {:.2f})'.format(yy.sum()))
# plt.legend()
# plt.show()

## z ##
yz = pdf_z(x_ax, mu, sigma)
plt.plot(yz, label='PDF z (sum: {:.2f})'.format(yz.sum()))
plt.legend()
plt.show()








### Scipy ###
from scipy.stats import norm
fig, ax = plt.subplots(1,1)

mean, var, skew, kurt = norm.stats(moments='mvsk')

# x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
x = np.linspace(-5, 100, 10000)
ax.plot(x, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='normal pdf')
ax.legend()

rv = norm()
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
rv2 = norm(loc=10, scale=3)
ax.plot(x, rv2.pdf(x), 'k-', lw=2, label='frozen pdf 2')

vals = norm.ppf([0.001, 0.5, 0.999])
norm.cdf(vals)

r = norm.rvs(size=10000)
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()

### Pytorch ###
from torch.distributions import normal

m = normal.Normal(mu, sigma)
# m.perplexity() # == np.exp(m.entropy())






# # https://qiita.com/Alreschas/items/847d164be04dd30d5035
# #ガウス分布の密度関数
# def gaussianDist(sig,mu,x):
#     y=np.exp(-(x-mu)**2/(2*sig**2))/(np.sqrt(2*np.pi)*sig)
#     return y

# #確率変数の変換関数
# def g(y):
#     x=np.log(y)-np.log(1-y)+5
#     return x

# #確率変数の変換関数の逆関数
# def invg(x):
#     y=1/(1+np.exp(-x+5))
#     return y

# #ガウス分布px(x)の平均、分散
# sig=1.0
# mu=5

# #ヒストグラムのサンプル数
# N = 50000

# plt.xlim([0,100])
# plt.ylim([0,10])

# ####
# x = np.linspace(0,100,1000)

#確率変数の変換関数をプロット
y=invg(x)
plt.plot(x,y,'b')

#px(x)のプロット
y = gaussianDist(sig,mu,x)
plt.plot(x,y,'r')

#px(x)からのサンプルを元にヒストグラムをプロット
x_sample = mu + sig * np.random.randn(N)
plt.hist(x_sample,bins=20,normed=True,color='lavender')


####
y=np.linspace(0.01,0.99,100)

##py(y)のプロット
x=gaussianDist(sig,mu,g(y))/(y*(1-y))
plt.plot(x,y,'m')

#px(x)からのサンプルをg^-1(x)で変換したデータのヒストグラムをプロット
y_sample = invg(mu + sig * np.random.randn(N))
plt.hist(y_sample,bins=20,normed=True,orientation="horizontal",color='lavender')

#px(g(y))のように単純に変換した関数をプロット
x = gaussianDist(sig,mu,g(y))
plt.plot(x/(x.sum()*0.01) ,y,'lime')

####
#平均muとg^-1(mu)との関係をプロット
plt.plot([mu, mu], [0, invg(mu)], 'k--')
plt.plot([0, mu], [invg(mu), invg(mu)], 'k--')
plt.show()
