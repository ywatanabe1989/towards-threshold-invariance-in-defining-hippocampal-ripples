#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://qiita.com/Alreschas/items/847d164be04dd30d5035

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
from scipy import stats

# The Transformation Functions
def f(x):
    y = 1 / ( 1 + np.exp(-x+5) )
    return y

def g(y): # invf
    x = np.log(y) - np.log(1-y) + 5
    return x

def dxdy(y):
  return 1/(y*(1-y))

def gaussianDist(x, mu=0., sigma=1.):
    y = np.exp(-(x-mu)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)
    return y

class tf_gen(stats.rv_continuous):
  def _pdf(self, y, mu=0., sigma=1.):
    return gaussianDist(g(y), mu=mu, sigma=sigma)*abs(dxdy(y))


# The mean and the std of px(x)
mu = 6
sigma = 1.0

N = 50000 # Sample Number
plt.xlim([0,10])
plt.ylim([0,1])

####
x = np.linspace(0,10,100)

# Plot the transformation function
y = f(x)
plt.plot(x,y,'b', label='The Transformation Function (y=f(x))')

# Plot px(x)
rv_x = norm(loc=mu, scale=sigma)
y = rv_x.pdf(x)
plt.plot(x,y,'r', label='px(x)')

# Plot the samples from px(x)
# x_sample = mu + sigma * np.random.randn(N)
x_sample = rv_x.rvs(size=N)
plt.hist(x_sample,bins=20,normed=True,color='bisque', label='Samples from px(x)')

####
y = np.linspace(0.01,0.99,100)

## Plot py(y)
rv_y = tf_gen(shapes='mu, sigma')
x = rv_y.pdf(y, mu=mu, sigma=sigma)
plt.plot(x,y,'m', label='py(y) = px(g(y))|dx/dy| (HORIZONTAL)')

# Plot the histogram transformed from the original one by the transformation function f(x)
# y_sample = invg(mu + sigma * np.random.randn(N))
y_sample = f(x_sample) # rv_y.rvs(size=N, mu=mu, sigma=sigma)
plt.hist(y_sample,bins=20,normed=True,orientation="horizontal",color='lavender', label='Samples transformed by y=f(x) (HORIZONTAL)')

# Plot False py(y) ( = px(g(y)) )
# x = gaussianDist(sigma,mu,g(y))
x = rv_x.pdf(g(y))
plt.plot(x/(x.sum()*0.01) ,y,'lime', label='False py(y) = px(g(y)) (HORIZONTAL)')

####
# Plot the relationship between mu and the transformation
plt.plot([mu, mu], [0, f(mu)], 'k--')
plt.plot([0, mu], [f(mu), f(mu)], 'k--')
plt.title('The Jacobian Transformation by y = f(x) ( = 1 / ( 1 + exp(-x+5) ) )')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()



# ## mean, median,
# mean_x = rv_x.mean()
# mean_y = f(mean_x)


# rv_x.median()
# rv_x.mode()



# rv_y.mean(mu=mu, sigma=sigma)

rv_z = lognorm(s=.25, loc=0)
# x = np.linspace(rv_z.ppf(0.01), rv_z.ppf(0.99), 1000)
x = np.linspace(0, 2.5, 1000)
# y = rv_z.pdf(x)
y1 = lognorm.pdf(x, s=0.25, loc=0)
y2 = lognorm.pdf(x, s=0.50, loc=0)
y3 = lognorm.pdf(x, s=1.00, loc=0)
plt.plot(x, y1, label='y1')
plt.plot(x, y2, label='y2')
plt.plot(x, y3, label='y3')
plt.legend()
plt.show()


mu = 0
sigma = 0.25
import scipy

def get_mean(mu, sigma):
    return np.exp(mu + sigma**2/2)

def get_median(mu, sigma):
    return np.exp(mu)

def get_mode(mu, sigma):
    return np.exp(mu - sigma**2)

def get_quantile(mu, sigma, F):
    return np.exp(mu + np.sqrt(2*(sigma**2)) * scipy.special.erfinv(2*F-1))

# scipy.special.erf(z)

import math

print(lognorm.mean(s=sigma, loc=mu), get_mean(mu, sigma))
print(math.isclose(lognorm.mean(s=sigma, loc=mu), get_mean(mu, sigma)))

print(lognorm.median(s=sigma, loc=mu), get_median(mu, sigma))
print(math.isclose(lognorm.median(s=sigma, loc=mu), get_median(mu, sigma)))

F = np.array([0.025, 0.975])
print(lognorm.ppf(F, s=sigma, loc=mu), get_quantile(mu, sigma, F))
print(np.isclose(lognorm.ppf(F, s=sigma, loc=mu), get_quantile(mu, sigma, F)))



# lognorm.mode(s=sigma, loc=mu) == get_mode(mu.sigma)
