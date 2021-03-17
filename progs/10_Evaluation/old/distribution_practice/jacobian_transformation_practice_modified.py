#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://qiita.com/Alreschas/items/847d164be04dd30d5035

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# PDF of the normal distribution
rv = norm()
# gaussianDist = norm.pdf(x, loc=mu, scale=sigma)

# def gaussianDist(sigma,mu,x):
#     y=np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
#     return y

# The Transformation Functions
def f(x):
    y = 1 / ( 1 + np.exp(-x+5) )
    return y

def g(y): # invf
    x = np.log(y) - np.log(1-y) + 5
    return x


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
# y = gaussianDist(sigma,mu,x)
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
dxdy = 1/(y*(1-y))
# x=gaussianDist(sigma,mu,g(y))/(y*(1-y))
x = rv_x.pdf(g(y))*abs(dxdy)
plt.plot(x,y,'m', label='py(y) = px(g(y))|dx/dy| (HORIZONTAL)')

# Plot the histogram transformed from the original one by the transformation function f(x)
# y_sample = invg(mu + sigma * np.random.randn(N))
y_sample = f(x_sample)
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
