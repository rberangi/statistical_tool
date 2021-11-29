import tkinter as tk
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch

def sum1(a, b):
    return a + b

if __name__ == '__main__':

    a = np.array([1., 2])
    b = np.array([3, 4])

    c = sum1(a, b)

    print(c)

    """ m=0;s=1;N=10000
    scale=.1;shape=.3
    a0=5
    df=3
    Nu=2
    a=np.random.normal(m,s,N)
    b=np.random.rayleigh(s,N)
    c=np.random.exponential(s,N)
    d=np.random.gamma(shape,scale,N)
    e=np.random.weibull(a0,N)
    f=np.random.chisquare(df,N)
    g=nakagami.rvs(Nu,N)
    #plt.figure(figsize=(9, 3))
    plt.plot(g)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
    plt.show() """
