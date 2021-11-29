import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import math

def pdf_cdf(x,bins,PDF=True):
    xmin=min(x)
    xmax=max(x)
    rangex=np.linspace(xmin, xmax, num=bins+1) # +1 to have exact number of bins
    H=np.histogram(x,rangex)
    cdf=np.cumsum(H[0][0:bins])/np.sum(H[0][0:bins])
    pdf=(cdf[1:bins]-cdf[0:bins-1])/((xmax-xmin)/bins)
    if PDF==1:
        range_pdf=(rangex[1:bins]+rangex[2:bins+1])/2
        return [range_pdf,pdf]
    else:
        range_cdf=rangex[1:bins+1];
        return [range_cdf,cdf]

def theory_cdf(rangex,parameters,process):
    if process=="normal":
        Mean=parameters[0]
        Std=parameters[1]
        cdf=[.5+.5*math.erf((rangev-Mean)/(Std*math.sqrt(2))) for rangev in rangex]
        return cdf
   elif process=="exponential":
        scale=parameters[0]
        cdf=[1-exp(-scale*x(rangev)) for rangev in rangex]
        return cdf

if __name__ == '__main__':
    # ---------- random generators
    N=10000
    Std=1;
    Mean=0
    r=np.random.normal(Mean,Std,N)

    """ Scale=1
    r=np.random.exponential(Scale,N) """

    """ Std=1
    r=np.random.rayleigh(Std,N) """

    """ Shape=1
    Scale=1
    r=np.random.gamma(Shape,Scale,N) """

    plt.figure(figsize=(9, 3))
    bins=50
    PC=pdf_cdf(r,bins,0)
    plt.plot(PC[0],PC[1])
    plt.grid()

    CDF=theory_cdf(PC[0],[Mean,Std],"normal")
    plt.plot(PC[0],CDF)
    Error=max(abs(CDF-PC[1]))
    plt.title(Error)
    plt.show()

