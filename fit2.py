import numpy as np
import scipy.special as ss
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import torch
import math
import csv
import pandas as pd


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
def datasource(Source,N,P):
    if Source=='normal':
        Mean=P[0]
        Std=P[1]
        r = np.random.normal(Mean, Std, N)
        return r
    elif Source=='exponential':
        Scale=P[0]
        r = np.random.exponential(Scale, N)
        return r
    elif Source=='rayleigh':
        Scale= P[0]
        r = np.random.rayleigh(Scale, N)
        return r
    elif Source=='gamma':
        Shape = P[0]
        Scale=P[1]
        r = np.random.gamma(Shape,Scale,N)
        return r
    elif Source=='weibull':
        Shape = P[0]
        Scale=P[1]
        rv=weibull_min(c=Shape, loc=0, scale=Scale)
        r = rv.rvs(N)
        return r
    elif Source=='external':
        pathname= "/Users/rezaberangi/Desktop/Desktop - Reza’s MacBook Pro/Book5csv.csv"
        data = pd.read_csv(pathname) # pandas data frame
        cName=data.columns[cNum]
        r=eval("data."+cName)
        return r[0:-1]
def theory_cdf(process,parameters,rangex):
    if process=="normal":
        Mean=parameters[0]
        Std=parameters[1]
        cdf=[.5+.5*math.erf((rangev-Mean)/(Std*math.sqrt(2))) for rangev in rangex]
        return cdf
    elif process=="exponential":
        scale=parameters[0]
        cdf=[1-math.exp(-scale*rangev) for rangev in rangex]
        return cdf
    elif process=="gamma":
        alpha=parameters[0]
        beta=parameters[1]
        cdf=[ss.gammainc(alpha, beta*rangev) for rangev in rangex]
        return cdf
    elif process=="rayleigh":
        Scale=parameters[0]
        cdf=[1-math.exp(-rangev**2/(2*Scale**2)) for rangev in rangex]
        #pdf=[(rangev/Var)*math.exp(-rangev**2/(2*Var)) for rangev in rangex]
        return cdf
    elif process=="weibull":
        Shape=parameters[0]
        Scale=parameters[1]
        cdf=[1-math.exp(-(rangev/Scale)**Shape) for rangev in rangex]
        return cdf

if __name__ == '__main__':
    #------ Data source
    hypothesis={"normal":[0,1],"exponential":[2],"gamma":[1,5],"rayleigh":[2],"weibull":[1,1]}
    hypothesis.update({"external":"/Users/rezaberangi/Desktop/Desktop - Reza’s MacBook Pro/Book5csv.csv"})
    N=10000;
    process="weibull"
    r=datasource(process,N,hypothesis[process]) # input random variable

    #-----------  Update hypothesis
    N=len(r)
    Std=r.std()
    Mean=r.mean()
    Var=r.var()
    hypothesis.update({"normal":[Mean,Std]})
    RScale=Mean/math.sqrt(np.pi/2)
    hypothesis.update({"rayleigh":[RScale]})
    hypothesis.update({"exponential":[1/Std]})
    hypothesis.update({"gamma":[Mean**2/Var, Mean/Var]})
    params = weibull_min.fit(r, floc=0)
    hypothesis.update({"weibull":[params[0], params[2]]})


    # ---------- random generators
    plt.figure(figsize=(9, 3))
    bins=50
    #------ find and plot measured CDF
    PC=pdf_cdf(r,bins,0)
    rangex=PC[0]
    distribution=PC[1];
    plt.plot(rangex,distribution)
    plt.grid()

    #------ find and plot calculated CDF based on measured parameters
    process="weibull"
    CDF=theory_cdf(process,hypothesis[process],rangex)
    plt.plot(PC[0],CDF)
    Error=max(abs(CDF-PC[1]))
    plt.title(Error)
    plt.show()

