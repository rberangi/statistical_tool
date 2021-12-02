import numpy as np
import scipy.special as ss
import scipy.stats as st

from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import math
import pandas as pd
from tabulate import tabulate

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
    if Source=='Normal':
        Mean=P[0]
        Std=P[1]
        r = np.random.normal(Mean, Std, N)
        return r
    elif Source=='Exponential':
        Scale=P[0]
        r = np.random.exponential(Scale, N)
        return r
    elif Source=='Rayleigh':
        Scale= P[0]
        r = np.random.rayleigh(Scale, N)
        return r
    elif Source=='Gamma':
        Shape = P[0]
        Scale=P[1]
        r = np.random.gamma(Shape,Scale,N)
        return r
    elif Source=='Weibull':
        Shape = P[0]
        Scale=P[1]
        rv=weibull_min(c=Shape, loc=0, scale=Scale)
        r = rv.rvs(N)
        return r
    elif Source=='External':
        cNum = 2 # column index
        pathname= "/Users/rezaberangi/Desktop/Desktop - Reza’s MacBook Pro/Book5csv.csv"
        data = pd.read_csv(pathname) # pandas data frame
        cName=data.columns[cNum]
        r=eval("data."+cName)
        return r[0:-1]
def theory_cdf(process,parameters,rangex):
    if process=="Normal":
        Mean=parameters[0]
        Std=parameters[1]
        cdf=[.5+.5*math.erf((rangev-Mean)/(Std*math.sqrt(2))) for rangev in rangex]
        return cdf
    elif process=="Exponential":
        loc=parameters[0]
        scale=parameters[1]
        cdf=[1-math.exp(-scale*(rangev-loc)) for rangev in rangex]
        return cdf
    elif process=="Gamma":
        loc=parameters[1]
        alpha=parameters[0]
        scale=parameters[2]
        cdf=[ss.gammainc(alpha, (rangev-loc)/scale) for rangev in rangex]
        return cdf
    elif process=="Rayleigh":
        loc=parameters[0]
        Scale=parameters[1]
        cdf=[1-math.exp(-(rangev-loc)**2/(2*Scale**2)) for rangev in rangex]
        #pdf=[(rangev/Var)*math.exp(-rangev**2/(2*Var)) for rangev in rangex]
        return cdf
    elif process=="Weibull":
        loc=parameters[1]
        Shape=parameters[0]
        Scale=parameters[2]
        cdf=[1-math.exp(-(rangev/Scale)**Shape) for rangev in rangex]
        return cdf

if __name__ == '__main__':
    #------ Data source [generate a random variable or import from a csv file
    hypothesis={"Normal":[0,1],"Exponential":[2],"Gamma":[1,5],"Rayleigh":[2],"Weibull":[1,2]}
    hypothesis.update({"External":[]})
    test_dist0="Normal=0 |"+"Exponential=1 |"+"Gamma=2 |"+"Rayleigh=3 |"+"Weibull=4 |"+"From external .csv file=5 |"
    test_dist1=["Normal","Exponential","Gamma","Rayleigh","Weibull","External"]
    print("Enter a process index")
    pindex = int(input(test_dist0+ "\n Enter a number [0-5] here >>>"))
    input_process=test_dist1[pindex]
    if pindex!=5:
        print("Enter number of samples to simulate:")
        N=int(input("Enter here >>>"))
    r=datasource(input_process,N,hypothesis[input_process]) # input random variable

    #-----------  Update hypothesis
    N=len(r)
    Std=r.std()
    Mean=r.mean()
    Var=r.var()

    params = st.norm.fit(r)
    hypothesis.update({"Normal":params})
    params = st.rayleigh.fit(r)
    hypothesis.update({"Rayleigh":params})
    params = st.expon.fit(r)
    hypothesis.update({"Exponential":[params[0], 1/params[1]]})
    params = st.gamma.fit(r)
    hypothesis.update({"Gamma":params})
    params = st.weibull_min.fit(r)
    hypothesis.update({"Weibull":params})


    #------ find CDF/PDF from random varaiable
    bins=100 # number of bins for histogram
    PC=pdf_cdf(r,bins,PDF=0)
    rangex=PC[0] # input range vector
    distribution=PC[1]; # PDF (PDF=1) or CDF (PDF=0)
    #=============  process fit using Kommogrove-Smironov test
    test_dist=["Normal","Exponential","Gamma","Rayleigh","Weibull"]
    print(test_dist)
    Error=np.zeros(len(test_dist))
    imin,Errormin=0,1
    PD=""
    for i in range(len(test_dist)):
        process=test_dist[i]
        CDF=theory_cdf(process,hypothesis[process],rangex)
        Error[i]=max(abs(CDF-PC[1]))
        if Error[i]>1:
            Error[i]=1
        elif np.isnan(Error[i]):
            Error[i]=1
        if Error[i]<0.05:
            PD=PD+" | "+test_dist[i]
        if Error[i]<Errormin:
            Errormin=Error[i]
            imin=i
    plt.figure(figsize=(9, 4))
    plt.bar(test_dist,Error)
    plt.ylabel("Error")
    plt.xlabel("Possible distributions, error<0.05: "+PD+" |")
    plt.title("Best fit= "+test_dist[imin]+", Error= "+str(Error[imin]))
    hypothesis2=hypothesis;
    hypothesis2.pop("External")
    print(tabulate(hypothesis2,headers="keys"))
    print("Errors:")
    print(Error)
    plt.show()

