import numpy as np
from matplotlib import pylab as plt

def ecdf(a, ax=None, add_zero=True, **kwargs):
    """Plot the Empirical Cumulative Density Function.
    
    parameters:
        a: The data of interest. Should be a list or a 1D numpy array.
        ax: The axes of the figure.
        add_zero: Set to True to connect the graph to y=0
        **kwargs: Any other option to pass to pylab.plot.
        
    Returns a dictionary providing the CDF value for each point on the x axis.
    """
    sorted=np.sort( a )
    yvals=np.arange(len(sorted)+1)/float(len(sorted))
    if add_zero:
        starti = 0
        sorted = np.append( sorted[0], sorted )
    else:
        starti=1
    if ax is None:
        plt.plot( sorted, yvals[starti:], **kwargs )
    else:
        ax.plot( sorted, yvals[starti:], **kwargs )

    return {k:v for k,v in zip(sorted, yvals)}


def eccdf(a, ax=None, **kwargs):
    """Plot the Empirical Complementary Cumulative Density Function.
    
    parameters:
        a: The data of interest. Should be a list or a 1D numpy array.
        ax: The axes of the figure.
        **kwargs: Any other option to pass to pylab.plot.

    Returns a dictionary providing the CCDF value for each point on the x axis.
    """
    sorted=np.sort( a )
    yvals=np.arange(len(sorted))/float(len(sorted))
    if ax is None:
        plt.plot( sorted, 1-yvals, **kwargs )
    else:
        ax.plot( sorted, 1-yvals, **kwargs )

    return {k:v for k,v in zip(sorted, 1-yvals)}
