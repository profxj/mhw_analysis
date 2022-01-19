""" Routines to perform fits """
import numpy as np
from scipy import special

def fit_discrete_powerlaw(xval:np.ndarray, xmin:int,
                          alph_mnx:tuple, npts=1000):
    """Perform a power-law fit to a distribution of 
    discrete (integer) values

    See https://www.jstor.org/stable/pdf/25662336.pdf

    Args:
        xval (np.ndarray): [description]
        xmin (int): [description]
        alph_mnx (tuple): [description]
            Minimum, maximum value to explore for the power-law exponent
        npts (int, optional): [description]. Defaults to 1000.
    
    Returns:
        tuple: C, best_alpha, alpha, logL
    """
    alpha = np.linspace(alph_mnx[0], alph_mnx[1], npts)

    # Zeta
    zeta = special.zeta(-1*alpha, xmin)


    # Natural log
    logL = -xval.size * np.log(zeta) + alpha*np.sum(np.log(xval))

    # Best 
    imax = np.argmax(logL)
    best_alpha = alpha[imax]

    # Constant
    C = 1./special.zeta(-1*best_alpha, xmin)

    # Return
    return C, best_alpha, alpha, logL

def fit_continuous_powerlaw(xval:np.ndarray, xmin:float):
    """Perform a continuous power-law fit to a distribution of 
    continuous values

    See https://www.jstor.org/stable/pdf/25662336.pdf

    Args:
        xval (np.ndarray): [description]
        xmin (float): [description]
    
    Returns:
        tuple: C, best_alpha
    """
    # MLE estimator [alpha is *positive* here]
    pos_alpha = 1 + xval.size / np.sum(np.log(xval/xmin))

    # Constant
    C = (pos_alpha-1) / xmin * (1./xmin)**(-1*pos_alpha)

    # Return
    return C, -1*pos_alpha