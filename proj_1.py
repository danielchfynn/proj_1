import random
import math
import numpy as np
import pandas as pd
from math import sqrt as sqrt
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from statsmodels.tsa.api import SARIMAX

def gen_data(psi, sigma2, N = 1000, seed = 1, prints = False):

    #initialisations
    p = len(psi)
    y = [0 for i in range(N)]
    
    random.seed(seed)
    noise = [random.gauss(0, sqrt(sigma2)) for i in range(N)]

    for i in range(0,p): #adding noise for first p values 
        y[i] = noise[i]
    
    #could do this with matrix multiplication instead of loop
    for i in range(p, N):
        for j in range(len(psi)):
            y[i] += psi[j] * y[i-j-1]
        y[i]+=noise[i]

    if prints:
        plt.plot(y)

    #Generating X matrix
    X_temp = pd.DataFrame()#0,index = range(0,N-p), columns = col)
    for i in range(p-1,0-1,-1):
        X = y[(i):(N-p+i)]
        X_temp["lag"+str(p-i)] = X
    #print(X_temp)

    X = X_temp.to_numpy()
    y = np.array(y[p:N])

    return y, X, noise

#reparamatisations
def sigma2_to_exp(sigma2):
    return math.exp(sigma2)

def exp_to_sigma2(exp):
    return math.log(exp)

def sigma2_to_sq(sigma2):
    return np.power(sigma2,2)

def sq_to_sigma2(sq):
    return np.sqrt(sq)

def barndorff_schou_transformation(phi_restricted):
    newparms = np.copy(phi_restricted)
    tmp = np.copy(phi_restricted)

    for i in range(1, len(phi_restricted)):
        newparm_temp = newparms[i - np.arange(1, i + 1)]
        tmp_new = tmp[:i] - newparms[i] * newparm_temp
        tmp = np.concatenate((tmp_new, newparms[i:]))
        newparms = np.concatenate((tmp[:i], newparms[i:]))

    return newparms

#def normdist(theta, y_t):
#
#    #thata = [psi1,..., psi_n, sigma]
#    # mu = psi * y_t      t = 1,...,p
#    #assume y_t = [y_t-p,...,y_t-1, y_t]
#
#    y_t.reverse()
#
#    mu = 0
#    for i in range(0, len(theta)-1):
#        mu += theta[i]* y_t[i+1] #y_t[0] = predicted quantity - after reversal
#
#    return 1/math.sqrt(2 * math.pi * theta[-1]) * math.exp (-1*((y_t[0] - mu )**2) / (2*theta[-1]))
#

def neg_log_lik(theta, y, X):
    p = len(theta) - 1
    N = len(y)
    B = np.array(theta[0:p])

    sigma2 = theta[-1]
    sigsq= sigma2_to_sq(sigma2)
    B = np.tanh(B)
    B = barndorff_schou_transformation(B)

    print(B, sigsq ,N/2*math.log(2*math.pi) + N/2*math.log(sigsq) + 1/(2*sigsq) *( y.T @ y - 2*y.T @ X @ B  + B.T @ X.T @ X @ B  ))
    return N/2*math.log(2*math.pi) + N/2*math.log(sigsq) + 1/(2*sigsq) *( y.T @ y - 2*y.T @ X @ B  + B.T @ X.T @ X @ B  ) 

def ARToPacf(phi):
    phik = phi.copy()
    L = len(phi)
    if L == 0:
        return [0]
    pi = np.zeros(L)
    for k in range(1, L+1):
        LL = L + 1 - k
        a = phik[LL-1]
        pi[L-k] = a
        phikp1 = np.delete(phik, LL-1)
        if np.isnan(a) or abs(a) == 1:
            raise ValueError("transformation is not defined, partial correlation = 1")
        phik = (phikp1 + a * np.flip(phikp1)) / (1 - a**2)
    return pi.tolist()


def fitting_values(y, p, phi_approx):
    fitted = []
    for i in range(len(y)-p):
        y_temp = list(y[i:p+i])
        y_temp.reverse()
        fitted.append(y_temp @ phi_approx)
    return fitted 

def predicting_values(y, p, N, phi_approx, forecast_len = 125):
    forecast = []

    initials = list(y[-p:N])
    initials.reverse()
    initials

    for i in range(forecast_len):
        forecast.append(initials @ phi_approx)
        temp = initials[0:(p-1)] #moving window of the p last values
        temp.insert(0, initials @ phi_approx) #adding the newest prediction
        initials = temp
    return forecast 

def mse(y , fitted):
    return np.sum((y - fitted)**2) / len(y) 

#Copula fns
def theta_to_exp(theta):
    return math.exp(theta)

def exp_to_theta(exp):
    return math.log(exp)

def theta_to_sq(theta):
    return np.power(theta,2)

def sq_to_theta(sq):
    return np.sqrt(sq)

def neg_log_clayton_pdf(theta, u, v):
    #theta = theta_to_exp(theta)
    print("theta", theta)

    #return -1* np.log((1 + theta) * (u@v) **(-1-theta) * np.sum((u ** (-theta) + v**(-theta) -1) **-((2*theta + 1)/theta))) 
    if theta ==0:
        theta += np.random.choice([0.0001, -0.0001], 1)
    
    if theta >0:
        print("neg log clayton", -1 *np.sum(( np.log(theta + 1) - (theta + 1) * ( np.log(u) + np.log(v)) - ((2*theta + 1)/theta) * np.log(u**(-theta) + v**(-theta) -1 )) ))
        return -1 *np.sum(( np.log(theta + 1) - (theta + 1) * ( np.log(u) + np.log(v) ) - ((2*theta + 1)/theta) * np.log(u**(-theta) + v**(-theta) -1  )) )
    
    else:
        print("neg log clayton b", -1 *np.sum(( np.log(theta + 1) - (theta + 1) * ( np.log(u) + np.log(v)) - ((2*theta + 1)/theta) * np.log(- 1* ( -1*(u**(-theta)-1) + -1*(v**(-theta)-1)  ) +1 ) ) ))
        return -1 *np.sum(( np.log(theta + 1) - (theta + 1) * ( np.log(u) + np.log(v)) - ((2*theta + 1)/theta) * np.log(- 1* ( -1*(u**(-theta)-1) + -1*(v**(-theta)-1)  ) +1 ) ) )

#def neg_log_clayton_pdf(theta, u, v):
#    print("hi", theta)
#    theta = theta_to_exp(theta )
#    print("theta", theta)
#    #theta = theta_to_exp(theta)
#    #return -1* np.log((1 + theta) * (u@v) **(-1-theta) * np.sum((u ** (-theta) + v**(-theta) -1) **-((2*theta + 1)/theta))) 
#    print("neg log clayton", -1 *np.sum(( np.log(theta + 1) - (theta + 1) * ( np.log(u) + np.log(v)) - ((2*theta + 1)/theta) * np.log(u**(-theta) + v**(-theta) -1 )) ))
#    return -1 *np.sum(( np.log(theta + 1) - (theta + 1) * ( np.log(u) + np.log(v)) - ((2*theta + 1)/theta) * np.log(u**(-theta) + v**(-theta) -1 )) )