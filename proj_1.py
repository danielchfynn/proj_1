import random
import math
import numpy as np
import pandas as pd
from math import sqrt as sqrt
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import scipy.stats as stats

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


def gen_data_copula(cop, psi1,psi2,  sigma21, sigma22, N = 1000, horizon = 125,  seed = 1):

    #initialisations
    p1 = len(psi1)
    p2 = len(psi2)
    y1 = [0 for i in range(N)]
    y2= [0 for i in range(N)]

    random.seed(seed)
    #noise = [random.gauss(0, sqrt(sigma2)) for i in range(N)] no cop

    u_gen = cop.random(N, seed = seed) #copulae
    e_1t = stats.norm.ppf(u_gen[:,0])
    e_2t = stats.norm.ppf(u_gen[:,1])

    for i in range(0,p1): #adding noise for first p values 
        y1[i] = e_1t[i]*sigma21
    
    for i in range(0,p2): #adding noise for first p values 
        y2[i] = e_2t[i]*sigma22

    #could do this with matrix multiplication instead of loop
    for i in range(p1, N):
        for j in range(len(psi1)):
            y1[i] += psi1[j] * y1[i-j-1]
        y1[i]+=e_1t[i]*sigma21

    for i in range(p2, N):
        for j in range(len(psi2)):
            y2[i] += psi2[j] * y2[i-j-1]
        y2[i]+=e_2t[i]*sigma22

    y1_test = np.array(y1[-horizon:N])
    y2_test = np.array(y2[-horizon:N])
    y1 = y1[0:-horizon]
    y2 = y2[0:-horizon]
    N = N - horizon

    #Generating X matrix
    X_temp = pd.DataFrame()#0,index = range(0,N-p), columns = col)
    for i in range(p1-1,0-1,-1):
        X = y1[(i):(N-p1+i)]
        X_temp["lag"+str(p1-i)] = X
    #print(X_temp)
    X1 = X_temp.to_numpy()
    y1 = np.array(y1[p1:N])

    X_temp = pd.DataFrame()#0,index = range(0,N-p), columns = col)
    for i in range(p2-1,0-1,-1):
        X = y2[(i):(N-p2+i)]
        X_temp["lag"+str(p2-i)] = X
    #print(X_temp)
    X2 = X_temp.to_numpy()
    y2 = np.array(y2[p2:N])

    return y1, X1, y2, X2, y1_test, y2_test



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
    sig= sigma2_to_exp(sigma2)
    B = np.tanh(B)
    B = barndorff_schou_transformation(B)

    print(B, sig ,N/2*math.log(2*math.pi) + N/2*math.log(sig) + 1/(2*sig) *( y.T @ y - 2*y.T @ X @ B  + B.T @ X.T @ X @ B  ))
    return N/2*math.log(2*math.pi) + N/2*math.log(sig) + 1/(2*sig) *( y.T @ y - 2*y.T @ X @ B  + B.T @ X.T @ X @ B  ) 

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
    theta = theta_to_exp(theta+1) #should expand the range for -1,inf
    

    #return -1* np.log((1 + theta) * (u@v) **(-1-theta) * np.sum((u ** (-theta) + v**(-theta) -1) **-((2*theta + 1)/theta))) 
    if theta ==0:
        theta += np.random.choice([0.0001, -0.0001], 1) #want to use sign of previous theta to jump gap, but dont have
    
    #if theta <= -1: #hmm works 
    #    theta =  0.999

    print("theta", theta)

    if theta >0:
        print("neg log clayton", -1 *np.sum(( np.log(theta + 1) - (theta + 1) * ( np.log(u) + np.log(v)) - ((2*theta + 1)/theta) * np.log(u**(-theta) + v**(-theta) -1 )) ))
        return -1 *np.sum(( np.log(theta + 1) - (theta + 1) * ( np.log(u) + np.log(v) ) - ((2*theta + 1)/theta) * np.log(u**(-theta) + v**(-theta) -1  )) )
    
    else:
        print("neg log clayton b", -1 *np.sum(( np.log(theta + 1) - (theta + 1) * ( np.log(u) + np.log(v)) - ((2*theta + 1)/theta) * np.log(- 1* ( -1*(u**(-theta)-1) + -1*(v**(-theta)-1)  ) +1 ) ) ))
        return -1 *np.sum(( np.log(theta + 1) - (theta + 1) * ( np.log(u) + np.log(v)) - ((2*theta + 1)/theta) * np.log(- 1* ( -1*(u**(-theta)-1) + -1*(v**(-theta)-1)  ) +1 ) ) )

def neg_log_frank_pdf(theta, u, v):
    print("theta", theta)
    #domain [-inf:inf] \{0}
    if theta ==0:
        theta += np.random.choice([0.0001, -0.0001], 1)
    print(- 1 * np.sum((-theta * np.exp(-theta*(u+v))*(np.exp(-theta) -1 )) / np.power((np.exp(-theta) - np.exp(-theta*u) - np.exp(-theta*v) + np.exp(-theta*(u+v))),2)))
    return - 1 * np.sum((-theta * np.exp(-theta*(u+v))*(np.exp(-theta) -1 )) / np.power((np.exp(-theta) - np.exp(-theta*u) - np.exp(-theta*v) + np.exp(-theta*(u+v))),2))


#def neg_log_clayton_pdf(theta, u, v):
#    print("hi", theta)
#    theta = theta_to_exp(theta )
#    print("theta", theta)
#    #theta = theta_to_exp(theta)
#    #return -1* np.log((1 + theta) * (u@v) **(-1-theta) * np.sum((u ** (-theta) + v**(-theta) -1) **-((2*theta + 1)/theta))) 
#    print("neg log clayton", -1 *np.sum(( np.log(theta + 1) - (theta + 1) * ( np.log(u) + np.log(v)) - ((2*theta + 1)/theta) * np.log(u**(-theta) + v**(-theta) -1 )) ))
#    return -1 *np.sum(( np.log(theta + 1) - (theta + 1) * ( np.log(u) + np.log(v)) - ((2*theta + 1)/theta) * np.log(u**(-theta) + v**(-theta) -1 )) )


#prediction fn not using previous mean values 
def predict_copula_pi(cop, y1, y2, results1, results2, p1, p2, N1, N2, repeats = 10000, horizon = 125, seed = 10):
    phi1_approx = barndorff_schou_transformation(np.tanh(results1.x[0:p1]))
    sigma21_approx = sigma2_to_exp(results1.x[-1])

    phi2_approx = barndorff_schou_transformation(np.tanh(results2.x[0:p2]))
    sigma22_approx = sigma2_to_exp(results2.x[-1])

    np.random.seed(seed) 

    pi_1 = {}
    pi_2 = {}

    pi_1b = {} #errors drawn from N(0,1)
    pi_2b = {}

    for j in range(repeats):

        initials1 = list(y1[-p1:N1])
        initials1.reverse()
        initials2 = list(y2[-p2:N2])
        initials2.reverse()

        seed = np.random.randint(2**16 - 1)
        #u_gen = cop.sample_unimargin() #clayton library
        u_gen = cop.random(horizon, seed = seed) #copulae
        e_1t = stats.norm.ppf(u_gen[:,0])
        e_2t = stats.norm.ppf(u_gen[:,1])
        
        y1it = []
        y2it = []
        y1itb = []
        y2itb = []
        for i in range(horizon):
            y1it.append(initials1 @ phi1_approx + np.sqrt(sigma21_approx) * e_2t[i])
            y2it.append(initials2 @ phi2_approx + np.sqrt(sigma22_approx) * e_1t[i])
            
            y1itb.append(initials1 @ phi1_approx + np.sqrt(sigma21_approx) * np.random.normal(loc=0, scale=1, size=1) )
            y2itb.append(initials2 @ phi2_approx + np.sqrt(sigma22_approx) * np.random.normal(loc=0, scale=1, size=1) )

            temp1 = initials1[0:(p1-1)] #moving window of the p last values
            temp1.insert(0, y1it[-1]) #adding the newest prediction
            initials1 = temp1

            temp2 = initials2[0:(p2-1)] #moving window of the p last values
            temp2.insert(0, y2it[-1]) #adding the newest prediction
            initials2 = temp2
        
        pi_1[j] = y1it
        pi_2[j] = y2it
        pi_1b[j] = y1itb
        pi_2b[j] = y2itb
    pi_1 = np.array(pd.DataFrame(pi_1))
    pi_2 = np.array(pd.DataFrame(pi_2))

    pi_1b = np.array(pd.DataFrame(pi_1b))
    pi_2b = np.array(pd.DataFrame(pi_2b))

    forecast1_param_copula = np.mean(pi_1, axis = 1)
    forecast2_param_copula = np.mean(pi_2, axis = 1)

    forecast1b_param = np.mean(pi_1b, axis = 1)
    forecast2b_param = np.mean(pi_2b, axis = 1)

    return pi_1, pi_2, forecast1_param_copula, forecast2_param_copula, pi_1b, pi_2b, forecast1b_param, forecast2b_param