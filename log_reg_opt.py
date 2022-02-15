import numpy as np
import matplotlib.pyplot
from numpy.linalg import *
from time import *
from scipy.optimize import line_search

data = np.load("LRData.npz")
X = data['X']
y = data['y']
X_t = data['X_test']
y_t = data['y_test']

n = np.shape(X)[1]
m = np.shape(X)[0]
m_test = np.shape(X_t)[0]
lam = 1e-5

def f(beta, X, y, lam, m):
    l = np.zeros(m)
    grad = np.zeros(m)
    for i in range(m):
        e = np.exp(-y[i]*beta.dot(X[i,:]))
        l[i] = np.log(1+e)
    grad = np.mean(l)+lam*beta.dot(beta)
    return grad

def fgrad(beta, X, y, lam, m):
    n = np.shape(X)[1]
    l = np.zeros(n)
    grad = 2*lam*beta
    for i in range(m):
        e = np.exp(y[i]*beta.dot(X[i,:]))
        l = (-y[i]*X[i,:])/(1+e)
        grad = grad + l/m
    return grad

def fhesse(beta, X, y, lam, m):
    n = np.shape(X)[1]
    hesse = 2*lam*np.eye(n)
    l = np.zeros([n,n])
    for i in range(m):
        e = np.exp(y[i]*beta.dot(X[i,:]))
        l = np.outer(X[i,:],X[i,:])*e/((1+e)**2)
        hesse = hesse + l/m
    return hesse

def trustregion(f, fgrad, fhesse, trhelp, beta, eta, t, tmax, tol):
        k = 0
        b = fgrad(beta)

        while ((norm(b) > tol)&(k <= 1000)):
            A = fhesse(beta)
            c = f(beta)
            d = work(A, b, t)

            md = c + d.dot(b) + 0.5*d.dot(A@d)
            r = (c-f(beta+d))/(c-md)

            if (r < 0.25):
                t = 0.25*norm(d)

            if (r >= 0.25):
                if ((r >= 0.75)and(t == norm(d))):
                    t = np.min(2*t,tmax)

            if (r > eta):
                beta = beta + d

            b = fgrad(beta)
            k = k + 1

        return beta, c

def f0(x):
    return f(x,X,y,lam,m)
def fgrad0(x):
    return fgrad(x,X,y,lam,m)
def fhesse0(x):
    return fhesse(x,X,y,lam,m)

beta0 = np.zeros(n)
t0 = 1
tmax = 1e+4
eta = 0.05
tol = 1e-5
beta_opt, beta_value = trustregion(f0, fgrad0, fhesse0, solve, beta0, eta, t0, tmax, tol)
print('The optimal parameter is:', beta_opt)
print('Optimal value: ', beta_value)

    """
    Compute sk as new direction based on cond in work func
    """
def get_sk(g, H, t):
    Hg = H@g
    gHg = np.dot(g,Hg)
    gnorm3 = np.linalg.norm(g)**3
    gHg_gnorm3 = gHg / (t*gnorm3) #download pep 8 standard plugin 

    if(gHg <= 0):
        return 1
    else:
        return min(gHg_gnorm3, 1)

def cauchy_point(g, H):
    Hg = H@g
    cauchy_p = -(np.dot(g,g)/np.dot(g,Hg))*g #self nur definieren, wenn instant´zen in Funktion gesetzt werden self.cauchy_p unnoetig
    return cauchy_p
    """
    Compute dk as new direction based on cond in work func
    """

def work(g, H, t):

    H_inv_g = inv(H)@g

    if(0 <= self.get_sk(g, H, t) <= 1):
        return self.get_sk(g, H, t) * self.cauchy_point(g, H)

    elif(1 <= self.get_sk(g, H, t) <= 2):
        return self.cauchy_point(g, H) + (self.get_sk(g, H, t) -1) * (H_inv_g - self.cauchy_point(g, H))

    else:
        print("sk out of bounds.")
        exit()
