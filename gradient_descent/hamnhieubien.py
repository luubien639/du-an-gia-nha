from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(2) #[cite: 3]

# Tạo 1000 điểm dữ liệu
X = np.random.rand(1000, 1) #[cite: 3]
y = 4 + 3 * X + .2*np.random.randn(1000, 1) #[cite: 3]

# Xây dựng Xbar
one = np.ones((X.shape[0],1)) #[cite: 3]
Xbar = np.concatenate((one, X), axis = 1) #[cite: 3]

def grad(w):
    N = Xbar.shape[0] #[cite: 3]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y) #[cite: 3]

def cost(w):
    N = Xbar.shape[0] #[cite: 3]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2 #[cite: 3]

def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1]) #[cite: 3]
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3: #[cite: 3]
            break 
        w.append(w_new)
    return (w, it) 

w_init = np.array([[2], [1]]) #[cite: 3]
(w1, it1) = myGD(w_init, grad, 1) #[cite: 3]
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1)) #[cite: 3]