import numpy as np

# Do not change function signatures
#
# input:
#   X is the input matrix of size n_samples x n_features.
#   pass the parameters of the kernel function via kwargs.
# output:
#   Kernel matrix of size n_samples x n_samples 
#   K[i][j] = f(X[i], X[j]) for kernel function f()

def linear(X: np.ndarray, **kwargs)-> np.ndarray:
    m = X.shape[0] #number of samples
    if(m==2):
        return np.dot(X[0], X[1])
    kernel_matrix = X @ X.T
    return kernel_matrix

def polynomial(X:np.ndarray,**kwargs)-> np.ndarray:
    assert X.ndim == 2
    m = X.shape[0]
    # polynomial transformation given by (eta + gamma (x^T . x'))^Q
    c = np.full((m, m), kwargs['c'])
    gamma = kwargs['gamma']
    q = kwargs['q']
    kernel_matrix =  (c + gamma * (X @ X.T))**q
    return kernel_matrix

def rbf(X:np.ndarray,**kwargs)-> np.ndarray:
    assert X.ndim == 2
    m = X.shape[0]
    gamma = kwargs['gamma']
    kernel_matrix = np.zeros([m,m])
    if(m==2):
        return np.exp((-1)*(gamma)*((np.linalg.norm(X[0]-X[1]))**2))
    for i in range(m):
         for j in range(m):
            kernel_matrix[i][j] = np.exp((-1)*(gamma)*((np.linalg.norm(X[i]-X[j]))**2))
    return kernel_matrix

def sigmoid(X:np.ndarray,**kwargs)-> np.ndarray:
    assert X.ndim == 2
    m = X.shape[0]
    gamma = kwargs['gamma']
    r = kwargs['r']
    kernel_matrix = np.tanh((gamma * (X @ X.T)) + r)
    return kernel_matrix

def laplacian(X:np.ndarray,**kwargs)-> np.ndarray:
    assert X.ndim == 2
    m = X.shape[0]
    kernel_matrix = np.zeros([m,m])
    gamma = kwargs['gamma']
    for i in range(m):
         for j in range(m):
            kernel_matrix[i][j] = np.exp((-1)*(gamma)*(np.linalg.norm(X[i]-X[j],ord=1)))
    return kernel_matrix