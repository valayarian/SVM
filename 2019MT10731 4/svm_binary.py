from typing import List
import numpy as np
import pandas as pd
from qpsolvers import solve_qp
import qpsolvers
class Trainer:
    def __init__(self,kernel,C=None,**kwargs) -> None:
        self.kernel = kernel
        self.kwargs = kwargs
        self.C=C
        self.support_vectors:List[np.ndarray] = []
        
    
    def fit(self, train_data_path:str)->None:
        
        df_train = pd.read_csv(train_data_path)
        df_train.pop(df_train.columns[0])
        X = df_train.loc[: , df_train.columns!='y' ] 
        y = df_train['y']
        y = y.astype(float)
        #To convert 0s to -1
        y = 2*y-1
        
        #no. of samples

        m = X.shape[0]
        X1 = np.array(X)
        K = self.kernel(X1, **self.kwargs)
  


        P = np.outer(y, y) * K
        q = -1 * np.ones(m)
        
        #Hard margin SVM case, hence no upper bounds on alpha
        if self.C is None:
            G = np.eye(m) * -1
            h = np.zeros(m)
        #Soft margin case, alpha has upper bound.
        else:
            G_max = np.eye(m) * -1
            G_min = np.eye(m)
            G = np.vstack((G_max, G_min))
            h_max = np.zeros(m)
            h_min = np.ones(m) * self.C
            h = np.hstack((h_max, h_min))
            
        A = y.values.reshape((1, m))
        b = np.zeros(1)
        alpha = solve_qp(P, q, G, h, A, b, solver="osqp")
   
        idx = np.where(alpha > 0)[0]
        s=idx[0]
        
        df1 = X.iloc[idx]    
        for row in df1.values:
            self.support_vectors.append(row.tolist())
  
        self.support_vector_labels = y[idx]
        self.support_vector_labels = self.support_vector_labels.reset_index(drop=True)
        self.support_vector_alpha = alpha[idx]
    
        sum = 0
        for i in range(m):
            sum = sum + y[i]*alpha[i]*K[i][s]
        
        biasterm = y[s] - sum
        self.bias_term = biasterm
        
    
    def predict(self, test_data_path:str)->np.ndarray:
       
        df_test = pd.read_csv(test_data_path)
        df_test.pop(df_test.columns[0])
        
        X_val = df_test.loc[: , df_test.columns!='y' ]
        y = np.array(df_test['y'])
        m = X_val.shape[0]
        n = len(self.support_vectors)
        X = np.array(X_val)
        y_pred = [0 for i in range(m)] 
    
        for i in range(m):
            for j in range(n):
                y_pred[i] = y_pred[i] + (self.support_vector_labels[j])*(self.support_vector_alpha[j])*(self.kernel(np.array([self.support_vectors[j],X[i]]),**self.kwargs))
            
            
        y_pred_final = [0]*m
        for i in range(m):
            y_pred_final[i] = (np.sign(y_pred[i] + self.bias_term)+1)/2
              
        return y_pred_final
        
        
