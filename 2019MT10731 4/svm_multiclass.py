from typing import List
import numpy as np
import pandas as pd
from svm_binary import Trainer
import os

class Trainer_OVO:
    def __init__(self, kernel, C=None, n_classes=-1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]

    def _init_trainers(self):
        # Initiate the svm trainers for all pairs of classes
        n = self.n_classes
        for i in range(n):
            for j in range(i+1, n):
                svm = Trainer(kernel=self.kernel, C=self.C, **self.kwargs)
                self.svms.append((i,j,svm))

    def fit(self, train_data_path:str, max_iter=None)->None:
        
        df_train = pd.read_csv(train_data_path)
        df_train.pop(df_train.columns[0])
        y = np.array(df_train['y'])
        self.classes = np.unique(y)
        self.n_classes = len(np.unique(y))
        self._init_trainers()

        for i, j, svm in self.svms:
            df_ij = df_train.loc[(df_train['y'] == self.classes[i]) | (df_train['y'] == self.classes[j])]
            df_ij['y'] = df_ij['y'].replace(self.classes[i], 1)
            df_ij['y'] = df_ij['y'].replace(self.classes[j], 0)
            df_ij.to_csv("mydata.csv") 
            svm.fit("mydata.csv")
            os.remove("mydata.csv")
            
    def predict(self, test_data_path:str)->np.ndarray:
        # Load test data
        df_test = pd.read_csv(test_data_path)
        y_val = df_test['y']
        n_samples = len(df_test.index)
        y_pred = np.zeros((n_samples, self.n_classes))
        for i, j, svm in self.svms:

            y_pair_pred = svm.predict(test_data_path)
            for g in range(len(y_pair_pred)):
                if(y_pair_pred[g]==1):
                    y_pair_pred[g]= self.classes[i]
                if(y_pair_pred[g]==0):
                    y_pair_pred[g]= self.classes[j]
            
            #print(y_pair_pred)
            # Accumulate the predictions
            for k in range(n_samples):
                y_pred[k, int(y_pair_pred[k])-1] += 1
                
            # Determine the class with the most votes for each sample

        return np.argmax(y_pred, axis=1)+1









class Trainer_OVA:
    def __init__(self, kernel, C=None, n_classes=-1, **kwargs)->None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
    
    def _init_trainers(self):
        for i in range(self.n_classes):
            svm_i = Trainer(kernel=self.kernel, C=self.C, **self.kwargs)
            self.svms.append(svm_i)
    
    def fit(self, train_data_path:str, max_iter=None)->None:
        
        df_train = pd.read_csv(train_data_path)
        df_train.pop(df_train.columns[0])
        y = np.array(df_train['y'])
        self.classes = np.unique(y)
        self.n_classes = len(np.unique(y))
        self._init_trainers()

        for i in range(self.n_classes):           
            svm_i = self.svms[i]
            y_new = np.ones(len(y))
            for l in range(len(y)):
                if(y[l]!=i):
                    y_new[l]=0
            df_train['y']=y_new
            df_train.to_csv("mydata.csv") 
            svm_i.fit("mydata.csv")
            os.remove("mydata.csv")
    
    def predict(self, test_data_path:str)->np.ndarray:
        df_test = pd.read_csv(test_data_path)
        n = len(df_test.index)
        y = np.array(df_test['y'])
        
        y_pred = np.zeros((n, self.n_classes))
        
        for i in range(self.n_classes):
            svm_i = self.svms[i]
            
            y_pred[:, i] = svm_i.predict(test_data_path)
               
        return np.argmax(y_pred, axis=1)+1
