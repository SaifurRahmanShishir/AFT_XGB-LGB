import numpy as np
import pandas as pd




class aft_simulation:
    
    
    def __init__(self,n, p, r, b0, b1, sig, test_set):
        self.n = n
        self.p = p
        self.r = r
        self.b0 = b0
        self.b1 = b1
        self.scale = sig
        self.test_set = test_set

        


###################################################################################
# Generating data for model y=b0+x%*%b1+error*sigma where e~N(0,1)#
###################################################################################
    def lognormal_data(self):
        
        def r_mat(a,r):
            jeroval = a == 0
            imputed = a
            imputed[jeroval] = r
            return(imputed)
        
        diagmat = np.diag(np.ones(self.p), k = 0)
        cormat = r_mat(diagmat, self.r)              
        rr = np.transpose(np.linalg.cholesky(cormat))
        v = np.random.uniform(size = (self.n, self.p))
        xx = np.dot(v, rr)
        e = np.random.normal(size = self.n)                   
        yy = self.b0 + np.dot(xx, self.b1) + e * self.scale
                  
        from sklearn.model_selection import train_test_split
        x, xtest, y, ytest = train_test_split(xx, yy, test_size = self.test_set)
  
        return(
            y,
            ytest,
            x,
            xtest,
            np.array([self.b0, self.b1],dtype = object)
            )  


###################################################################################
# Generating data for model y=b0+x%*%b1+error*sigma where e ~ EV(0,1)#
###################################################################################
    def weibull_data(self):
        

        def r_mat(a,r):
            jeroval = a == 0
            imputed = a
            imputed[jeroval] = r
            return(imputed)
    
                
        diagmat = np.diag(np.ones(self.p), k = 0)
        cormat = r_mat(diagmat, self.r)              #cormat<-matrix(r,nrow=n,ncol=p);diag(cormat)<-1#this will work for n=p
        rr = np.transpose(np.linalg.cholesky(cormat))
        v = np.random.uniform(size = (self.n, self.p))
        xx = np.dot(v, rr)
        e = np.random.gumbel(size = self.n)                   
        yy = self.b0 + np.dot(xx, self.b1) + e * self.scale
                  
        from sklearn.model_selection import train_test_split
        x, xtest, y, ytest = train_test_split(xx, yy, test_size = self.test_set)
  
        return(
            y,
            ytest,
            x,
            xtest,
            np.array([self.b0, self.b1],dtype = object)
            )  


###################################################################################
# Generating data for model y=b0+x%*%b1+error*sigma where e ~ logistic(0,1)#
###################################################################################
    def loglogistic_data(self):
        

        def r_mat(a,r):
            jeroval = a == 0
            imputed = a
            imputed[jeroval] = r
            return(imputed)
    
                
        diagmat = np.diag(np.ones(self.p), k = 0)
        cormat = r_mat(diagmat, self.r)              #cormat<-matrix(r,nrow=n,ncol=p);diag(cormat)<-1#this will work for n=p
        rr = np.transpose(np.linalg.cholesky(cormat))
        v = np.random.uniform(size = (self.n, self.p))
        xx = np.dot(v, rr)
        e = np.random.logistic(size = self.n)                   
        yy = self.b0 + np.dot(xx, self.b1) + e * self.scale
                  
        from sklearn.model_selection import train_test_split
        x, xtest, y, ytest = train_test_split(xx, yy, test_size = self.test_set)
  
        return(
            y,
            ytest,
            x,
            xtest,
            np.array([self.b0, self.b1],dtype = object)
            )  





