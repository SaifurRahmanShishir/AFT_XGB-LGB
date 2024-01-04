import numpy as np 
import pandas as pd


class stutes_weighted_data:
    
    

    
    
    def __init__(self, X, Y, delta):
        self.X = X
        self.Y = Y
        self.delta = delta
        self.n = X.shape[0]
        self.p = X.shape[1]
        
    def aft_kmweight(self):
        
        """
  y : observed data 
  delta: Censoring indicator
  kmweights[1] = 1/n, 
  kmweights[i] = kmweights[i-1]*(n-i+2)/(n-i+1)*(((n-i+1)/(n-i+2))^delta[i-1])
  for censored data, we set 1/n too.
        
        """
  
    
        y_ar = np.array(self.Y)
        d_ar = np.array(self.delta)
        n = len(y_ar)
        
        if(n != len(d_ar)):
            raise IndexError("length of Y and delta don't match!")
        sy = np.sort(y_ar)
        sdelta = d_ar[y_ar.argsort()]
    
        kweights = []
        kweights.append(1/n)
    
        for i in range(1, n):
            k = kweights[i-1] * (n-i+2)/(n-i+1) * (((n-i+1)/(n-i+2))**sdelta[i-1])
            kweights.append(k)    
    
        kmwts = kweights*sdelta   
        
        if (sdelta[n-1] == 0):
            kmwts[n-1] = 1 - np.sum(kmwts)
            
        return(kmwts)
 
    
    def weighted_data(self):
        
        
        kw = self.aft_kmweight()
        y_ar = np.array(self.Y)
        d_ar = np.array(self.delta)
        sy = np.sort(y_ar)
        sdelta = d_ar[y_ar.argsort()]
        sx = np.array(self.X)[y_ar.argsort()]
        
        Xw = np.sum((np.transpose(sx[sdelta == 1])*kw[sdelta == 1]).transpose(), axis = 0) / np.sum(kw[sdelta == 1])
        Yw = np.sum(sy[sdelta == 1]*kw[sdelta == 1]) / np.sum(kw[sdelta == 1])
        self.Yw = Yw

        for i in range(self.n):
            sx[i] = sx[i] - Xw
        
        x = (np.sqrt(kw) * np.transpose(sx)).transpose()
        y = np.sqrt(kw) * (sy - Yw)
        
        col_names = []
        for i in range(1, self.p+1):
            col_names.append('X'+str(i))

        dat = pd.DataFrame(x, columns = col_names)
        dat['status'] = sdelta
        dat['time'] = y
        
        dat_final = dat.sample(frac=1).reset_index(drop=True)
        return(dat_final)
    
