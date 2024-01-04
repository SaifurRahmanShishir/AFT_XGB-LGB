



def censored_data(x, y, c_percentage, c_lower, c_higher):
    
    import numpy as np
    import pandas as pd 
    
    '''
   y: Survive time
   X: Predictors with dimension n X p
   Cper: Quantity to obtain censoring percentage(Pper) as Cper=(Pper-50)/100
   for example, Cper is -0.20,0,0.20 to obtain respective Pper 30,50,70
   Censored observation (c) is calculated from Uniform distribution as
   c=runif(n,range(y)[1],range(y)[2]-range(y)[2]*Cper)
   
   Returned value:
   
   time = min(y,c), 
   dataset = data.frame('time', 'status', X)
   

    '''
 
    n = x.shape[0]
    p = x.shape[1]
    Cper = (c_percentage - 50)/100
    
    while True:
        
        n = len(y)
        c = np.random.uniform(low = np.min(y), high = np.max(y) - np.max(y)*Cper, size = n)
        t = []
        d = []                      

        
        t = np.where(y < c, y, c)
        d = np.where(y < c, 1, 0)
        Pper = (1 - np.sum(d) / n)*100
        print("The censoring percentage: ", Pper)
        if(Pper >= c_lower - 0.7  and Pper <= c_higher + 0.7 ): 
            break         
        
    d1 = pd.DataFrame({'time': t, 'status': d})
    col_names = []
    for i in range(1, p+1):
        col_names.append('X'+str(i))

    dat = pd.merge(d1, pd.DataFrame(x, columns = col_names), left_index = True, right_index = True)
        
    return dat, Pper

