



def loglogistic_censored_sim(n, p, r, b0, b1, sig, test_set, c_percentage, c_lower, c_higher):
    
    
    import numpy as np
    import pandas as pd 
    from survival_simulation import aft_model_sim as aft_sim
    
    

    
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


    


 
    
    Cper = (c_percentage - 50)/100
    
    while True:

        a = aft_sim.aft_simulation(n, p, r, b0 , b1, sig, test_set)
        
        y_train, y_test, X_train, X_test, _ = a.loglogistic_data()
        n_tr = X_train.shape[0]
        p_tr = X_train.shape[1]
        n_tt = X_test.shape[0]


        
        
        c_tr = np.random.uniform(low = np.min(y_train), 
                                 high = np.max(y_train) - np.max(y_train)*Cper, size = n_tr)
        c_tt = np.random.uniform(low = np.min(y_test), 
                                 high = np.max(y_test) - np.max(y_test)*Cper, size = n_tt)
        
        t_tr = []
        d_tr = [] 
        t_tt = []
        d_tt = []


        t_tr = np.where(y_train < c_tr, y_train, c_tr)
        d_tr = np.where(y_train < c_tr, 1, 0)

        t_tt = np.where(y_test < c_tt, y_test, c_tt)
        d_tt = np.where(y_test < c_tt, 1, 0) 


        Pper_tr = (1 - np.sum(d_tr) / n_tr)*100
        Pper_tt = (1 - np.sum(d_tt) / n_tt)*100



        print("The censoring percentage, train: {}, test: {}".format(Pper_tr, Pper_tt))
        if((Pper_tr >= c_lower - 0.7  and Pper_tr <= c_higher + 0.7) and (Pper_tt >= c_lower - 0.7  and Pper_tt <= c_higher + 0.7)):
            break         

        
    d1 = pd.DataFrame({'time': t_tr, 'status': d_tr})
    col_names = []
    for i in range(1, p_tr+1):
        col_names.append('X'+str(i))

    dat_tr = pd.merge(d1, pd.DataFrame(X_train, columns = col_names), left_index = True, right_index = True)


    d2 = pd.DataFrame({'time': t_tt, 'status': d_tt})
    
    dat_tt = pd.merge(d2, pd.DataFrame(X_test, columns = col_names), left_index = True, right_index = True)

        
    return dat_tr, dat_tt, Pper_tr, Pper_tt





################### Normal



def lognormal_censored_sim(n, p, r, b0, b1, sig, test_set, c_percentage, c_lower, c_higher):
    
    
    import numpy as np
    import pandas as pd 
    from survival_simulation import aft_model_sim as aft_sim
    
    

    
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


    


 
    
    Cper = (c_percentage - 50)/100
    
    while True:

        a = aft_sim.aft_simulation(n, p, r, b0 , b1, sig, test_set)
        
        y_train, y_test, X_train, X_test, _ = a.lognormal_data()
        n_tr = X_train.shape[0]
        p_tr = X_train.shape[1]
        n_tt = X_test.shape[0]


        
        
        c_tr = np.random.uniform(low = np.min(y_train), 
                                 high = np.max(y_train) - np.max(y_train)*Cper, size = n_tr)
        c_tt = np.random.uniform(low = np.min(y_test), 
                                 high = np.max(y_test) - np.max(y_test)*Cper, size = n_tt)
        
        t_tr = []
        d_tr = [] 
        t_tt = []
        d_tt = []


        t_tr = np.where(y_train < c_tr, y_train, c_tr)
        d_tr = np.where(y_train < c_tr, 1, 0)

        t_tt = np.where(y_test < c_tt, y_test, c_tt)
        d_tt = np.where(y_test < c_tt, 1, 0) 


        Pper_tr = (1 - np.sum(d_tr) / n_tr)*100
        Pper_tt = (1 - np.sum(d_tt) / n_tt)*100



        print("The censoring percentage, train: {}, test: {}".format(Pper_tr, Pper_tt))
        if((Pper_tr >= c_lower - 0.7  and Pper_tr <= c_higher + 0.7) and (Pper_tt >= c_lower - 0.7  and Pper_tt <= c_higher + 0.7)):
            break         

        
    d1 = pd.DataFrame({'time': t_tr, 'status': d_tr})
    col_names = []
    for i in range(1, p_tr+1):
        col_names.append('X'+str(i))

    dat_tr = pd.merge(d1, pd.DataFrame(X_train, columns = col_names), left_index = True, right_index = True)


    d2 = pd.DataFrame({'time': t_tt, 'status': d_tt})
    
    dat_tt = pd.merge(d2, pd.DataFrame(X_test, columns = col_names), left_index = True, right_index = True)

        
    return dat_tr, dat_tt, Pper_tr, Pper_tt











################### Weibull



def weibull_censored_sim(n, p, r, b0, b1, sig, test_set, c_percentage, c_lower, c_higher):
    
    
    import numpy as np
    import pandas as pd 
    from survival_simulation import aft_model_sim as aft_sim
    
    

    
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


    


 
    
    Cper = (c_percentage - 50)/100
    
    while True:

        a = aft_sim.aft_simulation(n, p, r, b0 , b1, sig, test_set)
        
        y_train, y_test, X_train, X_test, _ = a.weibull_data()
        n_tr = X_train.shape[0]
        p_tr = X_train.shape[1]
        n_tt = X_test.shape[0]


        
        
        c_tr = np.random.uniform(low = np.min(y_train), 
                                 high = np.max(y_train) - np.max(y_train)*Cper, size = n_tr)
        c_tt = np.random.uniform(low = np.min(y_test), 
                                 high = np.max(y_test) - np.max(y_test)*Cper, size = n_tt)
        
        t_tr = []
        d_tr = [] 
        t_tt = []
        d_tt = []


        t_tr = np.where(y_train < c_tr, y_train, c_tr)
        d_tr = np.where(y_train < c_tr, 1, 0)

        t_tt = np.where(y_test < c_tt, y_test, c_tt)
        d_tt = np.where(y_test < c_tt, 1, 0) 


        Pper_tr = (1 - np.sum(d_tr) / n_tr)*100
        Pper_tt = (1 - np.sum(d_tt) / n_tt)*100



        print("The censoring percentage, train: {}, test: {}".format(Pper_tr, Pper_tt))
        if((Pper_tr >= c_lower - 0.7  and Pper_tr <= c_higher + 0.7) and (Pper_tt >= c_lower - 0.7  and Pper_tt <= c_higher + 0.7)):
            break         

        
    d1 = pd.DataFrame({'time': t_tr, 'status': d_tr})
    col_names = []
    for i in range(1, p_tr+1):
        col_names.append('X'+str(i))

    dat_tr = pd.merge(d1, pd.DataFrame(X_train, columns = col_names), left_index = True, right_index = True)


    d2 = pd.DataFrame({'time': t_tt, 'status': d_tt})
    
    dat_tt = pd.merge(d2, pd.DataFrame(X_test, columns = col_names), left_index = True, right_index = True)

        
    return dat_tr, dat_tt, Pper_tr, Pper_tt




















