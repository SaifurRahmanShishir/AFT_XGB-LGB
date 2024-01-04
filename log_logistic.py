import numpy as np
import pandas as pd
import lightgbm as lgb
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import time
import xgboost as xgb
import lightgbm as lgb

from survival_simulation import aft_model_sim as aft_sim
from survival_simulation import censored_sim as c_sim
from survival_simulation import modified_censored_sim as mc_sim
from lgbm.lgbm_grad_hess import normal_distribution as norm
from survival_simulation.mixed import loglogistic_censored_sim as lg
from survival_simulation.mixed import lognormal_censored_sim as log_norm
from survival_simulation.mixed import weibull_censored_sim as wb

#from lgbm.metrics import aft_negative_loglikelihood as n_log
from stutes_weight import stute as st




########## median

def surv_median(true, pred, train):

    a = (true > pd.Series(train).median())
    b = pd.Series(pred, index = true.index) > pd.Series(train).median()
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(a,b)
    score = (cm[0][0] + cm[1][1])/len(a)
    return(score)









############# loglogistic   -    r = 0.5

import logging

# Gets or creates a logger
logger = logging.getLogger('loglogistic')  

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('30_lg_50000_3000_0.5.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)


import xgboost as xgb
import lightgbm as lgb


acc_tr_lgb = []
acc_tt_lgb = []
acc_tr_xgb = []
acc_tt_xgb = []
c_tr = []
c_tt = []
time_lgb = []
time_xgb = []

for i in range(0, 100):
        #dat = aft_sim.aft_simulation(n = 1000, p = 300, r = 0.5, 
        #                b0 = 0, b1 = np.concatenate((np.array([0.001]*150), np.array([0]*150))), sig = 2,
        #                test_set = 0.25)
        #y_train, y_test, X_train, X_test, beta = dat.loglogistic_data()                  
        #df_train, a_tr = c_sim.censored_data(x = X_train, y = y_train, c_percentage = 49, 
        #                               c_lower = 45, c_higher = 55)
        #df_test, a_tt = c_sim.censored_data(x = X_test, y = y_test, c_percentage = 49, 
        #                              c_lower = 45, c_higher = 55)

        df_train, df_test, a_tr, a_tt = lg(n = 50000, p = 3000, r = 0.5, 
                        b0 = 0, b1 = np.concatenate((np.array([0.01]*1500), np.array([0]*1500))), sig = 2,
                        test_set = 0.25, c_percentage = 70, c_lower = 27, c_higher = 33)



        c_tr.append(a_tr)
        c_tt.append(a_tt)


        logger.info('{}th iteration, train_censoring: {}, test_censoring: {}'.format(i+1, a_tr, a_tt))                              


        yc_tr = np.exp(df_train['time'])
        yc_tt = np.exp(df_test['time'])
        ind_train = df_train['status']
        ind_test = df_test['status']
        X_tr = df_train.iloc[:,2:]
        X_tt = df_test.iloc[:,2:]
        st_df_tr = st.stutes_weighted_data(X = X_tr, Y = yc_tr, delta = ind_train).weighted_data()
        st_df_tt = st.stutes_weighted_data(X = X_tt, Y = yc_tt, delta = ind_test).weighted_data()
        st_df_tr = st_df_tr.loc[st_df_tr['status'] == 1]
        st_df_tt = st_df_tt.loc[st_df_tt['status'] == 1]
        
        st_yc_tr = st_df_tr['time']
        st_yc_tt = st_df_tt['time']
        st_X_tr = st_df_tr.iloc[:,:-2]
        st_X_tt = st_df_tt.iloc[:,:-2]




        st_aft_lgb = lgb.LGBMRegressor()
        
        # updating objective function to custom
        # default is "regression"
        # also adding metrics to check different scores
        st_aft_lgb.set_params(#**{'objective': aft_norm},
            metrics = ['rmse', 'l1'], 
            boosting_type = "goss", 
            num_iterations = 300,
            n_jobs = 4, 
            max_depth = 6, 
            min_child_weight = 1,
            learning_rate = 0.1,
            reg_alpha = 0.001,
            reg_lambda = 1,
            importance_type = "gain")# early_stopping_round = 10)

        s_lgb = time.time()    
        # fitting model 
        st_aft_lgb.fit(
            st_X_tr,
            st_yc_tr,
            eval_set = [(st_X_tr, st_yc_tr)], 
            #   eval_metric = nlog_like,
            verbose = 10)
        t_lgb = (time.time() - s_lgb) 
        time_lgb.append(t_lgb)

        logger.info('{}th LGB model train time: {}'.format(i+1, t_lgb))    
        
        y_true = st_yc_tr
        y_pred = st_aft_lgb.predict(X = st_X_tr)
        acc_tr_lgb.append(surv_median(true = y_true, pred = np.exp(y_pred), train = st_yc_tr))
        acc_tt_lgb.append(surv_median(true = st_yc_tt, pred = st_aft_lgb.predict(X = st_X_tt), train = st_yc_tr))
          
         
        
        
        st_aft_xgb = xgb.XGBRegressor() 

        # updating objective function to custom
        # default is "regression"
        # also adding metrics to check different scores
        st_aft_xgb.set_params(#**{'objective': aft_norm},
               eval_metric = ['rmse', 'mae'], 
               booster = "gbtree", 
               n_estimators = 300,
               n_jobs = 4, 
               max_depth = 6,
               min_child_weight = 1,
               reg_alpha = 0.001,
               reg_lambda = 1,
               eta = 0.1,
               importance_type = "gain")# early_stopping_round = 10)

        s_xgb = time.time()       
        # fitting model 
        st_aft_xgb.fit(
            st_X_tr,
            st_yc_tr,
            eval_set = [(st_X_tr, st_yc_tr)], 
            #    eval_metric = nlog_like_sk,
            verbose = 100)
        
        t_xgb = (time.time() - s_xgb) 
        time_xgb.append(t_xgb)

        logger.info('{}th Xgboost model train time: {}'.format(i+1, t_xgb))    
    

        acc_tr_xgb.append(surv_median(true = st_yc_tr, pred = st_aft_xgb.predict(st_X_tr), train = st_yc_tr))
        acc_tt_xgb.append(surv_median(true = st_yc_tt, pred = st_aft_xgb.predict(st_X_tt), train = st_yc_tr))
        
        if (len(acc_tt_xgb) == 200):
            break


pd.DataFrame({'acc_tr_lgb': acc_tr_lgb,
             'acc_tt_lgb' : acc_tt_lgb,
             'acc_tr_xgb' : acc_tr_xgb,
             'acc_tt_xgb' : acc_tt_xgb,
             'c_tr' : c_tr, 
             'c_tt' : c_tt,
             'time_lgb': time_lgb,
             'time_xgb': time_xgb}).to_pickle('30_lg_simulation_50000_3000_0.5.pkl')















############# loglogistic   -    r = 0

import logging

# Gets or creates a logger
logger = logging.getLogger('loglogistic')  

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('30_lg_50000_3000_0.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)


import xgboost as xgb
import lightgbm as lgb


acc_tr_lgb = []
acc_tt_lgb = []
acc_tr_xgb = []
acc_tt_xgb = []
c_tr = []
c_tt = []
time_lgb = []
time_xgb = []

for i in range(0, 100):
        #dat = aft_sim.aft_simulation(n = 1000, p = 300, r = 0.5, 
        #                b0 = 0, b1 = np.concatenate((np.array([0.001]*150), np.array([0]*150))), sig = 2,
        #                test_set = 0.25)
        #y_train, y_test, X_train, X_test, beta = dat.loglogistic_data()                  
        #df_train, a_tr = c_sim.censored_data(x = X_train, y = y_train, c_percentage = 49, 
        #                               c_lower = 45, c_higher = 55)
        #df_test, a_tt = c_sim.censored_data(x = X_test, y = y_test, c_percentage = 49, 
        #                              c_lower = 45, c_higher = 55)

        df_train, df_test, a_tr, a_tt = lg(n = 50000, p = 3000, r = 0, 
                        b0 = 0, b1 = np.concatenate((np.array([0.01]*1500), np.array([0]*1500))), sig = 2,
                        test_set = 0.25, c_percentage = 70, c_lower = 27, c_higher = 33)



        c_tr.append(a_tr)
        c_tt.append(a_tt)


        logger.info('{}th iteration, train_censoring: {}, test_censoring: {}'.format(i+1, a_tr, a_tt))                              


        yc_tr = np.exp(df_train['time'])
        yc_tt = np.exp(df_test['time'])
        ind_train = df_train['status']
        ind_test = df_test['status']
        X_tr = df_train.iloc[:,2:]
        X_tt = df_test.iloc[:,2:]
        st_df_tr = st.stutes_weighted_data(X = X_tr, Y = yc_tr, delta = ind_train).weighted_data()
        st_df_tt = st.stutes_weighted_data(X = X_tt, Y = yc_tt, delta = ind_test).weighted_data()
        st_df_tr = st_df_tr.loc[st_df_tr['status'] == 1]
        st_df_tt = st_df_tt.loc[st_df_tt['status'] == 1]
        
        st_yc_tr = st_df_tr['time']
        st_yc_tt = st_df_tt['time']
        st_X_tr = st_df_tr.iloc[:,:-2]
        st_X_tt = st_df_tt.iloc[:,:-2]




        st_aft_lgb = lgb.LGBMRegressor()
        
        # updating objective function to custom
        # default is "regression"
        # also adding metrics to check different scores
        st_aft_lgb.set_params(#**{'objective': aft_norm},
            metrics = ['rmse', 'l1'], 
            boosting_type = "goss", 
            num_iterations = 300,
            n_jobs = 4, 
            max_depth = 6, 
            min_child_weight = 1,
            learning_rate = 0.1,
            reg_alpha = 0.001,
            reg_lambda = 1,
            importance_type = "gain")# early_stopping_round = 10)

        s_lgb = time.time()    
        # fitting model 
        st_aft_lgb.fit(
            st_X_tr,
            st_yc_tr,
            eval_set = [(st_X_tr, st_yc_tr)], 
            #   eval_metric = nlog_like,
            verbose = 10)
        t_lgb = (time.time() - s_lgb) 
        time_lgb.append(t_lgb)

        logger.info('{}th LGB model train time: {}'.format(i+1, t_lgb))    
        
        y_true = st_yc_tr
        y_pred = st_aft_lgb.predict(X = st_X_tr)
        acc_tr_lgb.append(surv_median(true = y_true, pred = np.exp(y_pred), train = st_yc_tr))
        acc_tt_lgb.append(surv_median(true = st_yc_tt, pred = st_aft_lgb.predict(X = st_X_tt), train = st_yc_tr))
          
         
        
        
        st_aft_xgb = xgb.XGBRegressor() 

        # updating objective function to custom
        # default is "regression"
        # also adding metrics to check different scores
        st_aft_xgb.set_params(#**{'objective': aft_norm},
               eval_metric = ['rmse', 'mae'], 
               booster = "gbtree", 
               n_estimators = 300,
               n_jobs = 4, 
               max_depth = 6,
               min_child_weight = 1,
               reg_alpha = 0.001,
               reg_lambda = 1,
               eta = 0.1,
               importance_type = "gain")# early_stopping_round = 10)

        s_xgb = time.time()       
        # fitting model 
        st_aft_xgb.fit(
            st_X_tr,
            st_yc_tr,
            eval_set = [(st_X_tr, st_yc_tr)], 
            #    eval_metric = nlog_like_sk,
            verbose = 100)
        
        t_xgb = (time.time() - s_xgb) 
        time_xgb.append(t_xgb)

        logger.info('{}th Xgboost model train time: {}'.format(i+1, t_xgb))    
    

        acc_tr_xgb.append(surv_median(true = st_yc_tr, pred = st_aft_xgb.predict(st_X_tr), train = st_yc_tr))
        acc_tt_xgb.append(surv_median(true = st_yc_tt, pred = st_aft_xgb.predict(st_X_tt), train = st_yc_tr))
        
        if (len(acc_tt_xgb) == 200):
            break


pd.DataFrame({'acc_tr_lgb': acc_tr_lgb,
             'acc_tt_lgb' : acc_tt_lgb,
             'acc_tr_xgb' : acc_tr_xgb,
             'acc_tt_xgb' : acc_tt_xgb,
             'c_tr' : c_tr, 
             'c_tt' : c_tt,
             'time_lgb': time_lgb,
             'time_xgb': time_xgb}).to_pickle('30_lg_simulation_50000_3000_0.pkl')



