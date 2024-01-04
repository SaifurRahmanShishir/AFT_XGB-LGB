import numpy as np
import pandas as pd
import seaborn as sns
import pickle

import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from survival_simulation import aft_model_sim 
from survival_simulation.censored_sim import censored_data 
from stutes_weight import stute as st
from sklearn.model_selection import GridSearchCV






def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv, scoring_fit,
                       do_probabilities = False):
    gs = GridSearchCV(
        estimator = model,
        param_grid = param_grid, 
        cv = cv, 
        n_jobs = 4, 
        scoring = scoring_fit,
        verbose = 1
    )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
        pred = fitted_model.predict_proba(X_test_data)
    else:
        pred = fitted_model.predict(X_test_data)
    
    return fitted_model, pred







df_DLC = pd.read_csv("DLBCL.csv")


train_set = df_DLC.iloc[0:160,:]
test_set = df_DLC.iloc[160:,:]

y_train = np.log(train_set['time'])
y_test = np.log(test_set['time'])

ind_train = train_set['status']
ind_test = test_set['status']

X_train = train_set.iloc[:,3:]
X_test = test_set.iloc[:,3:]



# stutes

st_train = st.stutes_weighted_data(X = X_train, Y = y_train, delta = ind_train).weighted_data()
st_test = st.stutes_weighted_data(X = X_test, Y = y_test, delta = ind_test).weighted_data()

st_tr = st_train.loc[st_train['status'] == 1]
st_tt = st_test.loc[st_test['status'] == 1]

yc_train = st_tr['time']
Xc_train = st_tr.iloc[:,:-2]

yc_test = st_tt['time']
Xc_test = st_tt.iloc[:,:-2]




params_grid_lgb = {
    'boosting_type': ['gbdt'],
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.1, 0.01, 0.001],
    'min_child_weight': [0.01, 1, 10, 100],
    'reg_alpha': [0.001, 0.01, 0.1, 1, 10],
    'reg_lambda': [0.001, 0.01, 0.1, 1, 10],
    'importance_type': ['gain']
    
}

mod = lgb.LGBMRegressor()
gs_cv, pred = algorithm_pipeline(X_train_data = Xc_train, X_test_data = Xc_test,
                                 y_train_data = yc_train, y_test_data = yc_test,
                   model = mod, cv = 5, scoring_fit = 'neg_mean_squared_error', param_grid = params_grid_lgb,
                               do_probabilities = False)

model_lgb = gs_cv.best_estimator_
with open('LGB_estimator.pkl', 'wb') as w:
    pickle.dump(model_lgb, w)




params_grid_xgb = {
    'booster': ['gbtree'],
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5, 6],
    'eta': [0.1, 0.01, 0.001],
    'min_child_weight': [0.01, 1, 10, 100],
    'reg_alpha': [0.001, 0.01, 0.1, 1, 10],
    'reg_lambda': [0.001, 0.01, 0.1, 1, 10],
    'importance_type': ['gain']
    
}

mod = xgb.XGBRegressor() 
gs_cv, pred = algorithm_pipeline(X_train_data = Xc_train, X_test_data = Xc_test,
                                 y_train_data = yc_train, y_test_data = yc_test,
                   model = mod, cv = 5, scoring_fit = 'neg_mean_squared_error', param_grid = params_grid_xgb,
                               do_probabilities = False)

model_xgb = gs_cv.best_estimator_
with open('XGB_estimator.pkl', 'wb') as w:
    pickle.dump(model_xgb, w)















