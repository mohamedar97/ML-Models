import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor# pylint: disable=import-error
from sklearn.model_selection import train_test_split, GridSearchCV # pylint: disable=import-error
from sklearn.metrics import roc_auc_score # pylint: disable=import-error
from load_data import load_data
import pickle




train_data = load_data()
# train the model
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(train_data.iloc[:, train_data.columns != 'compliance'], train_data['compliance'])
    regr_rf = RandomForestRegressor()
    grid_values = {'n_estimators': [10, 100], 'max_depth': [None, 30]}
    grid_clf_auc = GridSearchCV(regr_rf, param_grid=grid_values, scoring='roc_auc')
    grid_clf_auc.fit(X_train, y_train)
    print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
    print('Grid best score (AUC): ', grid_clf_auc.best_score_)
    filename = 'finalized_model.sav'
    pickle.dump(grid_clf_auc, open(filename, 'wb'))
    return pd.DataFrame(grid_clf_auc.predict(X_test), X_test.ticket_id) 
train_model()