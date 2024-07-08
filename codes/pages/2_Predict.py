import joblib
import xgboost
import numpy as np
import pandas as pd
import streamlit as st
from itertools import product


best_features=pd.read_csv('results/best_features.csv',index_col=0)['0'].tolist()
best_params=literal_eval(
    pd.read_csv('results/param_search_result.csv',index_col=0)
    .sort_values(by='polygon_area',ascending=False)
    .iloc[0]['params']
)
features=pd.read_csv('data/NUMT_features.csv')
features=features[best_features+['label']]
X,y=features.drop(columns=['label']),features['label'].replace(['random','numt'],[0,1]).values


clf=xgboost.XGBClassifier(**best_params)
optimized_model=clf.fit(X,y)