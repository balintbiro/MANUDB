from xgboost import xgboost
import numpy as np
import pandas as pd
import streamlit as st


X=pd.DataFrame(np.random.randint(0,100,size=(100,4)),columns=list('ABCD'))
y=np.random.choice([0,1],100)

clf=xgboost.XGBClassifier()
clf.fit(X,y)
clf.get_params()