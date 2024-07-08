import streamlit as st

import xgboost
import numpy as np
import pandas as pd

st.header("MANUDB")
st.write('Lorem Ipsum'*100)

with st.sidebar:
    st.markdown("[MANUDB](#manudb)")
    st.markdown("[Export](#export)")
    st.markdown("[Predict](#predict)")

st.header("Export")
st.write("Lorem Ipsum"*100)

st.header("Predict")
st.write("Lorem Ipsum"*100)
X=pd.DataFrame(np.random.randint(0,100,size=(100,4)),columns=list('ABCD'))
y=np.random.choice([0,1],100)

'''clf=xgboost.XGBClassifier()
clf.fit(X,y)
st.write(clf.get_params())'''
st.write(xgboost.__version__)