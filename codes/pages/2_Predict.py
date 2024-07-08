import joblib
import xgboost
import numpy as np
import pandas as pd
import streamlit as st
from itertools import product

st.set_page_config(page_title='Predict')

trained_clf=joblib.load(open('results/optimized_model.pkl','rb'))

st.write(trained_clf.get_params())