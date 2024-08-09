##################################################################################################################
#                                                                                                                #
#                                                                                                                #
#                         This file is the main file of MANUDB.                                                  #
#             It communicates with the functionalities file that contains the classes to build the DB.           #
#                                                                                                                #
#                                                                                                                #
##################################################################################################################
#import requirements
import streamlit as st

import io
import sys
import json
import joblib
import sqlite3
import xgboost
import numpy as np
import pandas as pd
from pycirclize import Circos
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

#functionalities are written into classes of a separate Python file
from functionalities import MANUDB,Export,Predict,Visualize

#get orgnames
numt_orgnames=pd.read_csv('numt_orgnames.csv',index_col=0)['organism_name'].sort_values().tolist()
assembly_orgnames=pd.read_csv('assembly_orgnames.csv',index_col=0)['0'].sort_values().str.replace('_',' ').tolist()

#set page configuration
st.set_page_config(page_title='MANUDB',initial_sidebar_state='expanded',page_icon=':cyclone:')

st.html('''<style>hr {border-color: green;}</style>''')
#########################################################################General Introduction
st.header("MANUDB")
st.header("The MAmmalian NUclear mitochondrial sequences DataBase")
#MANUDB is a general class that describe general info about AMNDUB
st.subheader('What is MANUDB?')
manudb=MANUDB()
manudb.introduction()

st.subheader('Current status and functionalities')
manudb.status()

st.subheader('Contact and/or bug report')
manudb.contact()

st.subheader('Upon usage please cite')
manudb.reference()

#sidebar for navigation between chapters
with st.sidebar:
    st.markdown("[MANUDB](#manudb)")
    st.markdown("[Export](#export)")
    st.markdown("[Predict](#predict)")
    st.markdown("[Visualize](#visualize)")

#########################################################################Export function
st.divider()
export_func=Export()
st.header("Export")
export_func.describe_functionality()


#connect to DB and initialize cursor
connection=sqlite3.connect('MANUDB_newest.db')
cursor=connection.cursor()

#########
result=cursor.execute("SELECT name FROM sqlite_master")
st.write(result.fetchall())
#########

