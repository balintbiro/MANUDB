import json
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

connection=sqlite3.connect('MANUDBrev.db')

trial=pd.read_sql_query("SELECT * FROM Statistic",connection)

if "available_options" not in st.session_state:
    st.session_state.available_options = trial.columns#alignemtn score, evalue, e2value, id, seq_identity
if "selected_items" not in st.session_state:
    st.session_state.selected_items = []

def update_selection(selected):
    st.session_state.selected_items = selected
    st.session_state.available_options = [
        item for item in trial.columns
        if item not in selected
    ]

columns=st.multiselect(
		label="Sample",options=st.session_state.available_options
	)

st.dataframe(trial[columns])