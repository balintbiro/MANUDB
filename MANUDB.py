#MMMMMMMM               MMMMMMMM               AAA               NNNNNNNN        NNNNNNNNUUUUUUUU     UUUUUUUUDDDDDDDDDDDDD      BBBBBBBBBBBBBBBBB   
#M:::::::M             M:::::::M              A:::A              N:::::::N       N::::::NU::::::U     U::::::UD::::::::::::DDD   B::::::::::::::::B  
#M::::::::M           M::::::::M             A:::::A             N::::::::N      N::::::NU::::::U     U::::::UD:::::::::::::::DD B::::::BBBBBB:::::B 
#M:::::::::M         M:::::::::M            A:::::::A            N:::::::::N     N::::::NUU:::::U     U:::::UUDDD:::::DDDDD:::::DBB:::::B     B:::::B
#M::::::::::M       M::::::::::M           A:::::::::A           N::::::::::N    N::::::N U:::::U     U:::::U   D:::::D    D:::::D B::::B     B:::::B
#M:::::::::::M     M:::::::::::M          A:::::A:::::A          N:::::::::::N   N::::::N U:::::D     D:::::U   D:::::D     D:::::DB::::B     B:::::B
#M:::::::M::::M   M::::M:::::::M         A:::::A A:::::A         N:::::::N::::N  N::::::N U:::::D     D:::::U   D:::::D     D:::::DB::::BBBBBB:::::B 
#M::::::M M::::M M::::M M::::::M        A:::::A   A:::::A        N::::::N N::::N N::::::N U:::::D     D:::::U   D:::::D     D:::::DB:::::::::::::BB  
#M::::::M  M::::M::::M  M::::::M       A:::::A     A:::::A       N::::::N  N::::N:::::::N U:::::D     D:::::U   D:::::D     D:::::DB::::BBBBBB:::::B 
#M::::::M   M:::::::M   M::::::M      A:::::AAAAAAAAA:::::A      N::::::N   N:::::::::::N U:::::D     D:::::U   D:::::D     D:::::DB::::B     B:::::B
#M::::::M    M:::::M    M::::::M     A:::::::::::::::::::::A     N::::::N    N::::::::::N U:::::D     D:::::U   D:::::D     D:::::DB::::B     B:::::B
#M::::::M     MMMMM     M::::::M    A:::::AAAAAAAAAAAAA:::::A    N::::::N     N:::::::::N U::::::U   U::::::U   D:::::D    D:::::D B::::B     B:::::B
#M::::::M               M::::::M   A:::::A             A:::::A   N::::::N      N::::::::N U:::::::UUU:::::::U DDD:::::DDDDD:::::DBB:::::BBBBBB::::::B
#M::::::M               M::::::M  A:::::A               A:::::A  N::::::N       N:::::::N  UU:::::::::::::UU  D:::::::::::::::DD B:::::::::::::::::B 
#M::::::M               M::::::M A:::::A                 A:::::A N::::::N        N::::::N    UU:::::::::UU    D::::::::::::DDD   B::::::::::::::::B  
#MMMMMMMM               MMMMMMMMAAAAAAA                   AAAAAAANNNNNNNN         NNNNNNN      UUUUUUUUU      DDDDDDDDDDDDD      BBBBBBBBBBBBBBBBB 
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
import seaborn as sns
from pycirclize import Circos
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import pairwise_distances

#functionalities are written into classes of a separate Python file
from functionalities import MANUDB,Export,Predict,Visualize,Compare

#get orgnames
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
#connect to DB and initialize cursor
connection=sqlite3.connect('MANUDB.db')

st.divider()
export_func=Export(connection=connection)
st.header("Export")
export_func.describe_functionality()

#get the organism names by querying the DB
organism_name=st.selectbox(
    label='',
    placeholder='Please select an organism',
    options=export_func.get_names(),
    index=None,
    key='export_organism_selection'
)

if organism_name!=None:
    with open('queries.json')as json_file:
        queries=json.load(json_file)

    query=st.selectbox(
        label='Please select table(s)',
        placeholder='Please select table(s)',
        index=None,
        options=queries.keys(),
        key='table_selection'
    )
    if query!=None:
        export_func.get_downloadable(organism_name=organism_name,queries=queries,query=query)
#########################################################################Predict function
st.divider()
st.header("Predict")
predict_func=Predict()
predict_func.describe_functionality()

trained_clf=joblib.load('optimized_model.pkl')
best_features=pd.read_csv('best_features.csv',index_col=0)['0'].tolist()


st.html('''<style>hr {border-color: green;}</style>''')

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Initialize session state for text if not already present
if 'sequence' not in st.session_state:
    st.session_state['sequence']=""
if "prediction" not in st.session_state:
    st.session_state["prediction"]=""

# Define a single text area in Streamlit with a unique key
text_area = st.text_area(
    label="",
    placeholder='Please paste your FASTA sequence(s) here',
    height=150,
    value=st.session_state['sequence'],
    key="sequence",  # Use the same key as session state to keep it in sync
    on_change=lambda: st.session_state.update({'sequence': st.session_state.sequence})
)

# Define button columns
left_column, middle_column, right_column = st.columns(3)

# Example sequence to populate
example_sequence = """>Seq1\nACGTTTTTTACGTAGGCTAGCTCGGTTGTGTCTGTAGCTC\n>Seq2\nAGGCGTCAGCTTTTTTAGGGGGGGCTTTTTTAGGGGGGGAGCTACGCATCGCATCGAGAAATTTCTCTCTTCTCTTTTTTTTTTTTTTTACGCCCTTTATACGTTTTTTACGTAGGCTAGCTCGGTTGTGTCTGTAGCTCACGTTTTTTACGTAGGCTAGCTCGGTTGTGTCTGTAGCTC"""

# Button to populate example text
example = left_column.button(
    'Example', 
    on_click=lambda: st.session_state.update({'sequence': example_sequence})
)

# Button to clear text
clear = middle_column.button(
    'Clear', 
    on_click=lambda: st.session_state.update({'sequence': "","prediction":""})
)

# Button to trigger prediction function
predict_button = right_column.button(
    "Predict",
    on_click=predict_func.predict
)
if ("prediction" in st.session_state) and (type(st.session_state["prediction"])==pd.DataFrame):
    csv=convert_df(st.session_state['prediction'])
    st.download_button(
        label="Download MANUD_prediction.csv",
        data=csv,
        file_name="MANUD_prediction.csv",
        mime="text/csv",
        key='download-prediction'
    )
        
#########################################################################Visualize function
st.divider()
st.header("Visualize")
st.subheader("Single species usecase")
visualize_func=Visualize()
visualize_func.describe_functionality()
st.image(
     image='sample_chords.png',
     caption="""
        Figure 1. - Chord diagrams of the visualize functionality using NUMTs of the rat genome.\n\n
        A.  Raw version with random colors. B. Optimized version with random colors. C. Raw version with 
        proportional coloring. D. Optimized version with proportional coloring.
     """
)

organism_name=st.selectbox(
    label='Please select an organism to visualize its NUMTs',
    placeholder='Please select an organism',
    options=assembly_orgnames,
    index=None,
    key='visualize_organism_selection'
)
#st.set_option('deprecation.showPyplotGlobalUse', False)
if organism_name!=None:
    plot_type=st.selectbox(
        label="Please select a plot type to visualize your selected organism's NUMTs",
        placeholder='Please select a plot type',
        options=['Raw (A)','Optimized (B)','Raw with proportional coloring (C)','Optimized with proportional coloring (D)'],
        index=None,
        key='plot_type'
    )
    scalers={'Raw (A)':1,'Optimized (B)':1_000_000,'Raw with proportional coloring (C)':1,'Optimized with proportional coloring (D)':1_000_000}
    coloring={'Raw (A)':False,'Optimized (B)':False,'Raw with proportional coloring (C)':True,'Optimized with proportional coloring (D)':True}
    if plot_type!=None:
        try:
            numts,assembly=visualize_func.get_dfs(organism_name=organism_name.replace(' ','_'))
            sectors=visualize_func.get_sectors(assembly=assembly,scaler=scalers[plot_type])
            links=visualize_func.get_links(numts=numts,assembly=assembly,scaler=scalers[plot_type])
            fig=visualize_func.plotter(numts=numts,sectors=sectors,links=links,organism_name=organism_name,proportional_coloring=coloring[plot_type])
            st.pyplot(fig=fig)
            plot_format=st.selectbox(
                label='Please select a format that you wish to download',
                placeholder='Please select a format',
                options=['png','svg'],
                index=None,
                key='plot_format'
            )
            if plot_format!=None:
                mimes={'png':'image/png','svg':'image/svg+xml'}
                img=io.BytesIO()
                plt.savefig(img,format=plot_format,dpi=800)
                download_fig=st.download_button(
                    label='Download figure',
                    data=img,
                    file_name=f'MANUDB_{organism_name}_NUMTs.{plot_format}',
                    mime=mimes[plot_format]
                )
        except Exception as e:
            st.error(body='Something went wrong; please contact the maintainers at biro[dot]balint[at]uni-mate[dot]hu!')

st.subheader("Comparative usecase")
compare=Compare(connection=connection)
compare.describe_functionality()
col1,col2=st.columns(2)
compOrgnames=compare.get_names()
#st.dataframe(compOrgnames)
with col1:
    org1=st.selectbox(
        label='',
        placeholder='Select species 1',
        index=None,
        options=compOrgnames,
        key='Species1Selection'
    )

with col2:
    org2=st.selectbox(
        label='',
        placeholder='Select species 2',
        index=None,
        options=compOrgnames,
        key='Species2Selection'
    )
MtSizes=pd.read_csv("MtSizes.csv",index_col=0)["mt_size"]
if (org1!=None) and (org2!=None):
    orgs=[org1,org2]
    Compdf=compare.get_compdf(MtSizes=MtSizes,orgs=orgs)

    fig=plt.figure(figsize=(8,10))

    # First row with 4 square subplots
    ax1=plt.subplot2grid(shape=(3,4), loc=(0, 0), colspan=1)  # (row, col)
    compare.boxplot(Compdf=Compdf,orgs=orgs,y_name="NUMT size (bp)",ax=ax1)

    ax2=plt.subplot2grid(shape=(3,4), loc=(0, 1), colspan=1)
    compare.boxplot(Compdf=Compdf,orgs=orgs,y_name="Relative NUMT size",ax=ax2)

    Regdf1,Regdf2=compare.get_regdf(Compdf=Compdf,orgs=orgs)
    ax3=plt.subplot2grid(shape=(3,4), loc=(0, 2), colspan=1)
    compare.regplot(Regdf=Regdf1,color="lightblue",ax=ax3)

    ax4=plt.subplot2grid(shape=(3,4), loc=(0, 3), colspan=1)
    compare.regplot(Regdf=Regdf2,color="orange",ax=ax4)

    ax5=plt.subplot2grid(shape=(3,4), loc=(1, 0), colspan=2)
    compare.histplot(Compdf=Compdf,org=orgs[0],color="lightblue",ax=ax5)

    ax6=plt.subplot2grid(shape=(3,4), loc=(1, 2), colspan=2)
    compare.histplot(Compdf=Compdf,org=orgs[1],color="orange",ax=ax6)

    ax7=plt.subplot2grid(shape=(3,4),loc=(2,0),colspan=4)
    compare.heatmap(orgs=orgs,Compdf=Compdf,ax=ax7)

    handles = [
        plt.Line2D([0], [0], color="lightblue", lw=4, label=orgs[0]),
        plt.Line2D([0], [0], color="orange", lw=4, label=orgs[1])
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)