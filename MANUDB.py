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


        
#########################################################################Visualize function
st.divider()
st.header("Visualize")
st.subheader("Single species usecase")
visualize_func=Visualize()
visualize_func.describe_functionality()

organism_name=st.selectbox(
    label='Please select an organism to visualize its NUMTs',
    placeholder='Please select an organism',
    options=visualize_func.get_names(),
    index=None,
    key='visualize_organism_selection'
)
#st.set_option('deprecation.showPyplotGlobalUse', False)
if organism_name!=None:
    numts,assembly=visualize_func.get_dfs(organism_name=organism_name)
    sectors,MtScaler=visualize_func.get_sectors(assembly=assembly)
    links=visualize_func.get_links(numts=numts,assembly=assembly,MtScaler=MtScaler)
    size_heatmap=pd.Series(sectors.index).apply(visualize_func.heatmap,args=(numts,sectors,MtScaler,))
    size_heatmap.index=sectors.index
    count_heatmap=pd.Series(sectors.index).apply(visualize_func.heatmap,args=(numts,sectors,MtScaler,True,))
    count_heatmap.index=sectors.index
    fig=visualize_func.plotter(numts=numts,sectors=sectors,links=links,organism_name=organism_name,size_heatmap=size_heatmap,count_heatmap=count_heatmap)
    st.pyplot(fig=fig)
    st.write(sectors["MT"])
    st.dataframe(numts)

