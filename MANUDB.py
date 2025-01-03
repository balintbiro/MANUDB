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
import matplotlib
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
    st.markdown("[Visualize](#visualize)")

#########################################################################Export function
#connect to DB and initialize cursor
connection=sqlite3.connect('MANUDBrev2.db')

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
    #modify the variable so it will be in the same form as the SQL uses it
    organism_name=(
            organism_name
            .split('(')[0]
            .strip()
            .replace(' ','_')
        )
    general_info=pd.read_sql_query(
            f"SELECT * FROM general_info WHERE organism_name LIKE '%{organism_name}%'",
            connection
        )
    mt_genes=pd.read_sql_query(
            f"SELECT * FROM mt_gene WHERE id LIKE '%{organism_name}%'",
            connection
        )
    genes=pd.read_sql_query(
            f"SELECT * FROM gene WHERE id LIKE '%{organism_name}%'",
            connection
        )
    taxonomy=pd.read_sql_query(
            f"SELECT * FROM taxonomy WHERE organism_name LIKE '%{organism_name}%'",
            connection
        )
    gdf=pd.read_csv("genomic_sequences.csv",index_col="id")
    gdf=gdf[gdf.index.str.contains(organism_name)]

    mtdf=pd.read_csv("mitochondrial_sequences.csv",index_col="id")
    mtdf=mtdf[mtdf.index.str.contains(organism_name)]

    merged=(
        general_info
        .join(mt_genes.set_index("id"),on="id")
        .join(genes.set_index("id"),on="id")
        .join(gdf,on="id")
        .join(mtdf,on="id")
        .join(taxonomy.set_index("organism_name"),on="organism_name")
    )

    all_columns=pd.Series(merged.columns).sort_values()

    fields_to_retrieve=st.multiselect(
            label=" ",
            options=np.concatenate([all_columns,["all"]]),
            default=None,
            key="multiselect",
            placeholder="Please select the field(s) you would like to retrieve"
        )
    if fields_to_retrieve!=None:
        if (
                (fields_to_retrieve!=["all"])
                and ("all" not in fields_to_retrieve)
                and (len(set(fields_to_retrieve)&set(taxonomy.drop(columns=["organism_name"])))==0)
            ):
            #export_func.get_downloadable(organism_name=organism_name,queries=queries,query=query)
            df=merged[fields_to_retrieve]
        else:
            st.warning("""Please note that the fields of the taxonomy table
                ("taxonomy_order", "genus", "family", "assembly_version") will be 
                redundant in the resulting .csv file!\n\nThis 
                is due to the fact that there is a 'many to one' relationship between other fields 
                and the fields of the taxonomy table. For further information about 
                the SQL schema, please see the corresponding article.""")
            df=merged[all_columns]
        export_func.get_downloadable(organism_name=organism_name,df=df)

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
    #modify the variable so it will be in the same form as the SQL uses it
    organism_name=(
            organism_name
            .split('(')[0]
            .strip()
            .replace(' ','_')
        )
    numts,assembly,seq_identity=visualize_func.get_dfs(organism_name=organism_name)
    sectors,MtScaler=visualize_func.get_sectors(assembly=assembly)
    links=visualize_func.get_links(numts=numts,assembly=assembly,MtScaler=MtScaler)
    size_heatmap=pd.Series(sectors.index).apply(visualize_func.heatmap,args=(numts,sectors,MtScaler,))
    size_heatmap.index=sectors.index
    count_heatmap=pd.Series(sectors.index).apply(visualize_func.heatmap,args=(numts,sectors,MtScaler,True,))
    count_heatmap.index=sectors.index
    fig=visualize_func.plotter(
            numts=numts,sectors=sectors,links=links,
            organism_name=organism_name,size_heatmap=size_heatmap,
            count_heatmap=count_heatmap,seq_identity=seq_identity
        )
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
    org1=(
            org1
            .split('(')[0]
            .strip()
            .replace(' ','_')
        )
    org2=(
            org2
            .split('(')[0]
            .strip()
            .replace(' ','_')
        )
    orgs=[org1,org2]
    Compdf=compare.get_compdf(MtSizes=MtSizes,orgs=orgs)
    Identitydf=compare.get_seq_identity(orgs=orgs)

    fig=plt.figure(figsize=(8,10))

    # First row with 4 square subplots
    ax1=plt.subplot2grid(shape=(3,4), loc=(0, 0), colspan=1)  # (row, col)
    compare.boxplot(Compdf=Compdf,orgs=orgs,y_name="NUMT size (bp)",ax=ax1)

    ax2=plt.subplot2grid(shape=(3,4), loc=(0, 1), colspan=1)
    compare.boxplot(Compdf=Identitydf,orgs=orgs,y_name="Sequence identity",ax=ax2)

    Regdf1,Regdf2=compare.get_regdf(Compdf=Compdf,orgs=orgs)
    ax3=plt.subplot2grid(shape=(3,4), loc=(0, 2), colspan=1)
    compare.regplot(Regdf=Regdf1,color="lightblue",ax=ax3)

    ax4=plt.subplot2grid(shape=(3,4), loc=(0, 3), colspan=1)
    compare.regplot(Regdf=Regdf2,color="orange",ax=ax4)

    ax5=plt.subplot2grid(shape=(3,4), loc=(1, 0), colspan=2)
    compare.histplot(Compdf=Compdf,org=orgs[0],color="lightblue",MtSizes=MtSizes,ax=ax5)

    ax6=plt.subplot2grid(shape=(3,4), loc=(1, 2), colspan=2)
    compare.histplot(Compdf=Compdf,org=orgs[1],color="orange",MtSizes=MtSizes,ax=ax6)

    handles = [
        plt.Line2D([0], [0], color="lightblue", lw=4, label=f"""{orgs[0]} ({orgs[0][:2]} {orgs[0].split("_")[1][:2]})"""),
        plt.Line2D([0], [0], color="orange", lw=4, label=f"""{orgs[1]} ({orgs[1][:2]} {orgs[1].split("_")[1][:2]})""")
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)