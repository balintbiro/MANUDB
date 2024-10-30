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
connection=sqlite3.connect('MANUDB.db')
cursor=connection.cursor()


#get the organism names by querying the DB

organism_name=st.selectbox(
    label='Please select an organism',
    placeholder='Please select an organism',
    options=numt_orgnames,
    index=None,
    key='export_organism_selection'
)


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

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
        if (query not in ["Sequence (genomic)","Sequence (mitochondrial)"]):
            csv = convert_df(pd.read_sql_query(
                queries[query].format(organism_name=organism_name.lower()),
                connection
            ))
        else:
            if query=="Sequence (genomic)":
                df=pd.read_csv("genomic_sequences.csv",index_col="id")
                df=df[df.index.str.contains(organism_name)]
                df['id']=df.index
                csv=convert_df(df)
            elif query=="Sequence (mitochondrial)":
                df=pd.read_csv("mitochondrial_sequences.csv",index_col="id")
                df=df[df.index.str.contains(organism_name)]
                df['id']=df.index
                csv=convert_df(df)

        st.download_button(
            f"Download {organism_name.lower().replace(' ','_')}_numts.csv",
            csv,
            f"{organism_name.lower().replace(' ','_')}_numts.csv",
            "text/csv",
            key='download-DBpart'
        )
#########################################################################Predict function
st.divider()
st.header("Predict")
predict_func=Predict()
predict_func.describe_functionality()

trained_clf=joblib.load('optimized_model.pkl')
best_features=pd.read_csv('best_features.csv',index_col=0)['0'].tolist()


k=3
bases=list('ACGT')
kmers=[''.join(p) for p in product(bases, repeat=k)]


st.html('''<style>hr {border-color: green;}</style>''')


if 'text_area_content' not in st.session_state:
    st.session_state.text_area_content=''


examples='''>Example_Sequence.1\nAATGCTATTAGGGTCTGAGAGACTCTGCGAGTATAGCGGTTAGCGGCTATAGCGATCGATCAGCTACGATCTACGACTATCCA\n>Example_Sequence.2\nATGGTTTTTTTTGGGGTTACGTACGTNNNNNATATCGCGGCTACGGCTCGATCGGTTGCTACG'''


def populate_example():
    st.session_state.text_area_content=examples


def clear():
    st.session_state.text_area_content=''
    if 'prediction' in st.session_state:
    	del st.session_state['prediction']


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


def predict():
    if st.session_state.text_area_content!='':
        items=st.session_state.text_area_content.split('\n')
        headers,sequences=[],[]
        items=pd.Series(items)
        headers=items[items.str.startswith('>')].str[1:]
        sequences=items[~items.str.startswith('>')].str.upper()
        sequences=sequences[sequences!='']
        kmer_counts=[]
        for sequence in sequences:
            kmer_per_seq=[]
            for kmer in kmers:
                kmer_per_seq.append(sequence.count(kmer))
            kmer_counts.append(kmer_per_seq)
        df=pd.DataFrame(data=kmer_counts,index=headers,columns=kmers)
        df=df[best_features]
        X=(df-np.mean(df))/np.std(df)
        prediction=pd.DataFrame()
        prediction['header']=headers
        prediction['label']=trained_clf.predict(X.values)
        prediction['prob-NUMT']=trained_clf.predict_proba(X.values)[:,1]
        prediction['label']=prediction['label'].replace([1,0],['NUMT','non-NUMT'])
        if 'prediction' not in st.session_state:
            st.session_state['prediction']=prediction
    else:
        st.write('No sequence found to predict. Please paste your sequence(s) or use the example to get help!')
        return None


text_area=st.text_area(
    label='Please paste your sequence(s) here',
    height=150,
    value=st.session_state.text_area_content,
    key='text_area'
)
st.session_state.text_area_content=text_area
left_column, middle_column, right_column = st.columns(3)
example=left_column.button('Example',on_click=populate_example)
clear=middle_column.button('Clear',on_click=clear)
predict=right_column.button('Predict',on_click=predict)
if 'prediction' in st.session_state:
	csv = convert_df(st.session_state['prediction'])
	st.download_button(f"Download MANUD_prediction.csv",csv,f"MANUD_prediction.csv","text/csv",key='download-prediction')
	del st.session_state['prediction']
        
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
        A.	Raw version with random colors. B. Optimized version with random colors. C. Raw version with 
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
visualize_func.describe_comparison()
col1,col2=st.columns(2)
compOrgnames=(pd.read_sql_query(
        "SELECT id FROM location",con=connection
    )["id"].str.split("_").str[:2].str.join("_")).sort_values().unique()
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
        placeholder='Select species 1',
        index=None,
        options=compOrgnames,
        key='Species2Selection'
    )

if (org1!=None) and (org2!=None):
    Compdf=pd.read_sql_query(f"SELECT * FROM location WHERE id LIKE '{org1}%' OR id LIKE '{org2}%'",con=connection)
    Compdf["Species"]=Compdf["id"].str[:2]+" "+Compdf["id"].str.split("_").str[1].str[:2]
    orgs=Compdf["Species"].unique()
    Compdf["Relative NUMT size"]=Compdf["genomic_length"]/Compdf["genomic_size"]
    Compdf["genomic_size"]=Compdf["genomic_size"]/1000_000

    fig=plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(6)

    ax1=plt.subplot2grid(shape=(3,3),loc=(0,0),colspan=1)
    sns.boxplot(data=Compdf,x="Species",y="genomic_length",ax=ax1,showfliers=False,hue="Species",order=orgs,palette=["lightblue","orange"],width=.4)
    ax1.set_ylabel("NUMT size\n(bp)")

    ax2=plt.subplot2grid(shape=(3,3),loc=(0,1),colspan=1)
    sns.boxplot(data=Compdf,x="Species",y="Relative NUMT size",ax=ax2,showfliers=False,hue="Species",order=orgs,palette=["lightblue","orange"],width=.4)

    ax3=plt.subplot2grid(shape=(3,3),loc=(1,0),colspan=1)
    Regdf=Compdf.groupby(by=["Species","genomic_id","genomic_size"])["genomic_length"].sum().reset_index()
    st.dataframe(Compdf)
    sns.regplot(data=Regdf[Regdf["Species"]==orgs[0]],x="genomic_size",y="genomic_length",ax=ax3,color="lightblue")
    ax3.set(xlabel="Size of genome part\n(Mb)",ylabel="NUMT size\n(bp)")

    ax4=plt.subplot2grid(shape=(3,3),loc=(1,1),colspan=1)
    sns.regplot(data=Regdf[Regdf["Species"]==orgs[1]],x="genomic_size",y="genomic_length",ax=ax4,color="orange")
    ax4.set(xlabel="Size of genome part\n(Mb)",ylabel="NUMT size\n(bp)")

    ax5=plt.subplot2grid(shape=(3,3),loc=(2,0),colspan=2)

    plt.tight_layout()
    st.pyplot(fig)



