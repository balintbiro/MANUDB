import pickle
import numpy as np
import pandas as pd
import streamlit as st
from itertools import product
st.set_page_config(page_title='Predict')


trained_clf=pickle.load(open('results/optimized_model.pkl','rb'))
best_features=pd.read_csv('results/best_features.csv',index_col=0)['0'].tolist()


k=3
bases=list('ACGT')
kmers=[''.join(p) for p in product(bases, repeat=k)]


st.html(
    '''
    <style>
    hr {
        border-color: green;
    }
    </style>
    '''
)


if 'text_area_content' not in st.session_state:
    st.session_state.text_area_content=''


examples='''>Example_Sequence.1\nAATGCTATTAGGGTCTGAGAGACTCTGCGAGTATAGCGGTTAGCGGCTATAGCGATCGATCAGCTACGATCTACGACTATCCA\n>Example_Sequence.2\nATGGTTTTTTTTGGGGTTACGTACGTNNNNNATATCGCGGCTACGGCTCGATCGGTTGCTACG'''


def populate_example():
    st.session_state.text_area_content=examples


def clear():
    st.session_state.text_area_content=''


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


def predict():
    if st.session_state.text_area_content!='':
        items=st.session_state.text_area_content.split('\n')
        headers,sequences=[],[]
        items=pd.Series(items)
        headers=items[items.str.startswith('>')].str[1:]
        sequences=items[~items.str.startswith('>')].str.upper()
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
        prediction['label']=trained_clf.predict(X)
        prediction['prob-NUMT']=trained_clf.predict_proba(df[best_features])[:,1]
        prediction['label']=prediction['label'].replace([1,0],['NUMT','non-NUMT'])
        #st.dataframe(prediction)
        csv = convert_df(prediction)


        st.download_button(
            f"Download MANUDB_prediction.csv",
            csv,
            f"MANUDB_prediction.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.write('No sequence found to predict. Please paste your sequence(s) or use the example to get help!')


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


st.divider()


st.header(
    '''
    Predict option
    '''
)
st.markdown(
    '''<div style="text-align: justify;">
    With this subpage one can predict whether a particular sequence is a NUMT or is not a NUMT.
    To use this functionality just simply paste your <a href="https://en.wikipedia.org/wiki/FASTA_format">
    .FASTA format</a> sequence(s) here. If you click on the 'Example' button above it will paste two 
    correctly formatted sequences into the text area. Then you can use the 'Predict' button to 
    calculate the probability that these two sequences are actual NUMTs. The output of this functionality 
    is downloadable in a .csv file which contains the FASTA headers and the corresponding predicted 
    probabilities.<br> 
    Please bear in mind that MANUDB prediction functionality is designed for mammalian sequences. 
    And so if you run prediction on non-mammalian sequences the results will be unreliable.
    </div>''',
    unsafe_allow_html=True
)


st.divider()


st.header(
    '''
    Troubleshooting common problems
    '''
)
st.markdown(
    '''<div style="text-align: justify;">
    The most common problem that we experienced during the maintainance of MANUDB is the 
    usage of wrong input format (aka intrasequence new line characters, no headers etc.):
    </div>''',
    unsafe_allow_html=True
)
bad,good=st.columns(2)
with bad.container(border=True,height=100):
   st.text('>Bad_example\nACGT\nCGGT')


with good.container(border=True,height=100):
   st.text('>Good_example\nACGTCGGT')