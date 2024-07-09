import streamlit as st

import sys
import joblib
import sqlite3
import xgboost
import numpy as np
import pandas as pd
from itertools import product

st.set_page_config(page_title='MANUDB',initial_sidebar_state='expanded')

st.html(
    '''
    <style>
    hr {
        border-color: green;
    }
    </style>
    '''
)

st.header("MANUDB")
st.header(
    '''
    What is MANUDB?
    '''
)
st.markdown(
    '''<div style="text-align: justify;">
    There is an ongoing process in which mitochondrial sequences are
    being integrated into the nuclear genome. These sequences are called NUclear MiTochondrial sequences (NUMTs)
    <a href="https://www.sciencedirect.com/science/article/pii/S0888754396901883?via%3Dihub">[1]</a>.<br>
    The importance of NUMTs has already been revealed in cancer biology
    <a href="https://link.springer.com/article/10.1186/s13073-017-0420-6">[2]</a>,
    forensic <a href="https://www.sciencedirect.com/science/article/pii/S1872497321000363">[3]</a>,
    phylogenetic studies <a href="https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000834">[4]</a>
    and in the evolution of the eukaryotic genetic information
    <a href="https://www.sciencedirect.com/science/article/pii/S1055790317302609?via%3Dihub">[5]</a>.<br> 
    Human and numerous model organisms’ genomes were described from the NUMTs point of view.
    Furthermore, recent studies were published on the patterns of these nuclear localised mitochondrial sequences
    in different taxa <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0286620">[6
    <a href="https://www.biorxiv.org/content/10.1101/2022.08.05.502930v2.full.pdf">,7].</a> <br>
    However, the results of the previously released studies are difficult to compare 
    due to the lack of standardised methods and/or using few numbers of genomes. To overcome this limitations,
    our group has already published a computational pipeline to mine NUMTs
    <a href="https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-024-10201-9">[8]</a>. Therefore, our goal with MANUDB is to
    store, visualize, predict and make NUMTs accessible that were mined with our workflow.
    </div>''',
    unsafe_allow_html=True
)
#############################
st.header(
    '''
    Current status and functionalities
    '''
)
st.markdown(
    '''<div style="text-align: justify;">
    MANUDB currently contains 79 649 NUMTs derived from 153 mammalian genomes of NCBI. These 153 genomes belong to 20 taxonomical orders.
    It supports the retrieval of species specific datasets into a format based on the end user's preference. 
    During the export one can download 14 features (e value, genomic identifier, genomic start position, mitochondrial start position, genomic length,
    mitochondrial length, genomic strand, mitochondrial strand, genomic size, genomic sequence, mitochondrial sequence, genus name, family name and order name).
    Furthermore, MANUDB makes specific NUMT visualizations accessible in downloadable format.
    It is also possible with MANUDB to perform NUMT predictions on .fasta files and on sequences.
    </div>''',
    unsafe_allow_html=True
)
#############################
st.header(
    '''
    Contact and bug report
    '''
)
st.markdown(
    '''<div style="text-align: justify;">
    MANUDB was created and is maintained by the Model Animal Genetics Group
    (PI.: Dr. Orsolya Ivett Hoffmann), Department of Animal Biotechnology,
    Institute of Genetics and Biotechnology,
    Hungarian University of Agriculture and Life Sciences.<br>
    For tehnical queries and or bug report, please contact biro[dot]balint[at]uni-mate[dot]hu or create a pull request at
    <a href="https://github.com/balintbiro/MANUDB">MANUDB's GitHub page</a>.
    </div>''',
    unsafe_allow_html=True
)
#############################
st.header(
    '''
    Upon usage please cite
    '''
)
st.markdown(
    '''<div style="text-align: justify;">
    Biró, B., Gál, Z., Fekete, Z., Klecska, E., & Hoffmann, O. I. (2024). <a href="https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-024-10201-9">
    Mitochondrial genome plasticity of mammalian species.</a> BMC genomics, 25(1), 278.
    </div>''',
    unsafe_allow_html=True
)
#############################

with st.sidebar:
    st.markdown("[MANUDB](#manudb)")
    st.markdown("[Export](#export)")
    st.markdown("[Predict](#predict)")

#########################################################################
st.divider()
st.header("Export")
st.markdown(
    '''<div style="text-align: justify;">
    This functionality makes it possible to export the selection of NUMTs based on 
    the selected organism name. 
    Just click on the dropdown above and check the available options. 
    Right now MANUDB supports <a href="https://en.wikipedia.org/wiki/Comma-separated_values">
    .csv format</a> exports. During the export one can download 14 features 
    (e value, genomic identifier, genomic start position, mitochondrial start position, genomic length,
    mitochondrial length, genomic strand, mitochondrial strand, genomic size, genomic sequence, 
    mitochondrial sequence, genus name, family name and order name).
    </div>''',
    unsafe_allow_html=True
)

#connect to DB and initialize cursor
connection=sqlite3.connect('MANUDB.db')
cursor=connection.cursor()

organism_names=(
    pd.read_sql_query(
            """
            SELECT id FROM statistic
        """,connection
    )['id']
    .str.split('_')
    .str[:2]
    .str.join('_').drop_duplicates()
    .str.capitalize()
    .str.replace('_',' ')
)
organism_name=st.selectbox(
    label='Please select an organism',
    placeholder='Please select an organism',
    options=organism_names,
    index=None
)


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

if organism_name!=None:
    queries={
        'Statistic':pd.read_sql_query(
                        f'''
                        SELECT

                        *

                        FROM statistic
                        WHERE id LIKE '%{organism_name.lower().replace(' ','_')}%'
                        ''',
                        con=connection
                    ),
        'Location':pd.read_sql_query(
                        f'''
                        SELECT

                        *

                        FROM location
                        WHERE id LIKE '%{organism_name.lower().replace(' ','_')}%'
                        ''',
                        con=connection
                    ),
        'Sequence':pd.read_sql_query(
                        f'''
                        SELECT

                        *

                        FROM sequence
                        WHERE id LIKE '%{organism_name.lower().replace(' ','_')}%'
                        ''',
                        con=connection
                    ),
        'Taxonomy':pd.read_sql_query(
                        f'''
                        SELECT

                        *

                        FROM taxonomy
                        WHERE id LIKE '%{organism_name.lower().replace(' ','_')}%'
                        ''',
                        con=connection
                    ).drop_duplicates(subset='genus'),
        'All':pd.read_sql_query(
                        f'''
                        SELECT

                        statistic.id,statistic.eg2_value,statistic.e_value,
                        location.genomic_id,location.genomic_start,location.mitochondrial_start,location.genomic_length,location.mitochondrial_length,location.genomic_strand,location.mitochondrial_strand,
                        sequence.genomic_sequence,sequence.mitochondrial_sequence,
                        taxonomy.taxonomy_order,taxonomy.family,taxonomy.genus


                        FROM statistic

                        JOIN location
                        ON statistic.id=location.id
                        JOIN sequence
                        ON statistic.id=sequence.id
                        JOIN taxonomy
                        ON statistic.id=taxonomy.id
                        
                        WHERE statistic.id LIKE '%{organism_name.lower().replace(' ','_')}%'
                        ''',
                        con=connection
                    )
    }

    query=st.selectbox(
        label='Please select table(s)',
        placeholder='Please select table(s)',
        index=None,
        options=['Statistic','Location','Sequence','Taxonomy','All']
    )
    if query!=None:
        csv = convert_df(queries[query])

        st.download_button(
            f"Download {organism_name.lower().replace(' ','_')}_numts.csv",
            csv,
            f"{organism_name.lower().replace(' ','_')}_numts.csv",
            "text/csv",
            key='download-DBpart'
        )
#########################################################################
st.divider()
st.header("Predict")
st.markdown(
    '''<div style="text-align: justify;">
    With this functionality one can predict whether a particular sequence is a NUMT or is not a NUMT.
    To use this functionality just simply paste your <a href="https://en.wikipedia.org/wiki/FASTA_format">
    .FASTA format</a> sequence(s) here. If you click on the 'Example' button above it will paste two 
    correctly formatted sequences into the text area. Then you can use the 'Predict' button to 
    calculate the probability that these two sequences are actual NUMTs. The output of this functionality 
    is downloadable in a <a href="https://en.wikipedia.org/wiki/Comma-separated_values">
    .csv format</a> file which contains the FASTA headers and the corresponding predicted 
    probabilities.<br> 
    Please bear in mind that MANUDB prediction functionality is designed for mammalian sequences. 
    And so if you run prediction on non-mammalian sequences the results will be unreliable.
    </div>''',
    unsafe_allow_html=True
)
trained_clf=joblib.load('optimized_model.pkl')
best_features=pd.read_csv('best_features.csv',index_col=0)['0'].tolist()


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
	st.download_button(
        f"Download MANUD_prediction.csv",
        csv,
        f"MANUD_prediction.csv",
        "text/csv",
        key='download-DBpart'
    )
    del st.session_state['prediction']
#########################################################################
st.divider()
st.header(
    '''
    References
    '''
)
st.markdown(
    '''<div style="text-align: justify;">
    [1] Lopez, J. V., Cevario, S., & O'Brien, S. J. (1996). Complete nucleotide sequences of the domestic cat (Felis catus)
    mitochondrial genome and a transposed mtDNA tandem repeat (Numt) in the nuclear genome. Genomics, 33(2), 229-246.<br>
    [2] Srinivasainagendra, V., Sandel, M. W., Singh, B., Sundaresan, A., Mooga, V. P., Bajpai, P., ... & Singh, K. K. (2017).
    Migration of mitochondrial DNA in the nuclear genome of colorectal adenocarcinoma. Genome medicine, 9, 1-15.<br>
    [3] Marshall, C., & Parson, W. (2021). Interpreting NUMTs in forensic genetics: Seeing the forest for the trees.
    Forensic Science International: Genetics, 53, 102497.
    [4] Hazkani-Covo, E., Zeller, R. M., & Martin, W. (2010). Molecular poltergeists: mitochondrial DNA copies (numts) in
    sequenced nuclear genomes. PLoS genetics, 6(2), e1000834.<br>
    [5] Nacer, D. F., & do Amaral, F. R. (2017). Striking pseudogenization in avian phylogenetics: Numts are large and
    common in falcons. Molecular phylogenetics and evolution, 115, 1-6.<br>
    [6] Hebert, P. D., Bock, D. G., & Prosser, S. W. (2023). Interrogating 1000 insect genomes for NUMTs: A risk
    assessment for estimates of species richness. PLoS One, 18(6), e0286620.<br>
    [7] Biró, B., Gál, Z., Brookman, M., & Hoffmann, O. I. (2022). Patterns of numtogenesis in sixteen different mice strains.
    bioRxiv, 2022-08.<br>
    [8] Biró, B., Gál, Z., Fekete, Z., Klecska, E., & Hoffmann, O. I. (2024). Mitochondrial genome plasticity of
    mammalian species. BMC genomics, 25(1), 278.
    </div>''',
    unsafe_allow_html=True
)
