import sqlite3
import pandas as pd
import streamlit as st

import os
st.write(os.getwd())
st.write(os.listdir('../data/'))

'''
#connect to DB and initialize cursor
connection=sqlite3.connect('../../data/MANUDB.db')
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
        label='Please select an option',
        placeholder='Please select an option',
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
            key='download-csv'
        )

st.divider()

st.header(
    '''
    Export option
    '''
)
st.markdown(
    '''<div style="text-align: justify;">
    This subpage makes it possible to export the selection of NUMTs based on 
    the selected organism name. 
    Just click on the dropdown above and check the available options. 
    Right now MANUDB supports <a href="https://en.wikipedia.org/wiki/Comma-separated_values">
    .csv format</a> exports. During the export one can download 14 features 
    (e value, genomic identifier, genomic start position, mitochondrial start position, genomic length,
    mitochondrial length, genomic strand, mitochondrial strand, genomic size, genomic sequence, 
    mitochondrial sequence, genus name, family name and order name).
    </div>''',
    unsafe_allow_html=True
)'''