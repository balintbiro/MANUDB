#importations
import sqlite3
import pandas as pd

#read in the data from the previous version of DB
connection=sqlite3.connect("MANUDBrev.db")
stat=pd.read_sql_query("SELECT * FROM statistic",connection)
loc=pd.read_sql_query("SELECT * FROM location",connection)
gen=pd.read_sql_query("SELECT * FROM genes",connection)
tax=pd.read_sql_query("SELECT * FROM taxonomy",connection)

#clean the genes table-> dropnas
mtgen=gen[["id","mt_gene"]].dropna(subset="mt_gene")
ugen=gen[["id","gene"]].dropna(subset="gene")

#join the statistic and location tables as they are on the NUMTs level (+sequences)
statloc=loc.join(stat.set_index("id"),on="id")

#adding organism name. This will be the field for joining with the taxonomy table
statloc["organism_name"]=statloc["id"].str.split("_").str[:-1].str.join('_')

#connect to DB and initialize cursor
newconnection=sqlite3.connect('MANUDBrev2.db')
cursor=newconnection.cursor()

#create table
cursor.execute(
        """CREATE TABLE general_info(
            id VARCHAR,
            genomic_id VARCHAR,
            genomic_size INT,
            genomic_start INT,
            mitochondrial_start INT,
            genomic_length INT,
            mitochondrial_length INT,
            genomic_strand CHAR(1),
            mitochondrial_strand CHAR(1),
            eg2_value FLOAT,
            e_value FLOAT,
            seq_identity FLOAT,
            alignment_score INT,
            organism_name VARCHAR
        )"""
    )
#create table
cursor.execute(
        """CREATE TABLE taxonomy(
            organism_name VARCHAR,
            taxonomy_order VARCHAR,
            family VARCHAR,
            genus VARCHAR,
            assembly_version VARCHAR
        )"""
    )

#create table
cursor.execute(
        """CREATE TABLE mt_gene(
            id VARCHAR,
            mt_gene VARCHAR
        )"""
    )

#create table
cursor.execute(
        """CREATE TABLE gene(
            id VARCHAR,
            gene VARCHAR
        )"""
    )


#sanity check->should list all the table names
result=cursor.execute("SELECT name FROM sqlite_master")
result.fetchall()

#function for getting elements in list of tuples
def fillDB(cols_list:list,query:str, input_df:pd.DataFrame)->None:
    tuples_list=[]
    for index,row in input_df[cols_list].iterrows():
        tuples_list.append(tuple(row.values))
    cursor.executemany(query, tuples_list)
    newconnection.commit()

#fillDB with elements
fillDB(
    cols_list=[
        'id','genomic_id','genomic_size','genomic_start','mitochondrial_start',
        'genomic_length','mitochondrial_length','genomic_strand','mitochondrial_strand',
        'eg2_value','e_value','seq_identity','alignment_score','organism_name'
    ],
    query="INSERT INTO general_info VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    input_df=statloc
)
#fillDB with elements
fillDB(
    cols_list=[
        'organism_name','taxonomy_order','genus','family','assembly_version'
    ],
    query="INSERT INTO taxonomy VALUES(?,?,?,?,?)",
    input_df=tax
)
#fillDB with elements
fillDB(
    cols_list=[
        'id','mt_gene'
    ],
    query="INSERT INTO mt_gene VALUES(?,?)",
    input_df=mtgen
)
#fillDB with elements
fillDB(
    cols_list=[
        'id','gene'
    ],
    query="INSERT INTO gene VALUES(?,?)",
    input_df=ugen
)

#sanity checks
pd.read_sql_query("""SELECT * FROM general_info where organism_name LIKE '%mus_musculus%'""",newconnection)
pd.read_sql_query("""SELECT * FROM taxonomy where organism_name LIKE '%mus_musculus%'""",newconnection)
pd.read_sql_query("""SELECT * FROM gene where id LIKE '%mus_musculus%'""",newconnection)
pd.read_sql_query("""SELECT * FROM mt_gene where id LIKE '%mus_musculus%'""",newconnection)

#closing the DB
newconnection.close()