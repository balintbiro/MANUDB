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
#                         This file contains the classes to construct MANUDB's functionalities.                  #
#                         It communicates with the main file called MANUDB.py                                    #
#                                                                                                                #
#                                                                                                                #
##################################################################################################################

import json
import joblib
import sqlite3
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pycirclize import Circos
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

class MANUDB:
    """
    General class to describe current status, functionalities, contact, bug report etc of the DB.
    """
    def __init__(self):
        self.name='MANUDB'

    def introduction(self)->st.markdown:
        return st.markdown(
            '''<div style="text-align: justify;">
            There is an ongoing process in which mitochondrial sequences are
            being integrated into the nuclear genome. These sequences are called NUclear MiTochondrial sequences (NUMTs)
            (<a href="https://www.sciencedirect.com/science/article/pii/S0888754396901883?via%3Dihub">Lopez et al., 1996</a>).<br>
            The importance of NUMTs has already been revealed in cancer biology
            (<a href="https://link.springer.com/article/10.1186/s13073-017-0420-6">Srinivasainagendra et al., 2017</a>),
            forensic (<a href="https://www.sciencedirect.com/science/article/pii/S1872497321000363">Marshall & Parson, 2021</a>),
            phylogenetic studies (<a href="https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000834">Hazkani-Covo et al., 2010</a>)
            and in the evolution of the eukaryotic genetic information
            (<a href="https://www.sciencedirect.com/science/article/pii/S1055790317302609?via%3Dihub">Nacer & Raposo do Amaral, 2017</a>).<br> 
            Human and numerous model organisms’ genomes were described from the NUMTs point of view.
            Furthermore, recent studies were published on the patterns of these nuclear localised mitochondrial sequences
            in different taxa (<a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0286620">Hebert et al., 2023; 
            <a href="https://www.biorxiv.org/content/10.1101/2022.08.05.502930v2.full.pdf"> Biró et al., 2022</a>). <br>
            However, the results of the previously released studies are difficult to compare 
            due to the lack of standardised methods and/or using few numbers of genomes. To overcome this limitations,
            our group has already published a computational pipeline to mine NUMTs
            (<a href="https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-024-10201-9">Biró et al., 2024</a>). Therefore, our goal with MANUDB is to
            store, visualize, predict and make NUMTs accessible that were mined with our workflow.
            </div>''',
            unsafe_allow_html=True
        )
    
    def status(self)->st.markdown:
        return st.markdown(
            '''<div style="text-align: justify;">
            MANUDB currently contains 100 043 NUMTs derived from 192 mammalian genomes of NCBI. These 192 genomes belong to 20 taxonomical orders.
            It supports the retrieval of species specific datasets into a format based on the end user's preference. 
            During the export one can download 14 features (e value, genomic identifier, genomic start position, mitochondrial start position, genomic length,
            mitochondrial length, genomic strand, mitochondrial strand, genomic size, genomic sequence, mitochondrial sequence, genus name, family name and order name).
            Furthermore, MANUDB makes specific NUMT visualizations accessible in downloadable format.
            It is also possible with MANUDB to perform NUMT predictions on .fasta style sequences.
            </div>''',
            unsafe_allow_html=True
        )
    
    def contact(self)->st.markdown:
        return st.markdown(
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
    
    def reference(self)->st.markdown:
        return st.markdown(
            '''<div style="text-align: justify;">
            Biró, B., Gál, Z., Fekete, Z., Klecska, E., & Hoffmann, O. I. (2024). <a href="https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-024-10201-9">
            Mitochondrial genome plasticity of mammalian species.</a> BMC genomics, 25(1), 278.
            </div>''',
            unsafe_allow_html=True
        )
    
class Export:
    """
    Class for exporting part(s) of MANUDB based on your preferred species of interest.
    """
    def __init__(self,connection:sqlite3.Connection):
        self.name='Export'
        self.connection=connection

    def describe_functionality(self):
        """
        Describe the usage of this method.
        """
        return st.markdown(
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

    def get_names(self)->np.array:
        """
        List the names that can be used to query the DB with this method.
        """
        names=(
                pd
                .read_sql_query("SELECT id FROM location",con=self.connection)
                ["id"]
                .str.split("_")
                .str[:2]
                .str.join("_")
                .drop_duplicates()
                .sort_values()
                .values
            )
        return names

    def get_downloadable(self,organism_name:str,queries:dict,query=None)->None:
        """
        Query SQL and load the part into a df which can be downloaded into a csv file.
        """
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        if query!=None:
            if (query not in ["Sequence (genomic)","Sequence (mitochondrial)"]):
                csv = convert_df(pd.read_sql_query(
                    queries[query].format(organism_name=organism_name.lower()),
                    self.connection
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
            if csv:
                st.download_button(
                    f"Download {organism_name.lower().replace(' ','_')}_numts.csv",
                    csv,
                    f"{organism_name.lower().replace(' ','_')}_numts.csv",
                    "text/csv",
                    key='download-DBpart'
                )

    
class Predict:
    """
    This class create the methods for doing prediction on DNA sequences. For further information about the training/testing and so on please
    read our artice https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-024-10201-9
    """
    def __init__(self):
        self.name='Predict'
        self.trained_clf=joblib.load('optimized_model.pkl')
        self.best_features=pd.read_csv('best_features.csv',index_col=0)['0'].tolist()

    def describe_functionality(self):
        return st.markdown(
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
    
    def predict(self):
        """
        Method for predict whether the provided sequence(s) is(are) NUMT(s).
        """
        k=3
        bases=list('ACGT')
        kmers=[''.join(p) for p in product(bases, repeat=k)]
        if st.session_state["sequence"]!="":
            items=st.session_state["sequence"].split('\n')
            headers,sequences=[],[]
            for index,item in enumerate(items):
                if ">" in item:
                    headers.append(item[1:])
                    sequences.append(items[index+1])
            kmer_counts=[]
            for sequence in sequences:
                kmer_per_seq=[]
                for kmer in kmers:
                    kmer_per_seq.append(sequence.count(kmer))
                kmer_counts.append(kmer_per_seq)
            df=pd.DataFrame(data=kmer_counts,index=headers,columns=kmers)
            df=df[self.best_features]
            X=(df-np.mean(df))/np.std(df)
            prediction=pd.DataFrame()
            prediction['header']=headers
            prediction['label']=self.trained_clf.predict(X.values)
            prediction['prob-NUMT']=self.trained_clf.predict_proba(X.values)[:,1]
            prediction['label']=prediction['label'].replace([1,0],['NUMT','non-NUMT'])
            st.session_state["prediction"]=prediction
        else:
            pass
    
class Visualize:
    """
    This class contains methods that visulize the genetic flow between mitochondria and nuclear genome.
    Once the visualization si done you can download the plot in svg or png.
    """
    def __init__(self):
        self.name='Visualize'

    def describe_functionality(self):
        return st.markdown(
            '''<div style="text-align: justify;">
            The visualize functionality makes it possible to visualize the genetic flow from the mitochondria to the
            nuclear genome. MANUDB offers four types of 
            <a href="https://en.wikipedia.org/wiki/Chord_diagram_(information_visualization)">chord diagrams</a> to 
            display NUMTs of a particular genome. 
            Unplaced and unlocalized scaffolds are plotted too in a merged form. We define raw version of visualization 
            where all genomic parts are displayed with their corresponding sizes in bp. The genomic parts are color coded 
            with randomly chosen colors (Figure 1./A). The same raw input can be visualized with proportional coloring 
            with respect to NUMT sizes. In this case the genomic parts are colored uniformly (Figure 1./C). This raw 
            version of visualization is more focused on the genomic parts of chromosomes and scaffolds rather than the 
            mitochondrion. With the help of it one can visualize the flow into individual chromosomes and scaffolds from 
            the mitochondrion. We define optimized version of visualization where all the genomic parts except the 
            mitochondrion are displayed with their corresponding sizes in Mbp. While mitochondrion is displayed in bp. 
            It is possible to color the optimized visualization version randomly or proportionally with respect to NUMT 
            sizes (Figure 1. /B and Figure 1./D respectively). The optimized visualization is focusing on the mitochondrion. 
            Contrary to the raw form of visualization this type is more appropriate if someone would like to visualize 
            the source of the NUMTs within the mitochondrion.
            </div>''',
            unsafe_allow_html=True
        )

    def get_names(self):
        with open("assemblies.json")as infile:
            assemblies=json.load(infile)
        return pd.Series(assemblies.keys()).sort_values()
    
    def get_dfs(self,organism_name)->tuple:
        with open('queries.json')as json_file:
            queries=json.load(json_file)
        #connect to DB and initialize cursor
        connection=sqlite3.connect('MANUDBrev.db')
        cursor=connection.cursor()
        numts=pd.read_sql_query(
            queries['Location'].format(organism_name=organism_name.replace(' ','_')),
            connection
        )
        alignment_scores=pd.read_sql_query(
            queries['Statistic'].format(organism_name=organism_name.replace(' ','_')),
            connection
        )["alignment_score"]
        alignment_scores=alignment_scores/numts["genomic_length"]
        with open('assemblies.json')as infile:
            assemblies=json.load(infile)
        assembly=assemblies[organism_name.replace(' ','_')]
        assembly=pd.DataFrame([
            assembly['assigned_molecule'],
            assembly['length'],
            assembly['sequence_role'],
            assembly['refseq_accn']
        ]).T
        assembly.columns=['molecule','length','role','refseq']
        assembly.loc[assembly['role'].str.contains('unlocal|scaffold'),'molecule']='scaffold'
        assembly['molecule']=assembly['molecule'].apply(lambda name: int(name) if name.isnumeric() else name)
        assembly=pd.concat([assembly[assembly['molecule']!='MT'],assembly[assembly['molecule']=='MT'].sample(1)])
        mapper=pd.Series(data=assembly['molecule'].values,index=assembly['refseq'].values)
        numts['molecule']=numts['genomic_id'].apply(lambda gid: mapper.get(gid,np.nan)).values#mapper[numts['genomic_id'].values].values
        numts=numts.dropna(subset=['molecule'])
        connection.close()
        return (numts,assembly,alignment_scores)
    
    def get_sectors(self,assembly:pd.DataFrame)->dict:
        assembly['length']=assembly['length'].astype(int)
        sectors=assembly.groupby(by='molecule')['length'].sum()
        sectors=sectors[sectors>0]
        MtScaler=int(sectors[sectors.index!="MT"].sum()/sectors["MT"])
        sectors["MT"]=sectors[sectors.index!="MT"].sum()
        sectors=pd.concat([sectors[sectors.index!="MT"],sectors[sectors.index=="MT"]])
        return (sectors,MtScaler)
    
    def get_links(self,numts:pd.DataFrame,assembly:pd.DataFrame,MtScaler:int)->list:
        mt_size=assembly[assembly['molecule']=='MT']['length'].values[0]
        fil=(numts['mitochondrial_start']+numts['mitochondrial_length'])<mt_size
        numts=numts[fil]
        links=numts[numts['molecule']!='scaffold'].apply(
            lambda row: (
                ('MT',int(row['mitochondrial_start']*MtScaler),int(row['mitochondrial_start']*MtScaler+row['mitochondrial_length']*MtScaler)),
                (row['molecule'],int(row['genomic_start']),int((row['genomic_start']+row['genomic_length'])))
            ),axis=1
        ).tolist()
        def get_scf_links(row):
            if row['genomic_id'] not in id_container:
                summation=gid_dict[id_container].sum()
                id_container.append(row['genomic_id'])
            else:
                mod_ids=pd.Series(id_container)
                mod_ids=mod_ids[mod_ids!=row['genomic_id']].tolist()
                summation=gid_dict[mod_ids].sum()
            links.append(
                    (('MT',int(row['mitochondrial_start']*MtScaler),int(row['mitochondrial_start']*MtScaler+row['mitochondrial_length']*MtScaler)),
                    ('scaffold',(summation+int(row['genomic_start'])),(summation+int(row['genomic_start'])+int(row['genomic_length']))))
                )
        id_container=[]
        scf_df=numts[numts['molecule']=='scaffold']
        clean_df=scf_df.drop_duplicates(subset=['genomic_id'])
        gid_dict=pd.Series(data=clean_df['genomic_size'].values,index=clean_df['genomic_id'].values)
        scf_df.apply(get_scf_links,axis=1)
        return links

    def heatmap(self,gid:str,numts:pd.DataFrame,sectors:dict,MtScaler:int,count=False)->list:
        if gid!="MT":
            nbins,tracker,container=20,0,[]
            heatmap_range=np.linspace(start=0,stop=sectors[gid],num=nbins,dtype=int)
            subdf=numts[numts["molecule"]==gid]
            subdf["genomic_end"]=subdf["genomic_start"]+subdf["genomic_length"]
            if subdf.shape[0]!=0:
                for limit in heatmap_range:
                    selected_df=subdf[subdf["genomic_start"]<limit]
                    numt_size=selected_df["genomic_end"]-selected_df["genomic_start"]
                    if count:
                        container.append((selected_df.shape[0])-tracker)
                        tracker=subdf[subdf["genomic_start"]<limit].shape[0]
                    else:
                        container.append(numt_size.sum()-tracker)
                        tracker=numt_size.sum()
            else:
                container=nbins*[0]
            return container
        else:
            nbins,tracker,container=100,0,[]
            heatmap_range=np.linspace(start=0,stop=sectors[gid],num=nbins,dtype=int)/MtScaler
            numts["mitochondrial_end"]=numts["mitochondrial_start"]+numts["mitochondrial_length"]
            for limit in heatmap_range:
                selected_df=numts[numts["mitochondrial_start"]<limit]
                numt_size=selected_df["mitochondrial_end"]-selected_df["mitochondrial_start"]
                if count:
                    container.append((selected_df.shape[0])-tracker)
                    tracker=numts[numts["mitochondrial_start"]<limit].shape[0]
                else:
                    container.append(numt_size.sum()-tracker)
                    tracker=numt_size.sum()
            return container

    def add_cbar(self,values:list,title:str,cbar_pos:tuple,cmap_name,ax)->None:
        norm=plt.Normalize(vmin=min(values),vmax=max(values))
        cmap=plt.get_cmap(cmap_name)
        colors=[cmap(norm(value)) for value in values]
        sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
        sm.set_array([])
        cbar_ax=plt.axes(cbar_pos)
        cbar=plt.colorbar(sm,cax=cbar_ax)
        cbar.ax.set_title(title,fontsize=8)
        cbar.ax.yaxis.label.set_verticalalignment("bottom")
        cbar.ax.yaxis.label.set_position((0.5,1.2))
        cbar.ax.yaxis.set_tick_params(labelsize=6)

    def plotter(self,numts:pd.DataFrame,sectors:dict,links:list,organism_name:str,size_heatmap:pd.Series,count_heatmap:pd.Series,alignment_scores:pd.Series)->None:
        fig,ax=plt.subplots(1,1,figsize=(7,7),subplot_kw={'projection': 'polar'})
        circos=Circos(sectors,space=2)
        fontsize=8
        for sector in circos.sectors:
            track=sector.add_track((93,100))
            track.axis(fc='black')
            if sector.name=='scaffold':
                track.text(sector.name,color='black',size=fontsize,r=120,orientation='vertical')
            elif len(str(sector.name))==2:
                track.text(sector.name,color='black',size=fontsize,r=110,orientation='vertical')
            else:
                track.text(sector.name,color='black',size=fontsize,r=110)
            hms_track=sector.add_track((85,92))
            hms_track.axis(fc="none")
            hms_track.heatmap(size_heatmap[sector.name],cmap="Greens")

            hms_track=sector.add_track((77,84))
            hms_track.axis(fc="none")
            hms_track.heatmap(count_heatmap[sector.name],cmap="Greys")
        cmap=plt.cm.coolwarm
        norm=matplotlib.colors.Normalize(vmin=min(alignment_scores),vmax=max(alignment_scores))
        sm=matplotlib.cm.ScalarMappable(cmap="seismic",norm=norm)
        for index,link in enumerate(links):
            circos.link(link[0],link[1],color=cmap(norm(alignment_scores[index])))
        circos.plotfig(ax=ax)
        plt.title(f"{organism_name.replace('_',' ')} NUMTs - MANUDB",x=.5,y=1.1)
        self.add_cbar(values=alignment_scores,title="Alignment score",cbar_pos=(-.1,.7,0.015,0.1),cmap_name="coolwarm",ax=ax)
        self.add_cbar(values=np.concatenate(size_heatmap.values),title="NUMT size (bp)",cbar_pos=(-.1,.5,0.015,0.1),cmap_name="Greens",ax=ax)
        self.add_cbar(values=np.concatenate(count_heatmap.values),title="NUMT count",cbar_pos=(-.1,.3,0.015,0.1),cmap_name="Greys",ax=ax)
        return fig


class Compare:
    def __init__(self,connection:sqlite3.Connection):
        self.name='Compare'
        self.connection=connection

    def describe_functionality(self)->st.markdown:
        return st.markdown(
            '''<div style="text-align: justify;">
            MANUDB makes it possible to visualize and comapre distinct species' NUMTs.
            To perform comparative analysis please select your species of interest.
            </div>''',
            unsafe_allow_html=True
        )

    def get_names(self)->np.array:
        return (
                pd
                .read_csv("MtSizes.csv")["orgname"]
                .sort_values()
                .values
            )

    def get_compdf(self,MtSizes:pd.Series,orgs:list)->tuple:
        Compdf=pd.read_sql_query(f"SELECT * FROM location WHERE id LIKE '{orgs[0]}%' OR id LIKE '{orgs[1]}%'",con=self.connection)
        Compdf["SpeciesFull"]=Compdf["id"].str.split("_").str[:2].str.join("_")
        Compdf=Compdf.groupby(by="SpeciesFull").apply(
            lambda subdf:
            subdf[(subdf["mitochondrial_start"]+subdf["mitochondrial_length"])<MtSizes[subdf["SpeciesFull"].unique()[0]]]
        ).reset_index(drop=True)
        Compdf["SpeciesShort"]=Compdf["SpeciesFull"].str[:2]+" "+Compdf["SpeciesFull"].str.split("_").str[1].str[:2]
        Compdf["Relative NUMT size"]=Compdf["genomic_length"]/Compdf["genomic_size"]
        Compdf["genomic_size"]=Compdf["genomic_size"]/1000_000
        Compdf.rename(columns={"genomic_length":"NUMT size (bp)"},inplace=True)
        return Compdf

    def get_regdf(self,Compdf:pd.DataFrame,orgs:list)->tuple:
        Regdf=(
            Compdf
            .groupby(by=["SpeciesFull","SpeciesShort","genomic_id","genomic_size"])["NUMT size (bp)"]
            .sum()
            .reset_index()
            )
        return (Regdf[Regdf["SpeciesFull"]==orgs[0]],Regdf[Regdf["SpeciesFull"]==orgs[1]])

    def get_seq_identity(self,orgs:list)->pd.DataFrame:
        Gseqs=pd.read_csv("genomic_sequences.csv")
        Mtseqs=pd.read_csv("mitochondrial_sequences.csv")
        seqs=Gseqs.join(Mtseqs.set_index("id"),on="id")
        seqs=seqs[
            (seqs["id"].str.contains(orgs[0]))
            |(seqs["id"].str.contains(orgs[1]))
        ]
        seqs["genomic_sequence"]=seqs["genomic_sequence"].str.upper()
        seqs["mitochondrial_sequence"]=seqs["mitochondrial_sequence"].str.upper()
        def identity(row)->float:
            Gseq=list(row["genomic_sequence"])
            Mtseq=list(row["mitochondrial_sequence"])
            sequences=pd.DataFrame(columns=["G","Mt"])
            sequences["G"]=Gseq
            sequences["Mt"]=Mtseq
            return (sequences["G"]==sequences["Mt"]).astype(int).sum()/sequences.shape[0]
        seqs["Sequence identity"]=seqs.apply(identity,axis=1)
        SpeciesFull=seqs["id"].str.split("_").str[:2].str.join("_")
        seqs["SpeciesShort"]=SpeciesFull.str[:2]+" "+SpeciesFull.str.split("_").str[1].str[:2]
        return seqs

    def boxplot(self,Compdf:pd.DataFrame,orgs:list,y_name:str,ax)->None:
        sns.boxplot(
                data=Compdf,x="SpeciesShort",y=y_name,
                ax=ax,showfliers=False,hue="SpeciesShort",
                palette=["lightblue","orange"],
                width=.4,order=Compdf["SpeciesShort"].unique()
            )
        ax.set(ylabel=y_name,xlabel="Species")

    def regplot(self,Regdf:pd.DataFrame,color:str,ax)->None:
        sns.regplot(
                data=Regdf,x="genomic_size",y="NUMT size (bp)",
                ax=ax,color=color
            )
        ax.set(xlabel="Size of genome part (Mb)",ylabel="Cumulative NUMT size (bp)")

    def histplot(self,Compdf:pd.DataFrame,org:str,color:str,MtSizes:pd.Series,ax)->None:
        sns.histplot(
            np.concatenate(
                Compdf[Compdf["SpeciesFull"]==org].apply(
                    lambda row:
                    np.arange(start=row["mitochondrial_start"],stop=(row["mitochondrial_start"]+row["mitochondrial_length"]),step=10,dtype=int),axis=1
                ).values
            ),
            bins=200,element="step",color=color,ax=ax
        )
        ax.set_xticks(ticks=np.arange(start=0,stop=MtSizes[org],step=2000))
        ax.set_xticklabels(np.arange(start=0,stop=MtSizes[org],step=2000),rotation=45)
        ax.set_xlabel("Mitochondrial nucleotides")

    def heatmap(self,orgs:list,Compdf:pd.DataFrame,ax)->None:
        k=3
        nucleotides=list('ACGT')
        kmers=[''.join(nucleotide) for nucleotide in product(nucleotides, repeat=k)]
        sequences=pd.read_csv("genomic_sequences.csv")
        sequences=Compdf.join(sequences.set_index("id"),on="id")[["id","genomic_sequence"]]
        sequences=sequences[
            (sequences["id"].str.contains(orgs[0]))
            |(sequences["id"].str.contains(orgs[1]))
        ]
        colors=sequences["id"].str.contains(orgs[0]).replace([True,False],["lightblue","orange"])
        sequences["genomic_sequence"]=sequences["genomic_sequence"].str.upper().str.replace("-","")
        def getKmers(sequence):
            kmerCounts=[]
            for kmer in kmers:
                kmerCounts.append(sequence.count(kmer))
            return (pd.Series(kmerCounts)/len(sequence)).values
        kmer_counts=sequences["genomic_sequence"].apply(getKmers).tolist()
        distances=pairwise_distances(kmer_counts)
        sns.heatmap(
                1-distances,
                cmap="coolwarm",
                ax=ax,
                cbar_kws={"orientation": "horizontal","shrink":.5,"label":"K-mer based similarity"}
            )
        ax.set_xticks(ticks=[],labels=[])
        ax.set_yticks(ticks=[],labels=[])
        for i, color in enumerate(colors):
            ax.add_patch(plt.Rectangle(xy=(-0.03, i), width=0.025, height=1, color=color, lw=0,
                                       transform=ax.get_yaxis_transform(), clip_on=False))
            ax.add_patch(plt.Rectangle(xy=(i, 1.03), width=1, height=0.05, color=color, lw=0,
                                       transform=ax.get_xaxis_transform(), clip_on=False))

