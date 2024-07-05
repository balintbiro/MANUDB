import streamlit as st
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
#############################
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
st.divider()
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
st.divider()
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
st.divider()
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