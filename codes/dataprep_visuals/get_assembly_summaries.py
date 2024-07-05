import requests
import pandas as pd
from subprocess import call

genomic_ids=pd.read_csv('../data/ncbi_numts_full_set.csv',usecols=['genomic_id','organism_name'])
genomic_ids=genomic_ids[genomic_ids['genomic_id'].str.contains('NC_')].drop_duplicates('organism_name')

def get_assembly_stats(nc_id):
	try:
	    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
	    esearch_url = f"{base_url}esearch.fcgi?db=nucleotide&term={nc_id}&retmode=json"
	    esearch_response = requests.get(esearch_url)
	    esearch_data = esearch_response.json()
	    uid = esearch_data['esearchresult']['idlist'][0]
	    esummary_url = f"{base_url}esummary.fcgi?db=nucleotide&id={uid}&retmode=json"
	    esummary_response = requests.get(esummary_url)
	    esummary_data = esummary_response.json()
	    assembly_accession = esummary_data['result'][uid]['assemblyacc']
	    esearch_url = f"{base_url}esearch.fcgi?db=assembly&term={assembly_accession}&retmode=json"
	    esearch_response = requests.get(esearch_url)
	    esearch_data = esearch_response.json()
	    assembly_uid = esearch_data['esearchresult']['idlist'][0]
	    esummary_url = f"{base_url}esummary.fcgi?db=assembly&id={assembly_uid}&retmode=json&report=full"
	    esummary_response = requests.get(esummary_url)
	    assembly_record = esummary_response.json()
	    aid=assembly_record['result']['uids'][0]
	    rep_url=assembly_record['result'][aid]['ftppath_assembly_rpt']
	    call(f'wget --directory-prefix=../data/assembly_reports/ {rep_url}',shell=True)
	except:
		pass

genomic_ids['genomic_id'].apply(get_assembly_stats)