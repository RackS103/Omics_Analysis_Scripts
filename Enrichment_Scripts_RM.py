# Rac Mukkamala, White Lab
# List of functions to help with pathway enrichment
# these functions connect to Enrichr, STRING, and KEA3 web apis to do analysis
# and download results to your computer.

import requests
import pandas as pd
import numpy as np
import json
import time


def run_Enrichr(gene_set, databases, desc="", p_cutoff=0.05):
    """
    This function connects to the Enrichr webserver and runs enrichment analysis on the 
    inputted gene set using the inputted list of databases. It returns a Pandas dataframe
    containing all of the enrichment scores and details for all databases.

    PARAMETERS:
    - gene_set (list): List of genes to run enrichment analysis on. 
    - desc (str, Default=''): Optional: Description of this analysis run to input into Enrichr
    - databases (list): List of enrichment databases (EX: "KEGG_2021_Human") to run pathway enrichment on.
    - p_cutoff (float, default=0.05): Optional: Adjusted p-value significance cutoff.

    RETURNS:
    - Pandas DataFrame containing the list of significant pathways and all relevant scoring info from Enrichr.
    - The DataFrame will have compiled results from all of the databases, and will filter for pathways that
    meet p_cutoff.
    """
    ENRICHR_ADD_URL = 'https://maayanlab.cloud/Enrichr/addList'
    ENRICHR_ENRICH_URL = 'https://maayanlab.cloud/Enrichr/enrich?userListId=%s&backgroundType=%s'

    gene_str = '\n'.join(gene_set)
    payload = {'list': gene_str, 'description': desc}
    r1 = requests.post(ENRICHR_ADD_URL, files=payload)
    ids = r1.json()

    enrichment_total = None
    for db in databases:
        r2 = requests.get(ENRICHR_ENRICH_URL % (ids['userListId'], db))

        columns=['Rank', 'Term_Name', 'P_Value', 'Z_Score', 'Combined_Score', 'Overlapping_Genes', 'Adj_P_Value', 'Old_P_Value', 'Old_Adj_P_Value']
        enrichment = pd.DataFrame(r2.json()[db], columns=columns)
        enrichment.drop(labels=['Old_P_Value', 'Old_Adj_P_Value', 'Rank'], axis=1, inplace=True)
        enrichment['Database'] = np.repeat(db, enrichment.shape[0])
        enrichment = enrichment[enrichment['Adj_P_Value'] <= p_cutoff]

        if enrichment_total is None:
            enrichment_total = enrichment
        else:
            enrichment_total = pd.concat([enrichment_total, enrichment], axis=0, join='outer')
    
    enrichment_total.sort_values(by='Adj_P_Value', inplace=True)
    enrichment_total['Rank'] = np.arange(1, enrichment_total.shape[0]+1)
    enrichment_total.set_index('Rank', inplace=True)
    return enrichment_total

#example code
#print(run_enrichr(gene_set=['CD11b', 'CD45', 'CD163', 'CD206', 'CD86'], databases=['KEGG_2021_Human', 'WikiPathway_2021_Human', 'Reactome_2016'], desc= 'Test101'))

def run_KEA3(gene_set, desc=""):
    """
    This function connects to the KEA3 webserver and runs kinase enrichment analysis on the 
    inputted gene set using the inputted list of databases. It returns a tuple of Pandas dataframes
    containing all of the KEA scores.

    PARAMETERS:
    - gene_set (list): List of genes to run enrichment analysis on. 
    - desc (str, Default=''): Optional: Description of this analysis run to input into Enrichr

    RETURNS:
    - Tuple containing two Pandas DataFrames. The first dataframe in the tuple will have the KEA3 toprank score list, and
    the second will have the KEA3 meanrank scores.
    """
    KEA3_URL = "https://maayanlab.cloud/kea3/api/enrich/"
    payload = {
        'gene_set': gene_set,
        'query_name': desc
    }
    request = requests.post(KEA3_URL, data=json.dumps(payload))
    with open('test_kea3.json', 'w') as json_file:
        json.dump(request.json(), json_file)

    toprank = json.dumps(request.json()['Integrated--topRank'])
    toprank = pd.read_json(toprank)

    meanrank = json.dumps(request.json()['Integrated--meanRank'])
    meanrank = pd.read_json(meanrank)
    print('KEA3 complete')

    return (toprank, meanrank)

#print(run_KEA3(['SRC', 'ERK', 'MEK', 'EGFR', 'KRAS', 'MAPK'], 'testrun1')[1])

def run_STRING(protein_list, species_id, img_filepath):
    """
    This function connects to the STRING webserver and runs protein interaction analysis on the 
    inputted protein set. It will save the network image to the inputted filepath and returns nothing.

    PARAMETERS:
    - protein_list (list): List of proteins to run STRING analysis on. 
    - species_id (int): The STRING species ID for the species of interest (ex: 9606 for human)
    - img_filepath (str): Path to save the image to, do not include the file extension here! (.png)

    RETURNS:
    None
    """
    #first convert all names into STRING approved names
    STRING_MAPPING_URL = 'https://string-db.org/api/json/get_string_ids'
    params = {
        "identifiers" : "\r".join(protein_list), # your protein list
        "species" : species_id, # species NCBI identifier 
        "limit" : 1, # only one (best) identifier per input protein
        "echo_query" : 0, # see your input identifiers in the output
        "caller_identity" : "STRING_enrichment_script_RM" # your app name
    }

    request_mapping = requests.post(url=STRING_MAPPING_URL, data=params)
    preferred_names_list = pd.read_json(request_mapping.text)['preferredName'].to_list()

    #generate STRING image
    STRING_IMAGE_URL = 'https://string-db.org/api/highres_image/network'
    params = {
        "identifiers" : "\r".join(preferred_names_list), # your protein list
        "species" : species_id, # species NCBI identifier 
        "caller_identity" : "STRING_enrichment_script_RM", # your app name
        "add_white_nodes": 10,
        "network_flavor": "confidence",
        "hide_disconnected_nodes": 1,

    }

    response_img = requests.post(url=STRING_IMAGE_URL, data=params)
    with open(f'{img_filepath}.png', 'wb') as output:
        output.write(response_img.content)

    print(f'Generated STRING network image and saved it to {img_filepath}.png')
    time.sleep(1)


#run_STRING(['SRC', 'ERK', 'MEK', 'EGFR', 'KRAS', 'MAPK'], 10090, 'testimg')