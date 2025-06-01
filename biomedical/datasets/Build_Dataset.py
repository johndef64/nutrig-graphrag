#%%
import pandas as pd 

df = pd.read_csv("RAG_LLM_nutrigentic_dataset.csv")
print(df.columns)
df.abstract[0:1000]#.to_csv("/workspace/code/AI_BIO_ws_paper_giovanni/nano-graphrag/biomedical/abstract1000.txt")



# %%# %%

# %%




#%%

# # Import necessary libraries for data manipulation, API requests, and progress bar
# import pandas as pd
# import requests
# import json
# from tqdm import tqdm

# # Load the dataset into a pandas DataFrame
# df = pd.read_csv("RAG_LLM_nutrigentic_dataset.csv")

# # Define a function to convert PMID to PMCID using NCBI E-utilities
# def pmid_to_pmcid(pmid):
#     # Step1: Convert the PMID to PMCID using NCBI E-utilities
#     esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
#     params = {
#         "db": "pmc",
#         "term": f"{pmid}[pmid]",
#         "retmode": "json"
#     }
#     try:
#         # Send a GET request to the E-utilities API
#         r = requests.get(esearch_url, params=params)
#         # Parse the JSON response
#         pmcids = r.json()["esearchresult"]["idlist"]

#         # Check if a PMCID was found
#         if not pmcids:
#             return None
#         else:
#             # Return the first PMCID found
#             return pmcids[0]
#     except Exception as e:
#         # Handle any exceptions that occur during the request or parsing
#         print(f"An error occurred for PMID {pmid}: {e}")
#         return None

# # Apply the pmid_to_pmcid function to each 'pubmed_id' in the DataFrame with a progress bar
# df['pmcid'] = [pmid_to_pmcid(str(pmid)) for pmid in tqdm(df['pubmed_id'], desc="Converting PMIDs to PMCIDs")]

# # Create a new column to indicate if a PMCID was found
# df['pmcid_found'] = df['pmcid'].notnull()

# # Create a new DataFrame with 'pubmed_id', 'pmcid', and 'pmcid_found' for boolean mapping
# bool_mapping_df = df[['pubmed_id', 'pmcid', 'pmcid_found']].copy()

# # Rename 'pmcid_found' to a more descriptive name if needed
# bool_mapping_df = bool_mapping_df.rename(columns={'pmcid_found': 'has_pmcid'})

# # Display the resulting DataFrame
# print(bool_mapping_df)
# bool_mapping_df.to_csv("pubmed_pmcid_mapping.csv", index=False)




# %%
import pandas as pd 

df = pd.read_csv("RAG_LLM_nutrigentic_dataset.csv")
# Import necessary libraries
import requests

# Define the function to get data from PubMed Central API
def get_pmcoa_data(article_id):
    BASE_URL = f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{article_id}/unicode'
    print(BASE_URL)
    try:
        response = requests.get(BASE_URL)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle potential errors (e.g., network issues, invalid article_id)
        print(f"Failed to get data for article ID {article_id}: {e}")
        return None

# Example usage:
# Replace 'PMC7000000' with your desired article ID
article_id = "10578010" #pmcid

import time


# for article_id in df.pubmed_id[0:30]:
#     data = get_pmcoa_data(article_id)
#     if data is None:
#         print(f"Failed to retrieve data for article ID {article_id}")
#     else:

#         # Define a list of section types we're interested in
#         target_section_types = ["INTRO", "RESULTS", "DISCUSSION"]

#         # Extract the passages directly to simplify the nested access
#         passages = data[0]["documents"][0]["passages"]

#         for passage in passages:
#             if passage["infons"]["section_type"] in target_section_types:
#                 print(passage["text"])
#                 print(passage["infons"]["section_type"])

#         print(len(data[0]["documents"][0]["passages"]))
#         time.sleep(0.2)

#%%
import pandas as pd
import time

# Initialize an empty list to store the combined texts
combined_texts = []


# Loop through each article ID
for article_id in df.pubmed_id[0:100]:
    data = get_pmcoa_data(article_id)

    if data is None:
        print(f"Failed to retrieve data for article ID {article_id}")
        combined_texts.append(" ")  # Add "missing" for failed retrievals
        continue  # Skip to next iteration

    # Initialize a list to store texts for current article
    article_texts = []

    # Define target section types
    target_section_types = ["INTRO", "RESULTS", "DISCUSSION"]

    # Extract passages
    passages = data[0]["documents"][0]["passages"]

    # Process each passage
    for passage in passages:
        # Check if section type is in target sections
        if passage["infons"]["section_type"] in target_section_types:
            article_texts.append(passage["text"])

    # Join all texts for the current article
    combined_text = " ".join(article_texts)

    # Append to the combined texts list
    combined_texts.append(combined_text)

    time.sleep(0.2)

#%%
# Create a DataFrame with pubmed_id and merged text
result_df = pd.DataFrame({
    "pubmed_id": df.pubmed_id[0:100],
    "text": combined_texts
})
result_df["words"] = result_df.text.apply(lambda x: len(x.split(" ")))

# Output the resulting DataFrame
print(result_df)


result_df.to_csv("100pmc_fulltext.csv", index=False)