#%%
import os
import pandas as pd 
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


# Import Dataset

os.chdir("/root/projects/nano-graphrag/biomedical/datasets")
df = pd.read_csv("RAG_LLM_nutrigentic_dataset.csv")
print(len(set(df["pubmed_id"])))
print(df.columns)
df.abstract[0:1000]#.to_csv("/workspace/code/AI_BIO_ws_paper_giovanni/nano-graphrag/biomedical/abstract1000.txt")




#%%
df = pd.read_parquet("grpm_nutrigen_int.parquet")

# print(len(set(df.pmid)), set(df.topic))

subdf = df[df.topic.isin([
    'Eating Behavior and Taste Sensation',
    #'Diet-induced Oxidative Stress',
    #'General Nutrition', 
    'Food Allergies', 
    'Vitamin and Micronutrients Metabolism and Deficiency-Related Diseases',
    #'Diabetes Mellitus Type II and Metabolic Syndrome',
    'Obesity, Weight Control and Compulsive Eating',
    #'Cardiovascular Health and Lipid Metabolism',
    #'Xenobiotics Metabolism', 
    'Food Intolerances'
  ])]
print(len(set(subdf.pmid)))
#%%
subdf.pmid.to_list()[0]
#%%
pmids = pd.DataFrame(subdf.pmid.to_list(), columns=["pubmed_id"])
pmids = pmids.drop_duplicates()
pmids.to_csv("pubmed_id.csv", index=False)

df = pd.read_csv("pubmed_id.csv")
len(df.pubmed_id) // 10
#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import requests
import pandas as pd
import time

df = pd.read_csv("pubmed_id.csv")

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


# Initialize an empty list to store the combined texts
articles_data = []
starting_index = 0
batch_size = 2000

# Loop through each article ID
for article_id in df.pubmed_id[starting_index:starting_index+batch_size]:
    data = get_pmcoa_data(article_id)

    if data is None:
        print(f"Failed to retrieve data for article ID {article_id}")
        # Append empty strings for each section
        articles_data.append({
            "INTRO": " ",
            'METHODS': " ",
            "RESULTS": " ",
            "DISCUSS": " "
        })
        continue  # Skip to next iteration

    # Initialize a dictionary to store sections for current article
    article_sections = {"INTRO": "", 'METHODS': "", "RESULTS": "", "DISCUSS": ""}

    # Define target section types
    target_section_types = ["INTRO", 'METHODS', "RESULTS", "DISCUSS"]

    # Extract passages
    passages = data[0]["documents"][0]["passages"]

    # Process each passage
    for passage in passages:
        section_type = passage["infons"]["section_type"]
        if section_type in target_section_types:
            article_sections[section_type] = passage["text"]

    # Append the sections dictionary to the list
    articles_data.append(article_sections)

    time.sleep(0.2)

# Create DataFrame with separate columns for each section
df_sections = pd.DataFrame(articles_data)


#%%
# Create a DataFrame with pubmed_id and merged text
result_df = pd.DataFrame({
    "pubmed_id": df.pubmed_id[0:batch_size],
    "INTRO": df_sections.INTRO[0:batch_size],
    "METHODS": df_sections.METHODS[0:batch_size],
    "RESULTS": df_sections.RESULTS[0:batch_size],
    "DISCUSS": df_sections.DISCUSS[0:batch_size]
})
#result_df["words"] = result_df.text.apply(lambda x: len(x.split(" ")))

# Output the resulting DataFrame
print(result_df)


result_df.to_csv(f"{starting_index}-{starting_index+batch_size}pmc_fulltext.csv", index=False)
result_df

# %%




# %% Sope section tyeps
data = get_pmcoa_data(df.pubmed_id[200])
i = 10
data[0]["documents"][0]["passages"][i]["infons"]["section_type"]

# Initialize an empty list to store section types
section_types = []

# Iterate through each passage in the documents
for passage in data[0]["documents"][0]["passages"]:
    # Check if 'infons' exists in the passage and 'section_type' is present
    if 'infons' in passage and 'section_type' in passage['infons']:
        section_types.append(passage['infons']['section_type'])

set(section_types)
# %%


import os
import pandas as pd

# Function to load and concatenate datasets
def load_concat_datasets(folder_path):
    # List all files in the folder
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Load each file into a DataFrame and store in a list
    dfs = []
    for file in file_names:
        print(file)
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Concatenate all DataFrames
    concatenated_df = pd.concat(dfs, ignore_index=True)

    return concatenated_df

# Example usage
folder_path = "/root/projects/nano-graphrag/biomedical/datasets"
df = load_concat_datasets(folder_path)
df.to_csv("fulltext_dataset.csv", index=False)
df
# %%
