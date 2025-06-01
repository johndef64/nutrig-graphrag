
#%%
####################################################################
##########################  ADD LABELS ############################

import pandas as pd
import os
os.chdir("/root/projects/nano-graphrag/biomedical/datasets")

#%%
grpm = pd.read_parquet("grpm_nutrigen_int.parquet")
df = pd.read_csv("fulltext_dataset.zip")

def filter_dataset(df):
    # Calcola la somma della lunghezza dei valori in ogni riga
    df['text_length'] = df.apply(lambda row: sum(len(str(value)) for value in row), axis=1)

    # Filtra le righe dove la somma e' minore di 20
    filtered_df = df[df['text_length'] >= 20]#.drop(columns=['row_length'])

    return filtered_df

# Esempio di utilizzo
# df = pd.read_csv("your_dataset.csv")  # Carica il tuo dataset
filtered_df = filter_dataset(df)
display(filtered_df[["RESULTS", "DISCUSS"]])#.to_csv("filtered_dataset_resdisc.csv", index=False)

#%%
grpm.topic.unique()

# Creazione della colonna 'TOPIC_ABBR' con abbreviazioni in maiuscolo e stessa lunghezza

# Dizionario di abbreviazioni a 6 caratteri
abbr_dict = {
    'General Nutrition': 'GENUTR',
    'Obesity, Weight Control and Compulsive Eating': 'OBWCCE',
    'Diabetes Mellitus Type II and Metabolic Syndrome': 'DIAMSY',
    'Cardiovascular Health and Lipid Metabolism': 'CARDML',
    'Vitamin and Micronutrients Metabolism and Deficiency-Related Diseases': 'VITMIN',
    'Eating Behavior and Taste Sensation': 'EATBTS',
    'Food Intolerances': 'FOODIN',
    'Food Allergies': 'FOODAL',
    'Diet-induced Oxidative Stress': 'DIOXST',
    'Xenobiotics Metabolism': 'XENOBI'
}

# Assumo che il DataFrame si chiami 'grpm'
grpm['TOPIC_LABEL'] = grpm['topic'].map(abbr_dict)
grpm
# %%

# you task is to merge  
grpm[['pmid', 'topic', 'TOPIC_LABEL']]

# on df.pubmed_id, add to the df as pricipal datraframe
grpm.pmid = grpm.pmid.astype(int)
# Merge based on PMID/pubmed_id and add topic info to df
new_df = df.merge(
    grpm[['pmid', 'topic', 'TOPIC_LABEL']],
    left_on='pubmed_id', right_on='pmid',
    how='left'
)

# Optionally drop 'pmid' if not needed after the merge
new_df = new_df.drop(columns=['pmid', "topic"])
new_df =  new_df.drop_duplicates()

label_df = new_df[['pubmed_id', 'TOPIC_LABEL']].drop_duplicates()

#%%

label_df = label_df[label_df.TOPIC_LABEL.isin(['OBWCCE', "EATBTS", 'VITMIN', 'FOODIN',  'FOODAL'])]

label_df.to_csv("dataset_category.csv", index=False)

#%%
label_df = pd.read_csv("dataset_category.csv")
print(filtered_df[["pubmed_id","text_length"]].head())
# print(label_df.head())

import pandas as pd

# Supponendo che siano già definiti filtered_df e label_df come nel tuo esempio
grouped_labels = label_df.groupby('pubmed_id')['TOPIC_LABEL'].apply(list).reset_index()
filtered_df = filtered_df.merge(grouped_labels, on='pubmed_id', how='left')
filtered_df['TOPIC_LABEL'] = filtered_df['TOPIC_LABEL'].apply(lambda x: x if isinstance(x, list) else [])
filtered_df
# %%
from biomedical.plots import *
import pandas as pd
import matplotlib.pyplot as plt


plot_topic_label_distribution(filtered_df)
plot_gaussian_length_distribution(filtered_df)
# %%

filtered_df
#%%

# filtered_df = pd.read_csv("fulltext_dataset_labeled.csv")

# filtered_df = filtered_df[['pubmed_id', 'INTRO', 'METHODS', 'RESULTS', 'DISCUSS', 'text_length',
#         'TOPIC_LABEL']]
# filtered_df.to_csv("fulltext_dataset_labeled.csv")
filtered_df = pd.read_csv("fulltext_dataset_labeled.csv", index_col=0)
#%%
filtered_df
#%%
###################################################################################
##########################  FILTERING FOR LENGTH ############################
def drop_na_results(df, verbose=False):
    before = len(df)
    cleaned_df = df.dropna(subset=['RESULTS']).reset_index(drop=True)
    after = len(cleaned_df)
    if verbose:
        print(f"{before - after} righe con NaN in 'RESULTS' eliminate.")
    return cleaned_df

def filter_results_length(df, length=10):
    # Tiene solo le righe in cui la lunghezza di RESULTS è >= 10
    mask = df['RESULTS'].astype(str).apply(len) >= length
    filtered_df = df[mask].reset_index(drop=True)
    return filtered_df

def filter_get_short_results(df, length=10):
    # Tiene solo le righe in cui la lunghezza di RESULTS è >= 10
    mask = df['RESULTS'].astype(str).apply(len) <= length
    filtered_df = df[mask].reset_index(drop=True)
    return filtered_df

dropnares = drop_na_results(filtered_df)
filter4length = filter_results_length(dropnares, 200)
filter4length = filter_get_short_results(filter4length, 30000)

shorts = filter_get_short_results(dropnares, 200)
longs = filter_results_length(dropnares, 50000)
# %%
filter4length
filtered_df
longs
# %%
plot_topic_label_distribution(filter4length)
plot_gaussian_length_distribution(filter4length)
filter4length
# %%

###################################################################################
##########################  FILTERING FOR RS AND GENES ############################
genes = grpm.gene.unique().tolist()

import pandas as pd
import re

# Pattern per identificare rsID variabili in formato rs seguito da almeno 3 cifre (generalizzato)
pattern_rsid = re.compile(r'rs\d{3,}', re.IGNORECASE)
# Pattern per identificare varianti in formato cromosoma:posizione (es. 1:12345678)
pattern_chr_pos = re.compile(r'\b(?:chr)?\d{1,2}:\d+\b', re.IGNORECASE)
# Pattern per identificare varianti in formato cromosoma-posizione-ref-alt (es. 7-140453136-A-T oppure chr7-140453136-A-T)
pattern_chr_pos_ref_alt = re.compile(r'\b(?:chr)?\d{1,2}[-:]\d+[-:][ACGT]+[-:][ACGT]+\b', re.IGNORECASE)
# Pattern per le varianti HGVS del DNA genomico (es. NM_000546.5:c.215C>G oppure NC_000023.11:g.32867801T>A)
pattern_hgvs_genomic = re.compile(
    r'\b[N][C][_\d]+\.\d+:[g]\.\d+(?:_\d+)?[ACGT]+>[ACGT]+\b', re.IGNORECASE)
pattern_hgvs_coding = re.compile(
    r'\b[NMP][A-Z]_\d+\.\d+:[c]\.\d+[ACGT]>[ACGT]\b', re.IGNORECASE)
# Pattern per varianti aminoacidiche (es. p.Gly12Asp, p.G12D)
pattern_protein = re.compile(
    r'\bp\.(?:[A-Z][a-z]{2}\d{1,4}[A-Z][a-z]{2}|[A-Z]\d{1,4}[A-Z])\b', re.IGNORECASE)

# Elenco dei pattern utili per il matching nei testi biomedici
genetic_variant_patterns = [
    pattern_rsid,
    pattern_chr_pos,
    pattern_chr_pos_ref_alt,
    pattern_hgvs_genomic,
    pattern_hgvs_coding,
    pattern_protein
]

def any_variant_match(text):
    # Questa funzione restituisce True se almeno uno dei pattern trova un match nel testo
    return any(pat.search(text) for pat in genetic_variant_patterns)

def filter_row(row):
    results = str(row['RESULTS'])
    # Controlla se c'è la wildcard rs+3 numeri
    # match_variant = bool(pattern_rs.search(results))
    match_variant = any_variant_match(results)
    # Controlla se c'è almeno un gene nella lista genes
    match_genes = any(gene in results for gene in genes)
    # Mantieni la riga se almeno una delle due condizioni è vera
    return match_variant or match_genes

def filter_row_rsid(row):
    results = str(row['RESULTS'])
    # Controlla se c'è la wildcard rs+3 numeri
    # match_variant = bool(pattern_rs.search(results))
    match_variant = any_variant_match(results)
    # Mantieni la riga se almeno una delle due condizioni è vera
    return match_variant 

# Assumi che filtered_df sia già definito e contenga la colonna RESULTS

df = filtered_df
df = filter4length
filteredRESULTS_df = df[df.apply(filter_row, axis=1)]
filteredRESULTS_df_rsid = df[df.apply(filter_row_rsid, axis=1)]
#%%
# Visualizza il risultato
filteredRESULTS_df
# filteredRESULTS_df_rsid

#%%
plot_topic_label_distribution(filteredRESULTS_df)
plot_gaussian_length_distribution(filteredRESULTS_df)

#%%

################## WORING DATASET SELECTION ##################
################## WORING DATASET SELECTION ##################
################## WORING DATASET SELECTION ##################
from biomedical.plots import *
import pandas as pd
import matplotlib.pyplot as plt
filteredRESULTS_df_rsid = pd.read_csv("filterd_results_dataset.csv", index_col=0)
filteredRESULTS_df_rsid['TOPIC_LABEL'] = filteredRESULTS_df_rsid['TOPIC_LABEL'].apply(ast.literal_eval)

plot_topic_label_distribution(filteredRESULTS_df_rsid)
plot_gaussian_length_distribution(filteredRESULTS_df_rsid)
filteredRESULTS_df_rsid#.to_csv("filterd_results_dataset.csv")

# %%
plot_topic_label_and_gaussian_length_distribution(filteredRESULTS_df_rsid)


# %%

expanded_df = filteredRESULTS_df_rsid.explode('TOPIC_LABEL')
expanded_df
# %%
# print(expanded_df[['pubmed_id', 'TOPIC_LABEL']].to_csv())

import pandas as pd

# carica il dataframe dal file CSV
df = expanded_df[['pubmed_id', 'TOPIC_LABEL']]

# seleziona 200 casuali per ogni TOPIC_LABEL, consentendo duplicazioni di pubmed_id tra label diverse
sampled_df = (
    df.groupby('TOPIC_LABEL', group_keys=False)
    .apply(lambda x: x.sample(n=200, random_state=92522).reset_index(drop=True)
           )
)    
sampled_df
# salva il risultato su un nuovo file
#sampled_df.to_csv('sampled_200_per_topic.csv', index=False)

# sampled_df['pubmed_id'].drop_duplicates()

#%%
# sampled_df = df.groupby('TOPIC_LABEL', group_keys=False).apply(lambda x: x.sample(n=200, random_state=42)).reset_index(drop=True)
# (40873, 980),
# (92522, 982),


# df[df.TOPIC_LABEL == "OBWCCE"]

# Questo codice garantisce che in sampled_df:
# 1. Ogni label sia rappresentata da 200 elementi
# 2. Nessun pubmed_id sia ripetuto (nessuna duplicazione nemmeno tra label diverse)

# Crea un dizionario {label: df_label}
label_dfs = {label: group.copy() for label, group in df.groupby("TOPIC_LABEL")}

# Prepara set per evitare duplicati
used_pubmed_ids = set()

# Campiona 200 elementi unici per ogni label

num = 20
final_samples = []
for label, group in label_dfs.items():
    available = group[~group["pubmed_id"].isin(used_pubmed_ids)]
    sample = available.sample(n=min(num, len(available)), random_state=92522)
    used_pubmed_ids.update(sample["pubmed_id"])
    final_samples.append(sample)

# Combina i risultati
sampled_df = pd.concat(final_samples).reset_index(drop=True)

sampled_df['pubmed_id'].drop_duplicates()
sampled_df
#plot_topic_label_and_gaussian_length_distribution(sampled_df)

working_dataset = filteredRESULTS_df_rsid[filteredRESULTS_df_rsid.pubmed_id.isin(sampled_df['pubmed_id'])]
working_dataset.to_csv(f"working_dataset_{str(num*5)}.csv", index=False)
# plot_topic_label_and_gaussian_length_distribution(working_dataset)
working_dataset

# %%
plot_topic_label_and_gaussian_length_distribution(working_dataset)
working_dataset.pubmed_id.nunique()
# %%
