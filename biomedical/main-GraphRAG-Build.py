#%%
import os
import logging
import ollama
import numpy as np
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer


from biomedical.llm_utils import *

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# Set WORKING_DIR
# os.chdir(r"/root/projects/nano-graphrag/biomedical")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

WORKING_DIR = "./cache_groqLLAMA4scout_biobert_bioprompt_20Results_TEST"
WORKING_DIR = "./cache_groqLLAMA4scout_openaiembed_bioprompt_20Results_TEST"
WORKING_DIR = "./cache_groqllama4_openaiemb_50Results"  # For testing purposes, use a dummy cache directory
WORKING_DIR = "./GAGA"  # For testing purposes, use a dummy cache directory


##### Choose the embedding model #####
# embedder = openai_embedding
# embedder = ollama_embedding
embedder = local_embedding

#### Choose the model to use for the RAG #####
USE_LLM = groq_model_if_cache
# USE_LLM = ollama_model_if_cache
# USE_LLM = deepseepk_model_if_cache

# ---> SET paremters in biomedical/llm_utils.py


#%%



def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def insert(TEXT):
    from time import time
    #remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    #remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    #remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    #remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    #remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=USE_LLM,
        cheap_model_func=USE_LLM,
        embedding_func=embedder,
    )
    start = time()
    rag.insert(TEXT)
    print("indexing time:", time() - start)


##### LOAD DATASET #####
import pandas as pd
# df = pd.read_csv("datasets/fulltext_dataset.zip")
df = pd.read_csv("datasets/chunks/0-3000pmc_fulltext.csv")

# Filter Dataset
df = df[~((df['INTRO'].isna() & df['METHODS'].isna() & df['RESULTS'].isna() & df['DISCUSS'].isna()))]
df = df[~((df['RESULTS'].isna() & df['DISCUSS'].isna()))].reset_index(drop=True)
df = df[~(df['RESULTS'].isna())].reset_index(drop=True)
df.fillna("", inplace=True)

df.text = df["RESULTS"] + "\n\n\n\n" + df["DISCUSS"]
df.text = df["RESULTS"]
df.text = df.text.str.replace("<SEP>","\n\n")
# df.to_csv("datasets/halftext_dataset5000.csv", index=False)


#%%

########## RUN THE JOB ##########
start_id = 46
batch_size = 1
import time

t1 = time.time() 

if __name__ == "__main__":
    #insert(TEXT)
    #insert(df.text[1])
    #insert("""  """)
    
    for i in range(start_id, start_id+batch_size):
        if len(df.text[i]) > 10: 
            # print(f"""Inserting text of len {len(df.text[i])}""")
            insert(df.text[i])
            print(f"""\n\n<<<<<<<<<<<<<<<<<<<<<  {i+1}/{batch_size+start_id}  >>>>>>>>>>>>>>>>>>\n\n """)
        else:
            print("\n\n\n\n\nNo text...\n\n\n\n\n")
        
        print("Sleeping...")
        time.sleep(10)
    
print(f"\033[94mBatch runtime: [{round(time.time() - t1, 2)}s]\033[0m")



from biomedical.graphml_visualize import CreateGraphVisualization

CreateGraphVisualization(WORKING_DIR.split("/")[-1])



#%%

# Nota: Inserire Risultati e Discussione insieme fa bloccare il batch 

# InternalServerError: Error code: 503 - {'error': {'message': 'Service Unavailable', 'type': 'internal_server_error'}}

# Limite Grow : Error code: 429 - {'error': {'message': 'Rate limit reached for model `meta-llama/llama-4-scout-17b-16e-instruct` in organization `org_01jsh6rb0jfm88d0qyz3b7zr1k` service tier `on_demand` on tokens per minute (TPM): Limit 300000, Used 320646, Requested 3149. Please try again in 4.759s. Need more tokens? Visit https://groq.com/self-serve-support/ to request higher limits.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
    
#%%
